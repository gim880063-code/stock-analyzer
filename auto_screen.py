"""
헤드리스 자동 스크리닝 — GitHub Actions cron 등 UI 없는 환경에서 매일 장 마감 후
스크리닝을 한 사이클 돌리고 scouted/screening_history/score_history 를 갱신.

app.py 의 _screen_worker 와 동일한 흐름을 Streamlit 의존성 없이 재구성:
  1. 유니버스 종목 코드 조회
  2. lite=True 로 1단계 분석 (병렬, LLM 본문 분석 생략)
  3. stale 재무 + recent_surge 자동 제외
  4. min_score 필터링
  5. 통과 종목에 한해 deep analysis (LLM 본문 깊이 분석)
  6. screening_history + scouted 에 저장
  7. history.commit_batch 로 score_history.json 1회만 저장

환경변수:
  DART_API_KEY  — 필수 (DART 재무·공시)
  GEMINI_API_KEY — 선택 (없으면 룰 기반 공시 분류, 정확도 떨어짐)
  GITHUB_PAT + GIST_ID — 강력 권장 (Actions 컨테이너는 ephemeral. 없으면 결과 휘발)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from analyzer import (
    SCREEN_MIN_SCORE,
    analyze,
    enrich_with_deep_analysis,
    get_universe_codes,
    recompute_score_after_deep,
    select_observation_targets,
    select_adaptive_picks,
)
import cloud_store
import history
import scouted
import screening_history


def _log(msg: str) -> None:
    print(f"[auto_screen] {msg}", flush=True)


def _build_drop_reason(r: dict, min_score: int, kind: str) -> str:
    """app.py 의 _build_drop_reason 과 동일 컨셉 — 헤드리스용 단순 버전."""
    total = r.get("total", "?")
    name = r.get("name") or r.get("code", "?")
    if kind == "stale":
        return f"{name}: 재무 데이터 stale + 잠정실적 없음"
    if kind == "surge":
        triggers = (r.get("recent_surge") or {}).get("triggers") or []
        return f"{name}: 최근 급등 자동 제외 ({', '.join(triggers) or '임계 초과'})"
    if kind == "deep":
        return f"{name}: 정밀 분석 후 점수 {total}점이 기준 {min_score}점 미만"
    return f"{name}: 점수 {total}점이 기준 {min_score}점 미만"


def run(universe: str, min_score: int, deep: bool, workers: int) -> dict:
    started = time.time()
    _log(f"start — universe={universe} min_score={min_score} deep={deep} workers={workers}")

    # 프리플라이트: Gist 가 설정돼 있으면 '읽기'가 되는지 먼저 확인한다.
    # 못 읽는 상태에서 분석 후 저장하면 read-modify-write 가 빈 데이터로 원격을
    # 통째 덮어써 발굴·점수·스크리닝 이력이 날아갈 수 있다(과거 실제 유실 사례).
    if cloud_store.is_configured() and not cloud_store.refresh():
        _log("FATAL: Gist 현재 상태를 읽지 못함 — 데이터 유실 방지 위해 이번 실행 중단")
        return {"status": "error", "error": "gist preflight failed"}

    try:
        codes = get_universe_codes(universe)
    except Exception as e:
        _log(f"FATAL: 유니버스 종목 목록 실패: {type(e).__name__}: {e}")
        return {"status": "error", "error": str(e)}
    if not codes:
        _log("FATAL: 유니버스 종목 코드가 비어있음")
        return {"status": "error", "error": "empty universe"}

    _log(f"universe {len(codes)}개 종목 — 1단계 lite 분석 시작")

    # 1단계: lite 분석 (병렬)
    screened: list[dict] = []
    errors = 0
    history.begin_batch()
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(analyze, c, True, 0): c for c in codes}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                try:
                    r = fut.result()
                    if r.get("error"):
                        errors += 1
                    else:
                        screened.append(r)
                except Exception as e:
                    errors += 1
                    _log(f"  ! {c}: {type(e).__name__}: {e}")
                if i % 20 == 0 or i == len(codes):
                    _log(f"  stage1 {i}/{len(codes)} (errors={errors})")
    finally:
        try:
            history.commit_batch()
        except Exception as e:
            _log(f"history.commit_batch 실패: {e}")

    # 2단계 필터: stale + surge 자동 제외
    excluded_stale = 0
    excluded_surge = 0
    dropped_details: list[dict] = []
    fresh: list[dict] = []
    for r in screened:
        src = r.get("sources") or {}
        is_stale = (src.get("fin_freshness") or {}).get("is_stale", False)
        if is_stale and not r.get("preliminary"):
            excluded_stale += 1
            dropped_details.append({
                "code": r.get("code"), "name": r.get("name", r.get("code")),
                "total": r.get("total"), "close": r.get("last_close"),
                "reason": _build_drop_reason(r, min_score, kind="stale"),
            })
            continue
        if (r.get("recent_surge") or {}).get("is_surge"):
            excluded_surge += 1
            dropped_details.append({
                "code": r.get("code"), "name": r.get("name", r.get("code")),
                "total": r.get("total"), "close": r.get("last_close"),
                "reason": _build_drop_reason(r, min_score, kind="surge"),
            })
            continue
        fresh.append(r)

    # 하락장 리스크오프 — KOSPI가 200일선 아래면 진입 기준을 높여 추세 역행 매수를 줄임
    from analyzer import market_regime_state, effective_min_score
    _base_min = min_score
    regime = market_regime_state()
    min_score, _boost = effective_min_score(_base_min, regime=regime)
    if _boost:
        _log(f"방어 국면 감지({regime.get('label')}) — 진입 기준 {_base_min} → {min_score} 상향")

    fresh.sort(key=lambda r: r.get("total", -999), reverse=True)
    candidates = [r for r in fresh if r.get("total", -999) >= min_score]
    score_dropped = [r for r in fresh if r.get("total", -999) < min_score]
    for r in score_dropped:
        dropped_details.append({
            "code": r.get("code"), "name": r.get("name", r.get("code")),
            "total": r.get("total"), "close": r.get("last_close"),
            "reason": _build_drop_reason(r, min_score, kind="score"),
        })
    _log(
        f"stage1 결과: fresh={len(fresh)} (stale 제외 {excluded_stale}, "
        f"surge 제외 {excluded_surge}) → min_score≥{min_score} 통과 {len(candidates)}"
    )

    # 3단계: deep analysis (LLM 본문 분석)
    if candidates and deep:
        _log(f"stage2 deep analysis — {len(candidates)}개 종목")

        def _enrich(r: dict) -> None:
            enrich_with_deep_analysis(r, top_n=3)
            recompute_score_after_deep(r)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_enrich, r): r for r in candidates}
            done = 0
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    _log(f"  ! deep {futs[fut].get('code')}: {type(e).__name__}: {e}")
                done += 1
                if done % 10 == 0 or done == len(candidates):
                    _log(f"  stage2 {done}/{len(candidates)}")

        pre_filter = list(candidates)
        deep_dropped = [r for r in pre_filter if r.get("total", -999) < min_score]
        for r in deep_dropped:
            dropped_details.append({
                "code": r.get("code"), "name": r.get("name", r.get("code")),
                "total": r.get("total"), "close": r.get("last_close"),
                "reason": _build_drop_reason(r, min_score, kind="deep"),
            })
        candidates = [r for r in pre_filter if r.get("total", -999) >= min_score]
        candidates.sort(key=lambda r: r.get("total", -999), reverse=True)
        _log(f"stage2 결과: deep 후 min_score 통과 {len(candidates)}개 (탈락 {len(deep_dropped)})")

    # 업종 집중 상한 — 같은 업종만 우르르 통과해 상관 하락에 통째로 노출되는 걸 제한.
    # 강등된 종목은 탈락 사유에 남기고 아래에서 '관찰'로 계속 추적한다.
    sector_demoted: list[dict] = []
    try:
        from analyzer import apply_sector_cap
        candidates, sector_demoted = apply_sector_cap(candidates)
        for d in sector_demoted:
            r = d["result"]
            dropped_details.append({
                "code": r.get("code"), "name": r.get("name", r.get("code")),
                "total": r.get("total"), "close": r.get("last_close"),
                "reason": d["reason"],
            })
        if sector_demoted:
            _log(f"업종 상한: {len(sector_demoted)}개 관찰로 전환 (통과 {len(candidates)}개 유지)")
    except Exception as e:
        _log(f"업종 상한 적용 실패(통과 목록 그대로 유지): {type(e).__name__}: {e}")
        sector_demoted = []

    # 저장
    try:
        if hasattr(screening_history, "record_today_details"):
            screening_history.record_today_details(
                candidates, dropped_details,
                min_score=min_score, universe=universe,
            )
        else:
            screening_history.record_today([r["code"] for r in candidates])
        _log("screening_history 저장 ✅")
    except Exception as e:
        _log(f"screening_history 저장 실패: {type(e).__name__}: {e}")

    try:
        added, skipped = scouted.add_many_from_analysis(candidates, universe=universe)
        _log(f"scouted: +{added}개 새로 등록, {skipped}개 이미 추적 중")
    except Exception as e:
        _log(f"scouted 저장 실패: {type(e).__name__}: {e}")
        added, skipped = 0, 0

    # 업종 상한으로 강등된 종목은 관찰(observed)로 추적 지속 — 검증 데이터 유지
    if sector_demoted:
        try:
            _dem_results = [d["result"] for d in sector_demoted]
            scouted.add_observed_from_analysis(_dem_results, universe=universe)
        except Exception as e:
            _log(f"업종 상한 강등분 관찰 기록 실패: {type(e).__name__}: {e}")

    passed_codes = {r.get("code") for r in candidates}

    # 과열장 적응 통과(adaptive) — 시장 대비 초과가 적정한 건전 주도주를 소수 통과시킨다.
    # 절대 급등이 아니라 '시장 대비'로 위험을 재 통과 0개를 막되 추격 매수는 배제.
    adaptive_added = 0
    try:
        adaptive = select_adaptive_picks(screened, regime, passed_codes=passed_codes)
        adaptive_added, _ad_skip = scouted.add_adaptive_from_analysis(adaptive, universe=universe)
        passed_codes |= {r.get("code") for r in adaptive}
        _log(f"과열장 적응 통과: {len(adaptive)}개 선정, +{adaptive_added} 기록")
    except Exception as e:
        _log(f"적응 통과 기록 실패: {type(e).__name__}: {e}")

    # 관찰(observed) 기록 — 통과 0개여도 점수 검증 데이터가 끊기지 않게.
    # 과열장에선 surge 로 제외된 주도주도 모멘텀 후보로 함께 추적한다(매수 추천 아님).
    obs_added = 0
    try:
        obs = select_observation_targets(
            screened, fresh, regime, passed_codes=passed_codes,
        )
        obs_added, obs_skipped = scouted.add_observed_from_analysis(obs, universe=universe)
        _log(f"관찰: 대상 {len(obs)}개 중 +{obs_added} 기록 ({obs_skipped} 이미 추적 중)")
    except Exception as e:
        _log(f"관찰 기록 실패: {type(e).__name__}: {e}")

    # 보유 종목 청산 점검 — 매수 발굴과 대칭으로 '나갈 때'도 매일 점검
    try:
        import holdings_monitor
        held = holdings_monitor.monitor_holdings()
        holdings_monitor.save_alerts_snapshot(held)
        n_act = holdings_monitor.count_actionable(held)
        _log(f"보유 점검: {len(held)}종목 중 조치 알림 {n_act}건")
        for hh in held:
            for a in hh.get("alerts", []):
                if a.get("level") in ("high", "medium"):
                    _log(f"  ⚠ {hh.get('name')}: {a.get('msg')}")
    except Exception as e:
        _log(f"보유 점검 실패: {type(e).__name__}: {e}")

    # Gist 동기화 결과
    try:
        for line in cloud_store.get_sync_log()[-10:]:
            _log(f"  gist: {line}")
    except Exception:
        pass

    elapsed = time.time() - started
    _log(f"DONE in {elapsed:.1f}s — 통과 {len(candidates)}, 신규 발굴 {added}")
    return {
        "status": "ok",
        "universe": universe,
        "min_score": min_score,
        "universe_size": len(codes),
        "errors": errors,
        "excluded_stale": excluded_stale,
        "excluded_surge": excluded_surge,
        "candidates": len(candidates),
        "scouted_added": added,
        "scouted_skipped": skipped,
        "adaptive_added": adaptive_added,
        "observed_added": obs_added,
        "elapsed_sec": round(elapsed, 1),
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="헤드리스 자동 스크리닝")
    p.add_argument("--universe", default="safe", choices=["safe", "kospi_30", "kospi_50"])
    # 기준점수는 코드 고정(analyzer.SCREEN_MIN_SCORE) — CLI 로 고르지 않음.
    p.add_argument(
        "--no-deep", action="store_true",
        help="2단계 LLM 깊이 분석 생략 (빠름, 정확도 약간 떨어짐)",
    )
    p.add_argument("--workers", type=int, default=3, help="동시 분석 워커 수")
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args(sys.argv[1:])

    if not os.environ.get("DART_API_KEY"):
        _log("WARNING: DART_API_KEY 환경변수 없음 — 재무·공시 점수 누락됨")
    if not (os.environ.get("GITHUB_PAT") and os.environ.get("GIST_ID")):
        _log("WARNING: GIST 미설정 — 결과가 로컬 디스크에만 저장되어 컨테이너 종료 시 휘발됨")

    result = run(
        universe=args.universe,
        min_score=SCREEN_MIN_SCORE,
        deep=not args.no_deep,
        workers=args.workers,
    )
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
