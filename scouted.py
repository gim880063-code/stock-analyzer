"""
📌 발굴 추적 — 스크리닝에서 발견한 종목들을 모아 매일 추이를 보는 기능 +
점수 기반 가상 매매 검증의 데이터 원천.

저장 위치: Gist (설정 시) + 로컬 data/scouted.json
형식:
  {
    "005930": {
      "added_at": "2026-05-13",       # 추적 시작일
      "added_score": 4,                # 추적 시작 시점 종합점수
      "added_close": 78500.0,          # 추적 시작 시점 종가 (가상 매수가) — 2026-05 이후 추가
      "added_scores": {                # 항목별 점수 스냅샷 — 2026-05 이후 추가
        "추세": 1, "모멘텀": 1, "거래량": 0, ...
      },
      "universe": "safe"               # 어느 유니버스에서 발굴됐는지
    },
    ...
  }

워치리스트·즐겨찾기와의 차이:
  - 워치리스트: 분석 대상 (수시로 바뀜)
  - 즐겨찾기: 관심 있는 종목 (수동, 점수 추세 모름)
  - 발굴 추적: 스크리닝에서 후보로 발굴된 것만, 점수 변화 추적, 가상 매매 PnL 계산
"""
from datetime import datetime

import cloud_store


FILENAME = "scouted.json"


def load_scouted() -> dict[str, dict]:
    data = cloud_store.load(FILENAME, {})
    return data if isinstance(data, dict) else {}


def save_scouted(items: dict[str, dict]) -> None:
    cloud_store.save(FILENAME, items)


# 추적 종류 우선순위 — 높을수록 우선. 이미 추적 중이면 더 높은 종류로만 승격한다.
_KIND_RANK = {"observed": 1, "adaptive": 2, "picked": 3}


def _entry_kind(entry: dict) -> str:
    """추적 종류. 레거시(kind 미설정) 엔트리는 'picked'로 간주(하위호환)."""
    k = (entry or {}).get("kind")
    return k if k in _KIND_RANK else "picked"


def _compact_scores(scores: list[dict] | None) -> dict[str, int] | None:
    if not scores:
        return None
    return {
        s["name"]: int(s["score"])
        for s in scores
        if isinstance(s, dict) and "name" in s and "score" in s
    }


def add(
    code: str,
    score: int,
    universe: str = "safe",
    close: float | None = None,
    scores: list[dict] | None = None,
    kind: str = "picked",
) -> bool:
    """추가됐으면 True, 이미 있으면 False.
    close: 발굴 시점 종가 (가상 매수가로 활용 → verifier가 수익률 계산)
    scores: analyzer.AnalysisResult["scores"] 그대로 (항목별 검증용)
    kind: "picked"(통과 발굴) / "observed"(상위 관찰)
    """
    s = load_scouted()
    if code in s:
        return False
    entry: dict = {
        "added_at": datetime.now().strftime("%Y-%m-%d"),
        "added_score": int(score),
        "universe": universe,
        "kind": kind,
    }
    if close is not None:
        entry["added_close"] = float(close)
    compact = _compact_scores(scores)
    if compact:
        entry["added_scores"] = compact
    s[code] = entry
    save_scouted(s)
    return True


def add_from_analysis(result: dict, universe: str = "safe") -> bool:
    """analyzer.analyze() 결과 dict를 그대로 받아 발굴 등록.
    종합점수·종가·항목별 점수까지 한 번에 스냅샷.
    """
    code = result.get("code")
    if not code:
        return False
    return add(
        code=code,
        score=int(result.get("total", 0)),
        universe=universe,
        close=result.get("last_close"),
        scores=result.get("scores"),
    )


def add_many_from_analysis(
    results: list[dict], universe: str = "safe", kind: str = "picked",
) -> tuple[int, int]:
    """스크리닝 결과 리스트를 한 번에 등록 (load 1회 + save 1회).
    kind="picked"(통과 발굴) / "observed"(상위 관찰). 이미 추적 중인 코드는 건너뛰되,
    picked 추가가 기존 observed 를 만나면 추적 시작점(added_at/score)은 보존한 채
    종류만 'picked' 로 승격한다.
    Returns: (새로 추가된 개수, 이미 있어서 건너뛴 개수)
    """
    s = load_scouted()
    today = datetime.now().strftime("%Y-%m-%d")
    added = 0
    skipped = 0
    changed = False
    for result in results:
        code = result.get("code")
        if not code:
            continue
        if code in s:
            if _KIND_RANK.get(kind, 0) > _KIND_RANK.get(_entry_kind(s[code]), 0):
                s[code]["kind"] = kind  # 더 높은 종류로 승격 — 추적 시작점은 유지
                changed = True
            skipped += 1
            continue
        entry: dict = {
            "added_at": today,
            "added_score": int(result.get("total", 0)),
            "universe": universe,
            "kind": kind,
        }
        close = result.get("last_close")
        if close is not None:
            entry["added_close"] = float(close)
        compact = _compact_scores(result.get("scores"))
        if compact:
            entry["added_scores"] = compact
        s[code] = entry
        added += 1
        changed = True
    if changed:
        save_scouted(s)
    return added, skipped


def add_observed_from_analysis(
    results: list[dict], universe: str = "safe",
) -> tuple[int, int]:
    """관찰(observed) 대상 기록 — 통과하지 못했지만 점수 검증을 위해 추적할 종목.
    매수 추천이 아니라 '점수가 실제 수익률과 맞았는지' 데이터가 끊기지 않게 쌓는 용도.
    이미 더 높은 종류로 추적 중이면 건너뛴다(picked > adaptive > observed).
    """
    return add_many_from_analysis(results, universe=universe, kind="observed")


def add_adaptive_from_analysis(
    results: list[dict], universe: str = "safe",
) -> tuple[int, int]:
    """과열장 적응 통과(adaptive) 기록 — 시장 대비 초과가 적정한 건전 주도주.
    실제 통과(picked류)지만, 기존 통과 수익률 통계와 분리 검증하려고 종류를 구분한다.
    """
    return add_many_from_analysis(results, universe=universe, kind="adaptive")


def add_many(items: list[tuple[str, int]], universe: str = "safe") -> int:
    """새로 추가된 개수 반환 (이미 있는 건 건너뜀).
    items: [(code, score), ...] — 종가·항목별 점수 없이 간단 등록.
    풍부한 스냅샷이 필요하면 add_from_analysis()를 종목별로 호출하세요.
    """
    s = load_scouted()
    today = datetime.now().strftime("%Y-%m-%d")
    added = 0
    for code, score in items:
        if code not in s:
            s[code] = {
                "added_at": today,
                "added_score": int(score),
                "universe": universe,
            }
            added += 1
    if added > 0:
        save_scouted(s)
    return added


def remove(code: str) -> None:
    s = load_scouted()
    if code in s:
        del s[code]
        save_scouted(s)


def clear_all() -> None:
    save_scouted({})
