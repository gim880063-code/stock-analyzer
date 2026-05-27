"""
점수 검증 — 점수 기반 가상 매매가 실제로 수익을 냈는지 사후 시뮬레이션.

두 가지 데이터 원천:
  1. scouted.json       — 스크리닝에서 발굴된 종목 + 발굴 시점 점수/종가
                          → 가상 매매 시뮬레이션 (실제 매수 없이 점수 검증)
  2. score_history.json — 일별 점수+종가 스냅샷
                          → 항목별 점수가 N일 뒤 수익률을 예측했는지 통계

답하려는 핵심 질문:
  Q1. 점수 높았던 종목이 실제로 더 올랐는가? (scouted 시뮬레이션)
  Q2. 어떤 항목이 진짜 예측력이 있었는가? (history — 데이터 축적 후)
  Q3. 종합/단기/중기 점수 중 어느 게 가장 잘 맞는가? (score_type 선택)

현재가 조회는 score_history의 가장 최근 close를 사용 (별도 가격 조회 없음 →
검증 페이지가 빠름). 즉, 시뮬레이션 대상이 되려면 그 종목이 최근 분석된 적이 있어야 함.
스크리닝을 주기적으로 돌리면 발굴 종목들의 가격이 자동으로 갱신됨.
"""
from collections import defaultdict
from statistics import mean

from analyzer import SHORT_TERM_ITEMS, MID_TERM_ITEMS
import history
import scouted


SCORE_TYPES = ("total", "short_term", "mid_term")


def _extract_score(info: dict, score_type: str) -> int | None:
    """발굴 엔트리에서 지정한 점수 종류 추출. 데이터 없으면 None."""
    if score_type == "total":
        return info.get("added_score")
    added_scores = info.get("added_scores")
    if not added_scores:
        return None
    if score_type == "short_term":
        items = SHORT_TERM_ITEMS
    elif score_type == "mid_term":
        items = MID_TERM_ITEMS
    else:
        return None
    return sum(int(v) for k, v in added_scores.items() if k in items)


# 점수 구간 — 종합/단기/중기 점수는 max 값이 달라서 bucket도 다르게.
# 종합 max ≈ 9~12, 단기 max ≈ 4, 중기 max ≈ 7.
_BUCKETS_TOTAL = [
    ("≥8 (강력)", lambda s: s >= 8),
    ("5~7 (긍정)", lambda s: 5 <= s <= 7),
    ("2~4 (중립+)", lambda s: 2 <= s <= 4),
    ("≤1 (관망)", lambda s: s <= 1),
]
_BUCKETS_SHORT = [
    ("≥3 (강력)", lambda s: s >= 3),
    ("1~2 (긍정)", lambda s: 1 <= s <= 2),
    ("0 (중립)", lambda s: s == 0),
    ("≤-1 (관망)", lambda s: s <= -1),
]
_BUCKETS_MID = [
    ("≥5 (강력)", lambda s: s >= 5),
    ("2~4 (긍정)", lambda s: 2 <= s <= 4),
    ("0~1 (중립)", lambda s: 0 <= s <= 1),
    ("≤-1 (관망)", lambda s: s <= -1),
]


def _buckets_for(score_type: str):
    if score_type == "short_term":
        return _BUCKETS_SHORT
    if score_type == "mid_term":
        return _BUCKETS_MID
    return _BUCKETS_TOTAL


def _latest_entry_from_history(code: str) -> dict | None:
    """history에 저장된 가장 최근 entry. 없으면 None."""
    entries = history.get_history(code, days=365)
    return entries[-1] if entries else None


def _extract_score_from_entry(entry: dict, score_type: str) -> int | None:
    """history entry에서 지정한 점수 종류 추출. 발굴 entry의 _extract_score와
    같은 로직이지만 필드명이 다름 (total/scores vs added_score/added_scores)."""
    if score_type == "total":
        return entry.get("total")
    scores = entry.get("scores")
    if not scores:
        return None
    if score_type == "short_term":
        items = SHORT_TERM_ITEMS
    elif score_type == "mid_term":
        items = MID_TERM_ITEMS
    else:
        return None
    return sum(int(v) for k, v in scores.items() if k in items)


def _bucket_stats(rows: list[dict], value_key: str, score_key: str, score_type: str = "total") -> dict:
    """점수 구간별 평균 수익률·승률 계산. score_type에 맞는 bucket 사용."""
    bucket_def = _buckets_for(score_type)
    buckets: dict[str, list[float]] = {label: [] for label, _ in bucket_def}
    for r in rows:
        s = r.get(score_key)
        if s is None:
            continue
        for label, predicate in bucket_def:
            if predicate(s):
                buckets[label].append(r[value_key])
                break
    return {
        label: {
            "count": len(returns),
            "avg_return": round(mean(returns), 2) if returns else None,
            "win_rate": (
                round(sum(1 for x in returns if x > 0) / len(returns) * 100, 1)
                if returns else None
            ),
            "best": round(max(returns), 2) if returns else None,
            "worst": round(min(returns), 2) if returns else None,
        }
        for label, returns in buckets.items()
    }


def verify_scouted(score_type: str = "total") -> dict:
    """
    발굴 종목들의 발굴 후 수익률 + 점수 구간별 평균.

    score_type: "total" | "short_term" | "mid_term"
      - total: 발굴 시점 종합 점수 (added_score) — 모든 발굴 엔트리 가능
      - short_term: SHORT_TERM_ITEMS 항목만 합산 — added_scores 있는 엔트리만
      - mid_term: MID_TERM_ITEMS 항목만 합산 — added_scores 있는 엔트리만

    반환:
      rows: 종목별 상세 (수익률 내림차순)
      bucket_stats: 점수 구간별 평균 수익률·승률
      total_count: 검증 가능한 종목 수
      missing_data_count: 검증 불가능한 종목 수 (점수 또는 가격 데이터 부족)
      score_type: 사용된 점수 종류
    """
    if score_type not in SCORE_TYPES:
        score_type = "total"

    items = scouted.load_scouted()
    rows: list[dict] = []
    for code, info in items.items():
        added_close = info.get("added_close")
        if not added_close:
            continue
        score = _extract_score(info, score_type)
        if score is None:
            continue
        last_entry = _latest_entry_from_history(code)
        if not last_entry:
            continue
        last_close = last_entry.get("close")
        if not last_close:
            continue
        ret_pct = (last_close / added_close - 1) * 100
        rows.append({
            "code": code,
            "added_at": info.get("added_at"),
            "added_score": score,
            "added_close": added_close,
            "current_close": last_close,
            "current_date": last_entry.get("date"),
            "current_score": _extract_score_from_entry(last_entry, score_type),
            "return_pct": round(ret_pct, 2),
            "universe": info.get("universe", ""),
        })

    rows.sort(key=lambda r: r["return_pct"], reverse=True)

    return {
        "rows": rows,
        "bucket_stats": _bucket_stats(rows, "return_pct", "added_score", score_type),
        "total_count": len(rows),
        "missing_data_count": len(items) - len(rows),
        "overall_avg": round(mean([r["return_pct"] for r in rows]), 2) if rows else None,
        "overall_win_rate": (
            round(sum(1 for r in rows if r["return_pct"] > 0) / len(rows) * 100, 1)
            if rows else None
        ),
        "score_type": score_type,
    }


def verify_item_scores(forward_days: int = 5) -> dict:
    """
    history 기반 항목별 점수 → forward_days일 뒤 수익률 분석.

    로직:
      각 종목의 일별 엔트리를 슬라이딩 윈도우로 보면서,
      "이 날 항목 X의 점수가 +1이었던 케이스"의 forward_days일 뒤 수익률을 모은다.
      점수가 -1이었던 케이스와 평균 수익률 차이가 크면 그 항목이 예측력 있음.

    forward_days: 5(단기) / 20(중기) / 60(장기) 권장.
                  데이터가 forward_days보다 짧으면 그 종목은 분석에서 제외됨.

    한계: 같은 종목의 시점별 점수 변화를 보는 것이라, 종목 수가 많아도
          분석 가능한 (시점, 종목) 페어 수가 보존된 일수에 비례. 의미 있는 통계가
          나오려면 일별 분석 누적이 수십~수백 케이스 필요.
    """
    all_hist = history.load_all()

    # {item_name: {score_value: [returns]}}
    per_item: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    sample_count = 0

    for code, entries in all_hist.items():
        # 날짜순 정렬 (저장 시 정렬되지만 안전을 위해)
        entries = sorted(entries, key=lambda e: e.get("date", ""))
        for i, e in enumerate(entries):
            scores = e.get("scores")
            if not scores:
                continue
            if i + forward_days >= len(entries):
                continue
            close_now = e.get("close")
            close_future = entries[i + forward_days].get("close")
            if not close_now or not close_future:
                continue
            ret = (close_future / close_now - 1) * 100
            sample_count += 1
            for item_name, item_score in scores.items():
                per_item[item_name][int(item_score)].append(ret)

    # 항목별 통계 — 각 점수값(-2, -1, 0, +1, +2)별 평균 수익률·승률
    item_stats: dict[str, dict] = {}
    for item, score_map in per_item.items():
        by_score = {}
        for score_val, returns in sorted(score_map.items()):
            by_score[score_val] = {
                "count": len(returns),
                "avg_return": round(mean(returns), 2) if returns else None,
                "win_rate": (
                    round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1)
                    if returns else None
                ),
            }
        # 예측력 지표: (양수 점수 평균) - (음수 점수 평균)
        # 양수 = 항목이 양호하다고 판단한 케이스, 음수 = 안 좋다고 판단한 케이스
        pos_returns = [r for s, rs in score_map.items() if s > 0 for r in rs]
        neg_returns = [r for s, rs in score_map.items() if s < 0 for r in rs]
        spread = None
        if pos_returns and neg_returns:
            spread = round(mean(pos_returns) - mean(neg_returns), 2)

        item_stats[item] = {
            "by_score": by_score,
            "predictive_spread": spread,  # +면 항목이 예측력 있음, -면 반대로 작동, None이면 데이터 부족
            "total_samples": sum(len(rs) for rs in score_map.values()),
        }

    # 예측력 spread 큰 순으로 정렬한 키 리스트도 제공
    ranked_items = sorted(
        item_stats.keys(),
        key=lambda k: (item_stats[k]["predictive_spread"] or -999),
        reverse=True,
    )

    return {
        "item_stats": item_stats,
        "ranked_items": ranked_items,
        "forward_days": forward_days,
        "total_samples": sample_count,
    }
