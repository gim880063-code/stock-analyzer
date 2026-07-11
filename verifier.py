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

핵심 보정 (2026-05 개편):
  - 고정 시계(horizon) 도입: 5d / 20d / 60d / 보유기간 전체 — 발굴일 다른 종목들의
    수익률을 같은 잣대로 비교할 수 있게 함.
  - 시장 대비 초과수익(excess return) 계산: 같은 기간 KOSPI/KOSDAQ 수익률을 빼서,
    하락장에 -3% 종목이 시장(-8%) 대비 +5%p 강세였다는 사실을 통계가 잡아냄.
  - 최소 보유일 필터: 발굴 후 며칠 안 된 종목은 통계에서 제외 (신호가 작동할
    시간이 없어서 노이즈만 만듦).
  - 분위수 기반 버킷: 고정 임계값 (≥11/6~10/...) 은 발굴 종목이 min_score 위로만
    걸려 들어와 분포가 편향됨 → 실제 분포에서 상/중/하 1/3씩.
  - 가격 데이터: FinanceDataReader 로 발굴일~현재 종가를 새로 조회 (stale history
    의존 제거). 종목당 1회 fetch 후 lru_cache.
"""
from collections import defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from statistics import mean, median, stdev

import pandas as pd
import FinanceDataReader as fdr

from analyzer import SHORT_TERM_ITEMS, MID_TERM_ITEMS, FOCUS_ITEMS, weighted_score, _stock_market
import history
import scouted


SCORE_TYPES = ("total", "short_term", "mid_term", "focus")

# 시뮬레이션에서 사용하는 고정 보유 기간 옵션.
# "all" 은 발굴일부터 현재(마지막 거래일)까지의 총 수익률.
HORIZONS = ("5d", "20d", "60d", "all")
HORIZON_DAYS = {"5d": 5, "20d": 20, "60d": 60}

# 신호가 작동할 시간 부족한 종목은 통계 노이즈 — 기본 최소 5 영업일 보유 필요.
DEFAULT_MIN_HOLD_DAYS = 5

# 항목별 rank-IC(순위상관) 를 보고할 최소 관측치. 이보다 적으면 IC=None 처리한다
# (점수가 -1/0/+1 같은 거친 정수라, 표본이 적으면 상관계수가 크게 요동침).
IC_MIN_OBS = 30


# ─────────── 가격 fetch 헬퍼 ───────────
# 종목별/지수별로 1회만 fetch (lru_cache). 일자 키는 오늘 날짜로 잡아
# 같은 날 안에서는 같은 결과 캐싱, 다음 날엔 자동 재조회.
@lru_cache(maxsize=256)
def _fetch_close_series(code: str, today_key: str) -> pd.Series | None:
    """종목의 일별 종가 시리즈 (최근 1년). 실패 시 None."""
    try:
        end = datetime.now()
        start = end - timedelta(days=400)
        df = fdr.DataReader(code, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if df is None or df.empty:
            return None
        return df["Close"].astype(float)
    except Exception:
        return None


@lru_cache(maxsize=4)
def _fetch_market_series(market: str, today_key: str) -> pd.Series | None:
    """시장 지수 종가 시리즈."""
    idx_code = "KS11" if market == "KOSPI" else "KQ11"
    try:
        end = datetime.now()
        start = end - timedelta(days=400)
        df = fdr.DataReader(idx_code, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if df is None or df.empty:
            return None
        return df["Close"].astype(float)
    except Exception:
        return None


def _today_key() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _safe_market_for_code(code: str) -> str:
    try:
        return _stock_market(code)
    except Exception:
        return "KOSPI"


def _find_idx_on_or_after(series: pd.Series, date_str: str) -> int | None:
    """series.index (DatetimeIndex) 에서 date_str 이상 첫 위치. 없으면 None."""
    try:
        target = pd.Timestamp(date_str)
    except Exception:
        return None
    # searchsorted 로 O(log n).
    pos = series.index.searchsorted(target, side="left")
    if pos >= len(series):
        return None
    return int(pos)


def _horizon_return(
    series: pd.Series,
    start_date: str,
    horizon: str,
) -> tuple[float | None, float | None, int | None, int | None]:
    """
    Args:
        series: 일별 종가 시리즈 (DatetimeIndex)
        start_date: 발굴일 (YYYY-MM-DD)
        horizon: "5d" | "20d" | "60d" | "all"

    Returns:
        (start_close, end_close, days_held_trading, end_idx)
        - days_held_trading: 영업일 수 (start_idx ~ end_idx 거리)
        - 데이터 부족 시 (None, None, None, None)
    """
    if series is None or series.empty:
        return None, None, None, None

    start_idx = _find_idx_on_or_after(series, start_date)
    if start_idx is None:
        return None, None, None, None

    if horizon == "all":
        end_idx = len(series) - 1
    else:
        offset = HORIZON_DAYS.get(horizon)
        if offset is None:
            return None, None, None, None
        end_idx = start_idx + offset
        # 아직 horizon 도달 안 한 종목은 통계 제외 (보유기간 부족).
        if end_idx >= len(series):
            return None, None, None, None

    if end_idx <= start_idx:
        return None, None, None, None

    start_close = float(series.iloc[start_idx])
    end_close = float(series.iloc[end_idx])
    days_held = end_idx - start_idx
    return start_close, end_close, days_held, end_idx


# ─────────── 점수 추출 (legacy 호환) ───────────
def _extract_score(info: dict, score_type: str) -> int | None:
    """발굴 엔트리에서 지정한 점수 종류 추출. 데이터 없으면 None.

    가중치 일관성 보장: added_scores (항목별 스냅샷) 가 있으면 *항상* 현재
    SCORE_WEIGHTS 로 재계산. 이렇게 안 하면 가중치 도입 이전 발굴분의
    added_score (단순합산) 과 이후 (가중합산) 가 섞여 bucket 분석이 왜곡됨.

    added_scores 없는 옛 엔트리만 저장된 added_score 사용.
    """
    added_scores = info.get("added_scores")
    if added_scores:
        added_scores = history.normalize_scores(added_scores)  # 옛 항목명 통일 (재무 → 재무 건전성)

    if score_type == "total":
        if added_scores:
            score_items = [
                {"name": k, "score": int(v), "max": 1}
                for k, v in added_scores.items()
            ]
            return weighted_score(score_items)[0]
        return info.get("added_score")

    if not added_scores:
        return None
    if score_type == "short_term":
        items = SHORT_TERM_ITEMS
    elif score_type == "mid_term":
        items = MID_TERM_ITEMS
    elif score_type == "focus":
        items = FOCUS_ITEMS
    else:
        return None
    score_items = [
        {"name": k, "score": int(v), "max": 1}
        for k, v in added_scores.items()
        if k in items
    ]
    return weighted_score(score_items, items)[0]


def _extract_score_from_entry(entry: dict, score_type: str) -> int | None:
    """history entry 에서 같은 로직으로 추출 (필드명만 다름)."""
    scores = entry.get("scores")

    if score_type == "total":
        if scores:
            score_items = [
                {"name": k, "score": int(v), "max": 1}
                for k, v in scores.items()
            ]
            return weighted_score(score_items)[0]
        return entry.get("total")

    if not scores:
        return None
    if score_type == "short_term":
        items = SHORT_TERM_ITEMS
    elif score_type == "mid_term":
        items = MID_TERM_ITEMS
    elif score_type == "focus":
        items = FOCUS_ITEMS
    else:
        return None
    score_items = [
        {"name": k, "score": int(v), "max": 1}
        for k, v in scores.items()
        if k in items
    ]
    return weighted_score(score_items, items)[0]


# ─────────── 분위수 기반 버킷 ───────────
# 스크리닝이 min_score 위만 scouted 에 등록하므로 점수 분포가 편향됨.
# 고정 임계값(≥11 강력 등) 은 빈 버킷 만들기 일쑤 → 실제 분포에서 상/중/하 1/3.
def _quantile_buckets(scores: list[int]) -> list[tuple[str, callable]]:
    """점수 리스트에서 분위수 임계를 잡아 (label, predicate) 튜플 반환.

    종목 수가 3개 미만이면 단일 '전체' 버킷만 반환 (분위 의미 없음).
    상/중/하 임계가 같은 값이면 (모두 동점) 의미 있는 분할 불가 → 단일 버킷.
    """
    valid = [s for s in scores if s is not None]
    if len(valid) < 3:
        return [("전체", lambda s: s is not None)]

    sorted_scores = sorted(valid)
    n = len(sorted_scores)
    q1 = sorted_scores[n // 3]          # 하위 1/3 경계
    q2 = sorted_scores[(2 * n) // 3]    # 상위 1/3 경계

    if q1 == q2:
        # 분포가 한 점에 너무 몰림 — 분위 의미 없음.
        return [("전체", lambda s: s is not None)]

    return [
        (f"상위 (≥{q2})", (lambda s, t=q2: s is not None and s >= t)),
        (f"중위 ({q1}~{q2 - 1})", (lambda s, lo=q1, hi=q2: s is not None and lo <= s < hi)),
        (f"하위 (<{q1})", (lambda s, t=q1: s is not None and s < t)),
    ]


def _bucket_stats(
    rows: list[dict],
    score_key: str,
    abs_return_key: str = "return_pct",
    excess_return_key: str = "excess_return_pct",
) -> dict:
    """점수 구간별 통계 — 절대 수익률·초과수익률 각각 평균/중앙값/승률 산출."""
    bucket_def = _quantile_buckets([r.get(score_key) for r in rows])

    groups: dict[str, list[dict]] = {label: [] for label, _ in bucket_def}
    for r in rows:
        s = r.get(score_key)
        for label, predicate in bucket_def:
            if predicate(s):
                groups[label].append(r)
                break

    out: dict[str, dict] = {}
    for label, entries in groups.items():
        abs_vals = [e[abs_return_key] for e in entries if e.get(abs_return_key) is not None]
        ex_vals = [e[excess_return_key] for e in entries if e.get(excess_return_key) is not None]
        out[label] = {
            "count": len(entries),
            "avg_return": round(mean(abs_vals), 2) if abs_vals else None,
            "median_return": round(median(abs_vals), 2) if abs_vals else None,
            "win_rate": (
                round(sum(1 for x in abs_vals if x > 0) / len(abs_vals) * 100, 1)
                if abs_vals else None
            ),
            "avg_excess": round(mean(ex_vals), 2) if ex_vals else None,
            "median_excess": round(median(ex_vals), 2) if ex_vals else None,
            "excess_win_rate": (
                round(sum(1 for x in ex_vals if x > 0) / len(ex_vals) * 100, 1)
                if ex_vals else None
            ),
            "best": round(max(abs_vals), 2) if abs_vals else None,
            "worst": round(min(abs_vals), 2) if abs_vals else None,
        }
    return out


# 트레일링 스톱 — 발굴 후 종가 기준으로 고점 대비 이만큼 빠지면 청산(손절).
# "사서 끝까지 보유" 대신 하락을 잘라낸 가상 수익을 함께 보기 위함.
# portfolio 설정의 trail_pct(기본 10%)를 따른다.
DEFAULT_TRAIL_PCT = 10.0


def _trailing_stop_return(close, start_idx, end_idx, trail_pct):
    """start_idx 종가에 진입한 뒤 종가 기준 트레일링 스톱을 적용한 수익률.

    진입 후 최고 종가 대비 trail_pct% 이상 하락한 첫날 종가에 청산한다. 끝까지
    안 걸리면 end_idx 종가까지 보유. 종가 기준이라 장중 변동은 무시(보수적).

    Returns: (수익률%, 청산까지 영업일수, 스톱 발동 여부). 계산 불가 시 (None, None, False).
    """
    if close is None or start_idx is None or end_idx is None or end_idx <= start_idx:
        return None, None, False
    start = float(close.iloc[start_idx])
    if start <= 0:
        return None, None, False
    thresh = 1.0 - max(0.0, float(trail_pct)) / 100.0
    peak = start
    for i in range(start_idx + 1, end_idx + 1):
        c = float(close.iloc[i])
        if c > peak:
            peak = c
        if c <= peak * thresh:
            return (c / start - 1) * 100, i - start_idx, True
    end = float(close.iloc[end_idx])
    return (end / start - 1) * 100, end_idx - start_idx, False


# ─────────── 메인: 발굴 가상 매매 검증 ───────────
def verify_scouted(
    score_type: str = "total",
    horizon: str = "all",
    min_hold_days: int = DEFAULT_MIN_HOLD_DAYS,
    kind: str = "picked",
) -> dict:
    """
    발굴 종목들의 발굴 후 수익률 통계.

    Args:
        score_type: "total" | "short_term" | "mid_term"
        horizon: "5d" | "20d" | "60d" | "all"
            - 5d/20d/60d: 발굴일 + N 영업일 시점 종가까지의 수익률
              (해당 기간 도달 못 한 종목은 자동 제외)
            - all: 발굴일 ~ 마지막 거래일 (보유기간 종목마다 다름)
        min_hold_days: 통계 계산에 포함되는 최소 보유 영업일.
            horizon="all" 일 때만 의미. horizon="5d"는 자동으로 5일 보유한 종목만.

    반환:
        rows: 종목별 상세 (excess_return_pct 내림차순)
        bucket_stats: 점수 분위별 평균/중앙값/승률 + 초과수익 통계
        total_count: 통계 포함 종목 수
        excluded_short_hold: min_hold_days 미달로 제외된 종목 수
        missing_data_count: 점수·가격 데이터 없어 검증 불가능한 종목 수
        overall_*: 전체 평균/중앙값/승률 (절대 + 초과 + 트레일링 스톱 적용)
        score_type, horizon, min_hold_days: echo back
    """
    if score_type not in SCORE_TYPES:
        score_type = "total"
    if horizon not in HORIZONS:
        horizon = "all"

    try:
        import portfolio
        trail_pct = float(portfolio.load_settings().get("trail_pct", DEFAULT_TRAIL_PCT))
    except Exception:
        trail_pct = DEFAULT_TRAIL_PCT

    items = scouted.load_scouted()
    today = _today_key()

    rows: list[dict] = []
    excluded_short_hold = 0
    missing_data_count = 0

    for code, info in items.items():
        # kind 필터 — 기본 'picked'(통과 발굴)만. observed(관찰)가 섞여 통과 종목
        # 성과 통계가 희석되지 않게 분리한다. "all" 이면 둘 다 포함.
        if kind != "all" and scouted._entry_kind(info) != kind:
            continue
        added_at = info.get("added_at")
        if not added_at:
            missing_data_count += 1
            continue

        score = _extract_score(info, score_type)
        if score is None:
            missing_data_count += 1
            continue

        series = _fetch_close_series(code, today)
        if series is None or series.empty:
            missing_data_count += 1
            continue

        start_close, end_close, days_held, end_idx = _horizon_return(series, added_at, horizon)
        if start_close is None:
            # 5d/20d/60d 도달 못 한 케이스 — "데이터 부족" 으로 분류 안 함 (시간만 부족).
            if horizon != "all":
                excluded_short_hold += 1
            else:
                missing_data_count += 1
            continue

        # horizon="all" + min_hold_days 미달 → 보유 부족으로 제외.
        if horizon == "all" and days_held is not None and days_held < min_hold_days:
            excluded_short_hold += 1
            continue

        ret_pct = (end_close / start_close - 1) * 100

        # 트레일링 스톱 적용 수익 — 하락을 잘라낸 가상 성과 (start_idx 복원: end - 보유일)
        trail_ret, trail_days, trail_stopped = _trailing_stop_return(
            series, end_idx - days_held, end_idx, trail_pct
        )

        # 시장 대비 초과수익
        market = _safe_market_for_code(code)
        market_series = _fetch_market_series(market, today)
        market_ret = None
        excess = None
        if market_series is not None and not market_series.empty:
            m_start_idx = _find_idx_on_or_after(market_series, added_at)
            if m_start_idx is not None:
                # 시장도 같은 영업일 기준으로 매핑.
                if horizon == "all":
                    m_end_idx = len(market_series) - 1
                else:
                    m_end_idx = m_start_idx + HORIZON_DAYS[horizon]
                if 0 <= m_end_idx < len(market_series) and m_end_idx > m_start_idx:
                    m_start = float(market_series.iloc[m_start_idx])
                    m_end = float(market_series.iloc[m_end_idx])
                    if m_start > 0:
                        market_ret = (m_end / m_start - 1) * 100
                        excess = ret_pct - market_ret

        # 현재 점수 (참고용) — history 에서 최근 entry
        last_entry = None
        try:
            hist_entries = history.get_history(code, days=365)
            last_entry = hist_entries[-1] if hist_entries else None
        except Exception:
            pass
        current_score = (
            _extract_score_from_entry(last_entry, score_type) if last_entry else None
        )

        rows.append({
            "code": code,
            "added_at": added_at,
            "added_score": score,
            "added_close": start_close,
            "current_close": end_close,
            "days_held": days_held,
            "return_pct": round(ret_pct, 2),
            "trail_return_pct": round(trail_ret, 2) if trail_ret is not None else None,
            "trail_days": trail_days,
            "trail_stopped": bool(trail_stopped),
            "market": market,
            "market_return_pct": round(market_ret, 2) if market_ret is not None else None,
            "excess_return_pct": round(excess, 2) if excess is not None else None,
            "current_score": current_score,
            "universe": info.get("universe", ""),
        })

    # 초과수익 내림차순 (None은 뒤로)
    rows.sort(
        key=lambda r: (
            r.get("excess_return_pct") is None,
            -(r.get("excess_return_pct") or 0),
        )
    )

    abs_vals = [r["return_pct"] for r in rows]
    ex_vals = [r["excess_return_pct"] for r in rows if r["excess_return_pct"] is not None]
    trail_vals = [r["trail_return_pct"] for r in rows if r["trail_return_pct"] is not None]

    return {
        "rows": rows,
        "bucket_stats": _bucket_stats(rows, score_key="added_score"),
        "total_count": len(rows),
        "excluded_short_hold": excluded_short_hold,
        "missing_data_count": missing_data_count,
        "overall_avg": round(mean(abs_vals), 2) if abs_vals else None,
        "overall_median": round(median(abs_vals), 2) if abs_vals else None,
        "overall_win_rate": (
            round(sum(1 for x in abs_vals if x > 0) / len(abs_vals) * 100, 1)
            if abs_vals else None
        ),
        "overall_avg_excess": round(mean(ex_vals), 2) if ex_vals else None,
        "overall_median_excess": round(median(ex_vals), 2) if ex_vals else None,
        "overall_excess_win_rate": (
            round(sum(1 for x in ex_vals if x > 0) / len(ex_vals) * 100, 1)
            if ex_vals else None
        ),
        "overall_avg_trail": round(mean(trail_vals), 2) if trail_vals else None,
        "overall_win_rate_trail": (
            round(sum(1 for x in trail_vals if x > 0) / len(trail_vals) * 100, 1)
            if trail_vals else None
        ),
        "trail_pct": trail_pct,
        "trail_stopped_count": sum(1 for r in rows if r.get("trail_stopped")),
        "score_type": score_type,
        "horizon": horizon,
        "min_hold_days": min_hold_days,
    }


# ─────────── 항목별 점수 예측력 ───────────
def _spearman_ic(xs: list[float], ys: list[float]) -> float | None:
    """순위상관(rank-IC). 순위로 바꾼 뒤 Pearson = Spearman 정의 → scipy 불필요(무료·로컬).

    표본(IC_MIN_OBS 미만)·분산(x가 모두 같음) 부족하면 None.
    """
    if len(xs) < IC_MIN_OBS or len(set(xs)) < 2:
        return None
    try:
        rx = pd.Series(xs, dtype="float64").rank()
        ry = pd.Series(ys, dtype="float64").rank()
        ic = rx.corr(ry)   # 기본 pearson — 순위에 적용하면 Spearman 과 동일(동점 보정 포함)
    except Exception:
        return None
    if ic is None or ic != ic:   # NaN 방어
        return None
    return round(float(ic), 3)


def verify_item_scores(forward_days: int = 5, use_excess: bool = True) -> dict:
    """
    history 기반 항목별 점수 → forward_days 영업일 뒤 수익률 분석.

    각 항목(추세·수급·거래량·공시 …) 점수가 실제로 수익을 예측했는지 두 가지로 본다:
      - rank_ic: 항목 점수와 forward 수익률의 순위상관(Spearman). +면 점수 높을수록
        더 오름(예측력 있음), −면 거꾸로, 0 근처면 무의미. use_excess=True 면
        시장(KOSPI/KOSDAQ) 대비 초과수익 기준 — 추세장에 모든 항목이 좋아 보이는
        착시를 제거한다. (예측력 판단의 주 지표)
      - predictive_spread: (양수 점수일 때 평균수익) − (음수 점수일 때 평균수익), 절대수익.

    개편 (2026-05): 점수 기록일을 anchor 로 잡고 FinanceDataReader 가격으로 정확히
    forward_days 영업일 뒤 종가를 가져와 forward return 계산.
    rank-IC·초과수익 추가 (2026-06): 예측력을 더 정직하게 측정.

    forward_days: 5(단기) / 20(중기) / 60(장기). 데이터 기간이 짧으면 20·60일은
    완성된 관측이 거의 없다(반환의 n_dates 로 표본 기간 확인).
    """
    all_hist = history.load_all(include_archive=True)  # 아카이브 포함 — 장기 검증 창
    today = _today_key()

    per_item: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    ic_x: dict[str, list[int]] = defaultdict(list)      # 항목 점수
    ic_y: dict[str, list[float]] = defaultdict(list)    # forward 수익(초과수익 옵션)
    sample_count = 0
    dates_seen: set[str] = set()
    stocks_seen: set[str] = set()

    market_cache: dict[str, "pd.Series | None"] = {}

    def _market(market: str):
        if market not in market_cache:
            market_cache[market] = _fetch_market_series(market, today) if use_excess else None
        return market_cache[market]

    for code, entries in all_hist.items():
        if not entries:
            continue
        series = _fetch_close_series(code, today)
        if series is None or series.empty:
            continue
        mser = _market(_safe_market_for_code(code))

        for e in sorted(entries, key=lambda e: e.get("date", "")):
            scores = e.get("scores")
            if not scores:
                continue
            anchor_date = e.get("date")
            if not anchor_date:
                continue
            start_idx = _find_idx_on_or_after(series, anchor_date)
            if start_idx is None:
                continue
            end_idx = start_idx + forward_days
            if end_idx >= len(series):
                continue
            start_close = float(series.iloc[start_idx])
            end_close = float(series.iloc[end_idx])
            if start_close <= 0:
                continue
            ret = (end_close / start_close - 1) * 100

            # 시장 대비 초과수익 (같은 날짜 anchor + 같은 forward_days)
            signal_ret = ret
            if use_excess and mser is not None and not mser.empty:
                m_start = _find_idx_on_or_after(mser, anchor_date)
                if m_start is not None and m_start + forward_days < len(mser):
                    m_s = float(mser.iloc[m_start])
                    m_e = float(mser.iloc[m_start + forward_days])
                    if m_s > 0:
                        signal_ret = ret - (m_e / m_s - 1) * 100

            sample_count += 1
            dates_seen.add(anchor_date)
            stocks_seen.add(code)
            for item_name, item_score in scores.items():
                iv = int(item_score)
                per_item[item_name][iv].append(ret)
                ic_x[item_name].append(iv)
                ic_y[item_name].append(signal_ret)

    item_stats: dict[str, dict] = {}
    for item, score_map in per_item.items():
        by_score = {}
        for score_val, returns in sorted(score_map.items()):
            by_score[score_val] = {
                "count": len(returns),
                "avg_return": round(mean(returns), 2) if returns else None,
                "median_return": round(median(returns), 2) if returns else None,
                "win_rate": (
                    round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1)
                    if returns else None
                ),
            }
        pos_returns = [r for s, rs in score_map.items() if s > 0 for r in rs]
        neg_returns = [r for s, rs in score_map.items() if s < 0 for r in rs]
        spread = None
        if pos_returns and neg_returns:
            spread = round(mean(pos_returns) - mean(neg_returns), 2)

        item_stats[item] = {
            "by_score": by_score,
            "predictive_spread": spread,
            "rank_ic": _spearman_ic(ic_x[item], ic_y[item]),
            "n": len(ic_x[item]),
            "total_samples": sum(len(rs) for rs in score_map.values()),
        }

    def _rank_key(k):
        ic = item_stats[k]["rank_ic"]
        if ic is not None:
            return (1, ic)
        sp = item_stats[k]["predictive_spread"]
        return (0, sp if sp is not None else -999)

    ranked_items = sorted(item_stats.keys(), key=_rank_key, reverse=True)

    return {
        "item_stats": item_stats,
        "ranked_items": ranked_items,
        "forward_days": forward_days,
        "use_excess": use_excess,
        "total_samples": sample_count,
        "n_dates": len(dates_seen),
        "n_stocks": len(stocks_seen),
    }


# ─────────── 종합점수 walk-forward 검증 ───────────
# verify_item_scores 는 모든 (점수, 수익) 쌍을 한 통에 모아 IC 를 내서 기간 효과가
# 섞인다(어떤 날 시장 전체가 오르면 그날 점수들이 다 좋아 보이는 식). walk-forward 는
# '그날 점수 → 그 이후 수익' 만으로 날짜별 cross-section IC 를 따로 계산하고(=각 기간이
# look-ahead 없는 out-of-sample) 그 IC 시계열을 집계해 평균·표준편차·t값까지 본다.
# 종합점수가 '내일 이후' 수익을 실제로 변별하는지 가장 정직하게 보여주는 지표.
WF_MIN_CROSS_SECTION = 15   # 그 날짜에 최소 이만큼 종목이 있어야 그날 IC 를 계산
WF_MIN_PERIODS = 4          # 최소 이만큼 날짜(기간)가 모여야 집계를 신뢰


def _composite_score_from_entry(e: dict, score_type: str = "total") -> float | None:
    """엔트리의 항목 점수로 종합(또는 단기/중기) 점수를 *현재* 가중치로 재계산.

    가중치가 바뀐 뒤에도 같은 잣대로 비교하기 위해 저장된 total 대신 scores 에서
    weighted_score 로 다시 계산한다(_extract_score 와 동일 철학). scores 없는 옛
    엔트리는 total 종류일 때만 저장값 사용.
    """
    scores = e.get("scores")
    if scores:
        items = [{"name": k, "score": int(v), "max": 1} for k, v in scores.items()]
        if score_type == "short_term":
            return float(weighted_score(items, SHORT_TERM_ITEMS)[0])
        if score_type == "mid_term":
            return float(weighted_score(items, MID_TERM_ITEMS)[0])
        if score_type == "focus":
            return float(weighted_score(items, FOCUS_ITEMS)[0])
        return float(weighted_score(items)[0])
    if score_type == "total" and e.get("total") is not None:
        return float(e["total"])
    return None


def _rank_ic(xs: list[float], ys: list[float]) -> float | None:
    """순위상관(Spearman). 표본 3 미만이거나 x 분산이 없으면 None. (날짜별 호출용)"""
    if len(xs) < 3 or len(set(xs)) < 2:
        return None
    try:
        rx = pd.Series(xs, dtype="float64").rank()
        ry = pd.Series(ys, dtype="float64").rank()
        ic = rx.corr(ry)
    except Exception:
        return None
    if ic is None or ic != ic:
        return None
    return float(ic)


def _aggregate_walk_forward(by_date: dict, min_cross: int = WF_MIN_CROSS_SECTION) -> dict:
    """{날짜: [(점수, forward수익), ...]} → walk-forward 집계. 순수 함수(테스트 용이).

    날짜별로 cross-section rank-IC 와 분위 스프레드(상위1/3 평균 − 하위1/3 평균)를
    계산하고, 그 시계열을 평균/표준편차/t값/양수비율로 요약한다.
    """
    ic_series: list[float] = []
    spread_series: list[float] = []
    periods: list[dict] = []
    n_obs = 0
    for d in sorted(by_date.keys()):
        pairs = [(s, r) for (s, r) in by_date[d] if s is not None and r is not None]
        if len(pairs) < min_cross:
            continue
        ic = _rank_ic([p[0] for p in pairs], [p[1] for p in pairs])
        if ic is None:
            continue
        ordered = sorted(pairs, key=lambda p: p[0], reverse=True)
        k = max(1, len(ordered) // 3)
        spread = mean([r for _, r in ordered[:k]]) - mean([r for _, r in ordered[-k:]])
        ic_series.append(ic)
        spread_series.append(spread)
        periods.append({"date": d, "n": len(pairs), "ic": round(ic, 3),
                        "spread": round(spread, 2)})
        n_obs += len(pairs)

    n = len(ic_series)
    if n == 0:
        return {"n_periods": 0, "n_obs": 0, "insufficient": True, "mean_ic": None,
                "ic_ir": None, "t_stat": None, "pct_ic_positive": None,
                "mean_spread": None, "pct_spread_positive": None, "periods": []}

    mean_ic = mean(ic_series)
    std_ic = stdev(ic_series) if n > 1 else 0.0
    ic_ir = (mean_ic / std_ic) if std_ic > 0 else None
    t_stat = (ic_ir * (n ** 0.5)) if ic_ir is not None else None
    return {
        "n_periods": n,
        "n_obs": n_obs,
        "insufficient": n < WF_MIN_PERIODS,
        "mean_ic": round(mean_ic, 3),
        "ic_ir": round(ic_ir, 2) if ic_ir is not None else None,
        "t_stat": round(t_stat, 2) if t_stat is not None else None,
        "pct_ic_positive": round(sum(1 for x in ic_series if x > 0) / n * 100, 1),
        "mean_spread": round(mean(spread_series), 2),
        "pct_spread_positive": round(sum(1 for x in spread_series if x > 0) / n * 100, 1),
        "periods": periods,
    }


def verify_composite_walk_forward(
    forward_days: int = 5,
    use_excess: bool = True,
    score_type: str = "total",
) -> dict:
    """종합점수의 walk-forward 예측력. score_history 의 날짜별 점수 → forward 수익으로
    날짜별 IC 시계열을 만들어 집계. (가격은 verify_item_scores 와 동일 인프라 사용)

    반환: _aggregate_walk_forward 결과 + forward_days/use_excess/score_type.
    """
    all_hist = history.load_all(include_archive=True)  # 아카이브 포함 — 장기 검증 창
    today = _today_key()
    market_cache: dict[str, "pd.Series | None"] = {}

    def _market(m: str):
        if m not in market_cache:
            market_cache[m] = _fetch_market_series(m, today) if use_excess else None
        return market_cache[m]

    by_date: dict[str, list] = defaultdict(list)
    for code, entries in all_hist.items():
        if not entries:
            continue
        series = _fetch_close_series(code, today)
        if series is None or series.empty:
            continue
        mser = _market(_safe_market_for_code(code))
        for e in entries:
            anchor = e.get("date")
            if not anchor:
                continue
            score = _composite_score_from_entry(e, score_type)
            if score is None:
                continue
            start_idx = _find_idx_on_or_after(series, anchor)
            if start_idx is None:
                continue
            end_idx = start_idx + forward_days
            if end_idx >= len(series):
                continue
            s_c = float(series.iloc[start_idx])
            e_c = float(series.iloc[end_idx])
            if s_c <= 0:
                continue
            ret = (e_c / s_c - 1) * 100
            if use_excess and mser is not None and not mser.empty:
                m_start = _find_idx_on_or_after(mser, anchor)
                if m_start is not None and m_start + forward_days < len(mser):
                    m_s = float(mser.iloc[m_start])
                    m_e = float(mser.iloc[m_start + forward_days])
                    if m_s > 0:
                        ret -= (m_e / m_s - 1) * 100
            by_date[anchor].append((float(score), ret))

    result = _aggregate_walk_forward(by_date)
    result["forward_days"] = forward_days
    result["use_excess"] = use_excess
    result["score_type"] = score_type
    return result
