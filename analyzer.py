"""
주식 분석 리포트 - 분석 엔진
한 종목의 가격/재무 데이터를 받아 항목 점수화 + 한글 리포트 생성
"""
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from typing import TypedDict
import FinanceDataReader as fdr
import pandas as pd

import dart
import naver
import llm
import history


@lru_cache(maxsize=1)
def _krx_listing() -> pd.DataFrame:
    return fdr.StockListing("KRX")


def get_shares_outstanding(code: str) -> int | None:
    df = _krx_listing()
    row = df[df["Code"] == code]
    if row.empty:
        return None
    val = row.iloc[0].get("Stocks")
    try:
        return int(val) if pd.notna(val) else None
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def all_korean_stocks() -> dict[str, str]:
    """KRX 전 종목 {코드: 이름} 매핑"""
    df = _krx_listing()
    return dict(zip(df["Code"].astype(str), df["Name"].astype(str)))


UNIVERSE_LABELS = {
    "safe":     "🛡️ 안전 유니버스 (시총 5조+ / 거래대금 500억+ / 관리종목 제외)",
    "kospi_30": "KOSPI 시가총액 TOP 30",
    "kospi_50": "KOSPI 시가총액 TOP 50",
}


# ─────────── 점수 항목 분류 ───────────
# 시간 축이 맞는 항목끼리 부분합을 따로 계산해 매매 기간에 맞는 신호를 본다.
# 학술·실무 근거:
#   - 단기(1~4주) 예측력 있는 팩터: 거래량, 수급, 공시, 시장 상대강도, 시장 국면
#   - 중기(분기~) 예측력 있는 팩터: 추세, 가치, 재무 건전성, 성장성
#   - 의심 항목: 모멘텀(RSI 14), 가격 리스크
#     - RSI(14)는 학술적으로 약한 mean-reversion 신호. 한국시장에서 검증 필요.
#     - 가격 리스크는 강세 종목을 깎아서 단기 momentum factor를 거꾸로 작용할 가능성.
#     - 일단 종합점수엔 남기되 부분합엔 미포함 → 시뮬레이션 비교로 검증 후 거취 결정.
SHORT_TERM_ITEMS = {"거래량", "수급", "공시", "시장 상대강도", "시장 국면"}
MID_TERM_ITEMS = {"추세", "가치", "재무 건전성", "성장성"}

# 매수/매도 판단에 쓰는 종합점수는 단순 합산보다 가격 반응 가능성이 큰 항목에
# 더 높은 가중치를 둔다. 실제 예측력은 앱의 점수 시뮬레이션으로 계속 검증한다.
SCORE_WEIGHTS: dict[str, int] = {
    # 가격 반응이 즉시 나타나는 단기 신호 — 2배 가중
    "시장 상대강도": 2,
    "수급": 2,
    "거래량": 2,
    "공시": 2,
    # 시장 국면은 모든 종목에 같은 ±이 곱해져 종합점수가 시장 trend에 과도 의존 →
    # 1배로 낮춤. 신호 가치는 있되 종목 selection에 미치는 비중은 줄임.
    "시장 국면": 1,
    "추세": 1,
    "모멘텀": 1,
    "가격 리스크": 1,
    "가치": 1,
    "재무 건전성": 1,
    "성장성": 1,
}


def weighted_score(
    scores: list[dict],
    item_names: set[str] | None = None,
) -> tuple[int, int]:
    """Return weighted score and weighted positive max for the selected items."""
    total = 0
    max_possible = 0
    for item in scores:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if item_names is not None and name not in item_names:
            continue
        weight = SCORE_WEIGHTS.get(str(name), 1)
        total += int(item.get("score", 0)) * weight
        max_possible += max(0, int(item.get("max", 0))) * weight
    return total, max_possible


# 작전주·관리종목 위험 분류는 자동 제외 (KOSDAQ 우량/중견/기술성장은 정상)
EXCLUDED_DEPTS = {
    "관리종목(소속부없음)",
    "투자주의환기종목(소속부없음)",
    "SPAC(소속부없음)",       # 기업인수목적회사 — 일반 종목 아님
    "외국기업(소속부없음)",    # 외국 기업은 DART 재무 형식 다름
}


def get_safe_universe_codes(
    min_marcap: int = 5_000_000_000_000,   # 5조원
    min_amount: int = 50_000_000_000,        # 500억원 (일별 거래대금)
) -> list[str]:
    """
    작전주 위험이 낮은 안전 유니버스.

    필터:
      - 시가총액 ≥ min_marcap (기본 5조원) → 대형주
      - 일별 거래대금 ≥ min_amount (기본 500억원) → 유동성 충분
      - 위험 분류 제외 (관리종목·투자주의환기·SPAC·외국기업)
      - KOSPI 정상 + KOSDAQ 우량/중견/기술성장 모두 포함
    """
    df = _krx_listing()
    if df is None or df.empty:
        return []

    dept_clean = df["Dept"].fillna("")
    sub = df[
        (df["Marcap"] >= min_marcap)
        & (df["Amount"] >= min_amount)
        & (~dept_clean.isin(EXCLUDED_DEPTS))
    ]
    return sub.sort_values("Marcap", ascending=False)["Code"].astype(str).tolist()


@lru_cache(maxsize=16)
def get_universe_codes(universe: str) -> list[str]:
    """미리 정의된 유니버스의 종목 코드 리스트 (시가총액 내림차순)."""
    if universe == "safe":
        return get_safe_universe_codes(5_000_000_000_000, 50_000_000_000)

    df = _krx_listing()
    if df is None or df.empty:
        return []

    spec = {
        "kospi_30": ("KOSPI", 30),
        "kospi_50": ("KOSPI", 50),
    }
    if universe not in spec:
        return []
    market, n = spec[universe]
    sub = df[df["Market"] == market] if market else df
    return sub.nlargest(n, "Marcap")["Code"].astype(str).tolist()


def _build_history(df: pd.DataFrame, days: int = 120) -> list[dict]:
    h = df.copy()
    h["20일선"] = h["Close"].rolling(20).mean()
    h["60일선"] = h["Close"].rolling(60).mean()
    h = h.tail(days).reset_index()
    h["Date"] = h["Date"].dt.strftime("%Y-%m-%d")
    out = h[["Date", "Close", "20일선", "60일선"]].rename(columns={"Close": "종가"})
    return out.to_dict("records")


KOREAN_NAMES = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "035420": "NAVER",
    "035720": "카카오",
    "005380": "현대차",
    "051910": "LG화학",
    "207940": "삼성바이오로직스",
    "005490": "POSCO홀딩스",
    "068270": "셀트리온",
    "012330": "현대모비스",
    "105560": "KB금융",
    "055550": "신한지주",
    "066570": "LG전자",
    "017670": "SK텔레콤",
    "030200": "KT",
}


class ScoreItem(TypedDict):
    name: str
    score: int
    msg: str
    max: int


class AnalysisResult(TypedDict):
    code: str
    name: str
    last_date: str
    last_close: float
    change_pct: float
    scores: list[ScoreItem]
    total: int  # 매매 목적 가중 종합점수
    # 단기·중기 부분합 — 시간 축이 맞는 항목만 집계 (SHORT_TERM_ITEMS / MID_TERM_ITEMS)
    short_term_score: int
    short_term_max: int
    mid_term_score: int
    mid_term_max: int
    opinion: str
    error: str | None
    history: list[dict]  # [{"Date": "YYYY-MM-DD", "종가": .., "20일선": .., "60일선": ..}, ...]
    disclosures: list[dict]  # 최근 30일 DART 공시 목록
    news: list[dict]  # 최근 종목 뉴스 (참고용, 점수 영향 없음)
    preliminary: dict | None  # 잠정실적공시에서 추출한 최신 실적 (있으면)
    trade_plan: dict
    sources: dict  # 데이터 출처 + 기준 시점


def fetch_price_data(code: str, days: int = 200) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=days + 100)
    df = fdr.DataReader(code, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    return df.tail(days).copy()


def calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def score_trend(df: pd.DataFrame) -> ScoreItem:
    close = df["Close"]
    ma60 = close.rolling(60).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    last = close.iloc[-1]

    if last > ma60 and ma20 > ma60:
        score, msg = 1, f"60일선({ma60:,.0f}원) 위, 중기 흐름 양호"
    elif last > ma60:
        score, msg = 0, "60일선은 위지만 단기선이 약함"
    elif last < ma60 and ma20 < ma60:
        score, msg = -1, f"60일선({ma60:,.0f}원) 아래, 중기 흐름 약함"
    else:
        score, msg = 0, "방향성 불분명"
    return {"name": "추세", "score": score, "msg": msg, "max": 1}


def score_momentum(df: pd.DataFrame) -> ScoreItem:
    rsi = calc_rsi(df["Close"])
    if pd.isna(rsi):
        score, msg = 0, "RSI 산출 불가, 모멘텀 중립"
    elif rsi >= 80:
        score, msg = -1, f"RSI {rsi:.0f}, 극단 과열로 단기 되돌림 주의"
    elif rsi >= 70:
        score, msg = 0, f"RSI {rsi:.0f}, 강하지만 과열권"
    elif 55 <= rsi < 70:
        score, msg = 1, f"RSI {rsi:.0f}, 상승 모멘텀 유효"
    elif 45 <= rsi < 55:
        score, msg = 0, f"RSI {rsi:.0f}, 중립"
    elif 30 <= rsi < 45:
        score, msg = -1, f"RSI {rsi:.0f}, 하락 모멘텀 우세"
    else:
        score, msg = -1, f"RSI {rsi:.0f}, 과매도지만 추세 확인 전 매수 신호 제외"
    return {"name": "모멘텀", "score": score, "msg": msg, "max": 1}


def _fmt_vol(n: float) -> str:
    """거래량을 한국식 단위로 포맷 (만/억주)"""
    if not n:
        return "0주"
    if n >= 100_000_000:
        return f"{n / 100_000_000:.1f}억주"
    if n >= 10_000:
        return f"{n / 10_000:,.0f}만주"
    return f"{n:,.0f}주"


def score_volume(df: pd.DataFrame) -> ScoreItem:
    vol = df["Volume"]
    today_vol = vol.iloc[-1]
    avg20_raw = vol.tail(20).mean()

    # 데이터 이상치 감지 — 마지막 거래량이 평균의 5% 미만이면 장중 부분 데이터·
    # 단일가매매·거래정지 등으로 보고 전일 데이터로 fallback.
    data_anomaly = False
    if avg20_raw and today_vol < avg20_raw * 0.05 and len(vol) >= 21:
        today_vol = vol.iloc[-2]
        avg20 = vol.iloc[-21:-1].mean()
        data_anomaly = True
    else:
        avg20 = avg20_raw

    ratio = today_vol / avg20 if avg20 else 0

    close = df["Close"]
    if data_anomaly and len(close) > 6:
        ret5 = (close.iloc[-2] / close.iloc[-7] - 1) * 100
    else:
        ret5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0

    detail = f"오늘 {_fmt_vol(today_vol)} / 20일 평균 {_fmt_vol(avg20)}"

    if ratio >= 1.5 and ret5 > 0:
        score, label = 1, "매수세 유입"
    elif ratio >= 1.5 and ret5 < 0:
        score, label = -1, "거래량 급증하며 하락, 매도 압력"
    elif ratio < 0.6 and ret5 > 3:
        score, label = -1, "단기 상승에 거래량 부족, 신뢰도 약함"
    elif ratio < 0.6 and ret5 < -3:
        score, label = 0, "거래량 부족 — 매도 압력도 약함"
    elif ratio < 0.6:
        score, label = 0, "시장 관심 약함"
    else:
        score, label = 0, "평소 수준"

    msg = f"{ratio:.2f}배 · {label} ({detail})"
    if data_anomaly:
        msg += " · 전일 기준 (오늘 거래량 미완성)"

    return {"name": "거래량", "score": score, "msg": msg, "max": 1}


def score_supply(summary: dict) -> ScoreItem | None:
    """수급: 외국인/기관 5일 누적 순매수 + 연속 순매수일"""
    f5 = summary.get("foreign_net_5d")
    i5 = summary.get("inst_net_5d")
    streak_f = summary.get("foreign_buy_streak", 0)
    ratio_change = summary.get("ratio_change_20d")

    if f5 is None and i5 is None:
        return None

    f5 = f5 or 0
    i5 = i5 or 0

    if f5 > 0 and i5 > 0:
        score, base = 1, "외국인+기관 동반 순매수"
    elif f5 < 0 and i5 < 0:
        score, base = -1, "외국인+기관 동반 순매도"
    elif f5 > 0:
        if streak_f >= 5:
            score, base = 1, f"외국인 {streak_f}일 연속 순매수"
        else:
            score, base = 0, "외국인 순매수, 기관 순매도 (엇갈림)"
    elif f5 < 0:
        if streak_f <= -5:
            score, base = -1, f"외국인 {-streak_f}일 연속 순매도"
        else:
            score, base = 0, "외국인 순매도, 기관 순매수 (엇갈림)"
    else:
        score, base = 0, "수급 중립"

    if ratio_change is not None and abs(ratio_change) >= 0.5:
        sign = "+" if ratio_change > 0 else ""
        base += f", 외국인 보유율 20일간 {sign}{ratio_change:.1f}%p"

    return {"name": "수급", "score": score, "msg": base, "max": 1}


def score_value(per: float | None, pbr: float | None) -> ScoreItem | None:
    """가치 평가: PER/PBR 기준. 둘 다 없으면 None 반환."""
    if per is None and pbr is None:
        return None

    parts = []
    score = 0

    if per is None:
        parts.append("PER 산출 불가 (적자 또는 데이터 부족)")
    elif per <= 0:
        parts.append(f"PER 의미 없음 (이익이 적자)")
        score -= 1
    elif per < 10:
        parts.append(f"PER {per:.1f} 저평가")
        score += 1
    elif per <= 20:
        parts.append(f"PER {per:.1f} 보통")
    else:
        parts.append(f"PER {per:.1f} 고평가")
        score -= 1

    if pbr is None:
        parts.append("PBR 산출 불가")
    elif 0 < pbr < 1:
        parts.append(f"PBR {pbr:.2f} 자본 대비 저평가")
        score += 1
    elif pbr <= 2:
        parts.append(f"PBR {pbr:.2f} 보통")
    elif pbr > 2:
        parts.append(f"PBR {pbr:.2f} 자본 대비 고평가")
        score -= 1

    score = max(-1, min(1, score))
    return {"name": "가치", "score": score, "msg": ", ".join(parts), "max": 1}


def score_health(roe: float | None, debt_ratio: float | None) -> ScoreItem | None:
    """재무 건전성: ROE + 부채비율"""
    if roe is None and debt_ratio is None:
        return None

    parts = []
    score = 0

    if roe is not None:
        if roe >= 15:
            parts.append(f"ROE {roe:.1f}% 우수")
            score += 1
        elif roe >= 8:
            parts.append(f"ROE {roe:.1f}% 양호")
        elif roe >= 0:
            parts.append(f"ROE {roe:.1f}% 낮음")
            score -= 1
        else:
            parts.append(f"ROE {roe:.1f}% 적자")
            score -= 1

    if debt_ratio is not None:
        if debt_ratio <= 100:
            parts.append(f"부채비율 {debt_ratio:.0f}% 안정적")
            score += 1
        elif debt_ratio <= 200:
            parts.append(f"부채비율 {debt_ratio:.0f}% 양호")
        else:
            parts.append(f"부채비율 {debt_ratio:.0f}% 높음")
            score -= 1

    score = max(-2, min(2, score))
    return {"name": "재무 건전성", "score": score, "msg": ", ".join(parts), "max": 2}


def score_growth(fin: dict) -> ScoreItem | None:
    """
    성장성: 매출 성장 + 영업이익 흑자/적자 상태와 변화를 종합 판단.
    단순 증가율 %만 보면 적자 축소도 +400% 같이 보일 수 있어 위험.
    """
    revenue = fin.get("revenue")
    revenue_prev = fin.get("revenue_prev")
    op = fin.get("operating_income")
    op_prev = fin.get("operating_income_prev")

    if revenue is None or revenue_prev in (None, 0) or revenue == 0:
        return None

    rev_growth = (revenue / revenue_prev - 1) * 100
    parts: list[str] = []
    score = 0

    # 매출
    if rev_growth >= 15:
        parts.append(f"매출 +{rev_growth:.1f}% 고성장")
        score += 1
    elif rev_growth >= 5:
        parts.append(f"매출 +{rev_growth:.1f}% 성장")
    elif rev_growth >= 0:
        parts.append(f"매출 +{rev_growth:.1f}% 정체")
    else:
        parts.append(f"매출 {rev_growth:.1f}% 역성장")
        score -= 1

    # 영업이익 — 흑자/적자 + 마진 변화 기반
    if op is not None and op_prev is not None:
        profitable_now = op > 0
        profitable_before = op_prev > 0
        margin = op / revenue * 100
        margin_prev = op_prev / revenue_prev * 100

        # 비정상적 마진(±200% 초과)은 DART 파싱 오류 가능성 — 점수 적용 안 함
        # 100% 초과는 지주회사·운용사 등 특수 케이스에서 가능하므로 허용
        if abs(margin) > 200 or abs(margin_prev) > 200:
            parts.append(
                f"⚠️ 영업이익률 {margin_prev:.0f}% → {margin:.0f}% "
                f"(데이터 이상 가능성, 영업이익 점수 미반영)"
            )
            score = max(-2, min(2, score))
            return {"name": "성장성", "score": score, "msg": ", ".join(parts), "max": 2}

        if profitable_now and profitable_before:
            margin_diff = margin - margin_prev
            if margin_diff >= 1.0:
                parts.append(f"영업이익률 {margin_prev:.1f}% → {margin:.1f}% 개선")
                score += 1
            elif margin_diff <= -2.0:
                parts.append(f"영업이익률 {margin_prev:.1f}% → {margin:.1f}% 악화")
                score -= 1
            else:
                parts.append(f"영업이익률 {margin:.1f}% 유지")
        elif profitable_now and not profitable_before:
            parts.append(f"영업이익 흑자전환 (이익률 {margin:.1f}%)")
            score += 2
        elif not profitable_now and profitable_before:
            parts.append(f"영업이익 적자전환 (이익률 {margin:.1f}%)")
            score -= 2
        else:  # 양쪽 모두 적자
            if op > op_prev:
                parts.append(f"영업적자 지속, 손실 폭 축소 ({margin_prev:.1f}% → {margin:.1f}%)")
            else:
                parts.append(f"영업적자 확대 ({margin_prev:.1f}% → {margin:.1f}%)")
                score -= 1
    elif op is not None:
        if op > 0:
            parts.append(f"영업이익 흑자 (이익률 {op/revenue*100:.1f}%)")
        else:
            parts.append(f"영업이익 적자 (이익률 {op/revenue*100:.1f}%)")
            score -= 1

    score = max(-2, min(2, score))
    return {"name": "성장성", "score": score, "msg": ", ".join(parts), "max": 2}


def score_growth_from_preliminary(prelim: dict) -> ScoreItem | None:
    """잠정실적공시에서 추출한 분기 데이터로 성장성 점수 계산.
    같은 score_growth() 로직을 재사용하지만 메시지에 잠정 표시."""
    fin_like = {
        "revenue": prelim.get("revenue"),
        "revenue_prev": prelim.get("revenue_yoy"),
        "operating_income": prelim.get("operating_income"),
        "operating_income_prev": prelim.get("operating_income_yoy"),
    }
    item = score_growth(fin_like)
    if item is None:
        return None
    period = prelim.get("period_label") or "잠정실적"
    item = {**item}
    item["msg"] = f"📢 {period} (잠정): {item['msg']}"
    return item


@lru_cache(maxsize=10)
def _market_index_cached(market: str, date_key: str) -> pd.DataFrame:
    """시장 지수 데이터 캐시 — date_key를 일별로 바꿔 매일 1회 fetch."""
    idx_code = "KS11" if market == "KOSPI" else "KQ11"
    end = datetime.now()
    start = end - timedelta(days=260)
    return fdr.DataReader(idx_code, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


def _market_index_data(market: str) -> pd.DataFrame:
    """오늘 날짜 키로 캐시된 시장 지수 데이터."""
    today = datetime.now().strftime("%Y-%m-%d")
    return _market_index_cached(market, today)


def _stock_market(code: str) -> str:
    """종목의 시장 (KOSPI/KOSDAQ). 못 찾으면 KOSPI로."""
    df = _krx_listing()
    row = df[df["Code"] == code]
    if row.empty:
        return "KOSPI"
    return str(row.iloc[0].get("Market", "KOSPI"))


def score_market_relative(df: pd.DataFrame, code: str) -> ScoreItem | None:
    """
    시장 지수(KOSPI/KOSDAQ) 대비 종목의 20일 상대 수익률.

    한국 시장에서 단기 모멘텀의 가장 검증된 신호 중 하나.
    절대 수익률(+5%)이 시장 -3%인 날엔 사실 +8%p 초과 강세인데,
    이 항목 없으면 점수에 안 반영됨.

    ±5%p 임계는 한국시장 20일 변동성 고려한 보수적 값.
    """
    try:
        market = _stock_market(code)
        market_df = _market_index_data(market)
    except Exception:
        return None

    if len(df) < 21 or market_df is None or len(market_df) < 21:
        return None

    stock_ret = (df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1) * 100
    market_ret = (market_df["Close"].iloc[-1] / market_df["Close"].iloc[-21] - 1) * 100
    relative = stock_ret - market_ret

    if relative >= 5:
        score, label = 1, f"시장 대비 +{relative:.1f}%p 강세"
    elif relative <= -5:
        score, label = -1, f"시장 대비 {relative:.1f}%p 약세"
    else:
        score, label = 0, f"시장 대비 {relative:+.1f}%p 중립"

    msg = f"{label} (20일, 종목 {stock_ret:+.1f}% vs {market} {market_ret:+.1f}%)"
    return {"name": "시장 상대강도", "score": score, "msg": msg, "max": 1}


def score_market_regime(code: str) -> ScoreItem | None:
    """
    KOSPI/KOSDAQ 자체의 추세 필터.
    상승장에서는 종목 모멘텀이 더 잘 이어지고, 하락장에서는 좋은 종목도 동반 하락할
    위험이 커서 매수 점수 신뢰도를 낮춘다.
    """
    try:
        market = _stock_market(code)
        market_df = _market_index_data(market)
    except Exception:
        return None

    if market_df is None or len(market_df) < 60:
        return None

    close = market_df["Close"].dropna()
    if len(close) < 60:
        return None

    last = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1])
    ret20 = (last / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0
    ret60 = (last / float(close.iloc[-61]) - 1) * 100 if len(close) >= 61 else 0

    if last > ma60 and ma20 > ma60 and ret20 > 0:
        score, label = 1, "상승 국면"
    elif last < ma60 and ma20 < ma60 and ret20 < 0:
        score, label = -1, "하락 국면"
    elif ret60 < -8 and last < ma60:
        score, label = -1, "약세 지속"
    else:
        score, label = 0, "중립/혼조"

    msg = (
        f"{market} {label} "
        f"(20일 {ret20:+.1f}%, 60일 {ret60:+.1f}%, 지수 {last:,.0f} vs 60일선 {ma60:,.0f})"
    )
    return {"name": "시장 국면", "score": score, "msg": msg, "max": 1}


def score_risk(df: pd.DataFrame) -> ScoreItem:
    close = df["Close"]
    vol = df["Volume"]
    ret1 = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) >= 2 else 0
    ret3 = (close.iloc[-1] / close.iloc[-4] - 1) * 100 if len(close) >= 4 else 0
    ret5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
    high20 = close.tail(20).max()
    drawdown = (close.iloc[-1] / high20 - 1) * 100
    high52w = close.tail(252).max() if len(close) >= 252 else close.max()
    proximity = close.iloc[-1] / high52w * 100  # 100 = 52주 고점
    rsi = calc_rsi(close)

    # 거래량 동반 여부 — 급등에 거래량이 안 붙으면 컨빅션 약함
    avg20_vol = vol.tail(20).mean() if len(vol) >= 20 else vol.mean()
    today_vol = vol.iloc[-1]
    vol_ratio = (today_vol / avg20_vol) if avg20_vol else 1.0

    flags: list[str] = []
    score = 0
    surge_flagged = False  # 거래량 가중 페널티 트리거 여부

    # 5일 누적 — 강도별 차등 (15~25% / 25~40% / 40%+)
    if ret5 >= 40:
        score -= 3
        flags.append(f"5일 +{ret5:.1f}% 극단 급등")
        surge_flagged = True
    elif ret5 >= 25:
        score -= 2
        flags.append(f"5일 +{ret5:.1f}% 강한 급등")
        surge_flagged = True
    elif ret5 >= 15:
        score -= 2
        flags.append(f"5일 +{ret5:.1f}% 급등")
        surge_flagged = True
    elif ret5 >= 8:
        score -= 1
        flags.append(f"5일 +{ret5:.1f}% 단기 상승폭 확대")
        surge_flagged = True

    # 3일 누적 — 5일 임계 미달이라도 3일에 응축된 상승 잡기
    if ret3 >= 10 and ret5 < 15:
        score -= 1
        flags.append(f"3일 +{ret3:.1f}% 단기 응축 상승")
        surge_flagged = True

    # 1일 큰 양봉 — 직전 하루에 큰 폭으로 튄 경우
    if ret1 >= 5:
        score -= 1
        flags.append(f"전일 +{ret1:.1f}% 큰 양봉")
        surge_flagged = True

    # 거래량 미동반 가중 페널티 — 급등이 있는데 거래량이 평균 80% 미만이면 약한 컨빅션
    if surge_flagged and vol_ratio < 0.8:
        score -= 1
        flags.append(f"거래량 {vol_ratio:.2f}배 — 급등에 거래량 미동반")

    if rsi >= 80:
        score -= 1
        flags.append(f"RSI {rsi:.0f} 극단적 과열")

    if proximity >= 95:
        score -= 1
        flags.append(f"52주 고점 {proximity:.1f}% 근접")

    if drawdown <= -10:
        score -= 1
        flags.append(f"20일 고점 대비 {drawdown:.1f}% 조정")

    # 다중 과열 신호 결합 가산 — 급등 + RSI 극단 + 52w 고점이 동시면 위험 가중
    if surge_flagged and rsi >= 80 and proximity >= 95:
        score -= 1
        flags.append("3중 과열 결합")

    score = max(-3, score)  # max 항목값(0) 안에서 -3까지

    if not flags:
        msg = f"단기 변동성 양호 (5일 {ret5:+.1f}%, RSI {rsi:.0f}, 52주 고점 대비 {proximity:.0f}%)"
    else:
        msg = ", ".join(flags)

    return {"name": "가격 리스크", "score": score, "msg": msg, "max": 0}


def calc_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    """Average True Range for stop-loss distance."""
    needed = {"High", "Low", "Close"}
    if df is None or len(df) < period + 1 or not needed.issubset(df.columns):
        return None
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean().iloc[-1]
    if pd.isna(atr) or atr <= 0:
        return None
    return float(atr)


def trade_signal_from_scores(
    scores: list[dict],
    total: int,
    short_term_score: int,
    max_possible: int,
) -> dict:
    """Convert weighted score components into a trading-oriented signal."""
    score_map = {
        s.get("name"): int(s.get("score", 0))
        for s in scores
        if isinstance(s, dict)
    }
    market_score = score_map.get("시장 국면", 0)
    risk_score = score_map.get("가격 리스크", 0)
    ratio = total / max_possible if max_possible > 0 else 0

    # 라벨은 처방형("매수하세요")이 아닌 서술형("긍정 우세") — 사용자 판단 영역 보존.
    if total >= 8 and short_term_score >= 4 and market_score >= 0 and risk_score >= -1:
        action = "긍정 신호 강함"
        confidence = "높음"
        reason = "가중 종합점수와 단기 수급·가격 반응 신호가 함께 우호적"
    elif total >= 4 and market_score >= 0 and risk_score >= -2:
        action = "긍정 우세"
        confidence = "보통"
        reason = "긍정 신호가 우세하지만 일부 리스크 확인 필요"
    elif total <= -5 or risk_score <= -3 or market_score < 0:
        action = "위험 신호 강함"
        confidence = "높음" if total <= -5 else "보통"
        reason = "하락 국면 또는 가격 리스크가 커서 보유 비중·진입 시점 점검 필요"
    elif total <= -1 or ratio < 0:
        action = "위험 우세"
        confidence = "보통"
        reason = "매수 근거보다 리스크 신호가 우세"
    else:
        action = "중립"
        confidence = "낮음"
        reason = "긍정·부정 신호가 충분히 갈리지 않음"

    return {
        "action": action,
        "confidence": confidence,
        "reason": reason,
    }


def build_trade_plan(
    df: pd.DataFrame,
    scores: list[dict],
    total: int,
    short_term_score: int,
    max_possible: int,
) -> dict:
    """
    Score-based trade plan. It is a risk-management guide, not a price guarantee.
    """
    last = float(df["Close"].iloc[-1])
    atr = calc_atr(df)
    signal = trade_signal_from_scores(scores, total, short_term_score, max_possible)

    stop_loss = None
    target_1r = None
    target_2r = None
    risk_pct = None
    if atr:
        ma20 = df["Close"].rolling(20).mean().iloc[-1] if len(df) >= 20 else None
        atr_stop = last - 2 * atr
        ma_stop = float(ma20) * 0.97 if ma20 is not None and not pd.isna(ma20) else atr_stop
        stop_loss = min(last * 0.97, max(atr_stop, ma_stop))
        if stop_loss >= last:
            stop_loss = atr_stop
        risk = last - stop_loss
        if risk > 0:
            target_1r = last + risk
            target_2r = last + 2 * risk
            risk_pct = risk / last * 100

    return {
        **signal,
        "entry_price": last,
        "stop_loss": round(stop_loss, 2) if stop_loss else None,
        "target_1r": round(target_1r, 2) if target_1r else None,
        "target_2r": round(target_2r, 2) if target_2r else None,
        "risk_pct": round(risk_pct, 2) if risk_pct is not None else None,
        "atr14": round(atr, 2) if atr else None,
    }


def recompute_score_after_deep(result: dict) -> None:
    """
    깊이 분석이 공시 카테고리를 바꿨을 수 있으니 종합점수·의견을 재계산.
    예: 제목만 보면 'negative' (유상증자결정) → 본문 보면 'positive' (신사업 투자).
    """
    disclosures = result.get("disclosures") or []
    cats = [d.get("category", "neutral") for d in disclosures]
    n_crit = cats.count("critical")
    n_neg = cats.count("negative")
    n_pos = cats.count("positive")

    score = 0
    parts: list[str] = []
    if n_crit > 0:
        score = -2
        parts.append(f"중대 공시 {n_crit}건")
    elif n_neg > 0:
        score = -1
        parts.append(f"부정 공시 {n_neg}건")
    if n_pos > 0:
        if score >= 0:
            score = 1
        parts.append(f"긍정 공시 {n_pos}건")

    if not parts:
        msg = f"최근 30일 routine 공시만 ({len(disclosures)}건)"
    else:
        msg = ", ".join(parts)

    new_item: ScoreItem = {"name": "공시", "score": score, "msg": msg, "max": 1}
    scores = result.get("scores", [])
    replaced = False
    for i, s in enumerate(scores):
        if s.get("name") == "공시":
            scores[i] = new_item
            replaced = True
            break
    if not replaced:
        scores.append(new_item)

    total, max_possible = weighted_score(scores)
    result["total"] = total
    result["opinion"] = overall_opinion(total, max_possible)

    # 공시 점수 변경에 따라 단기 부분합도 재계산 (공시는 SHORT_TERM_ITEMS 소속)
    short_term_score, short_term_max = weighted_score(scores, SHORT_TERM_ITEMS)
    mid_term_score, mid_term_max = weighted_score(scores, MID_TERM_ITEMS)
    result["short_term_score"] = short_term_score
    result["short_term_max"] = short_term_max
    result["mid_term_score"] = mid_term_score
    result["mid_term_max"] = mid_term_max
    if result.get("trade_plan"):
        result["trade_plan"].update(
            trade_signal_from_scores(scores, total, short_term_score, max_possible)
        )

    # 깊이 분석으로 점수가 바뀌었을 수 있으니 history 스냅샷도 새 값으로 덮어씀
    # → 검증 도구가 "사용자에게 실제로 보여진 점수" 기준으로 수익률 계산하게 됨
    try:
        history.record_snapshot(
            result.get("code", ""),
            total,
            float(result.get("last_close", 0) or 0),
            result.get("opinion", ""),
            scores=scores,
        )
    except Exception:
        pass


def enrich_with_deep_analysis(result: dict, top_n: int = 3) -> None:
    """
    이미 분석된 결과(result)의 중요 공시(critical/negative/positive 상위 N개)에
    LLM 깊이 분석(rationale·key_points)을 추가. In-place 수정.

    스크리닝은 빠르게 돌리기 위해 깊이 분석 생략(deep_top=0)하고,
    통과한 종목에만 사후적으로 이 함수로 깊이 분석을 추가하는 2단계 구조에 사용.
    """
    disclosures = result.get("disclosures") or []
    if not disclosures or not llm.is_configured():
        return

    priority = {"critical": 0, "negative": 1, "positive": 2}
    important = [
        d for d in disclosures
        if d.get("category") in priority
        and d.get("rcept_no")
        and "자회사의 주요경영사항" not in d.get("title", "")
    ]
    important.sort(key=lambda d: (
        priority.get(d.get("category", ""), 99),
        -int(d.get("rcept_no", "0") or 0),
    ))

    cache = llm.get_cache()
    for d in important[:top_n]:
        rcept_no = d.get("rcept_no", "")
        # 이미 깊이 분석돼있으면 캐시에서 가져와 머지
        if cache.has_pro_analysis(rcept_no):
            cached = cache.get(rcept_no)
            if cached:
                d.update(cached)
            continue
        try:
            content = dart.get_disclosure_content(rcept_no)
            if not content:
                continue
            deep_result = llm.deep_analyze(
                rcept_no,
                d.get("title", ""),
                d.get("submitter", ""),
                d.get("date", ""),
                content,
            )
            d.update(deep_result)
        except Exception:
            pass


def analyze_and_score_disclosures(
    disclosures: list[dict],
    deep_analysis_top_n: int = 3,
    use_llm: bool = True,
) -> tuple[list[dict], ScoreItem | None]:
    """
    공시 목록을 LLM으로 분류·요약하고 점수를 산출.
    - 1단계: 모든 공시를 Flash로 제목 분류 (rcept_no 캐시)
    - 2단계: 중요(non-neutral) 공시 상위 N개를 Pro로 본문 깊이 분석
    - 3단계: 카테고리별 개수 → 점수 매핑
    """
    if not disclosures:
        return [], None

    enriched: list[dict] = []
    llm_ready = use_llm and llm.is_configured()

    # 1단계: 분류 (LLM 일괄 호출 — 15회 → 1회로 압축)
    if llm_ready:
        try:
            batch = llm.classify_titles_batch(disclosures)
            enriched = [{**d, **r} for d, r in zip(disclosures, batch)]
        except Exception:
            enriched = [
                {**d, **llm.classify_by_rule(d.get("title", ""))} for d in disclosures
            ]
    else:
        enriched = [
            {**d, **llm.classify_by_rule(d.get("title", ""))} for d in disclosures
        ]

    # 2단계: 중요 공시 본문 깊이 분석
    if llm_ready:
        priority = {"critical": 0, "negative": 1, "positive": 2}
        important = [
            d for d in enriched
            if d.get("category") in priority and d.get("rcept_no")
        ]
        important.sort(key=lambda d: priority.get(d["category"], 99))

        for d in important[:deep_analysis_top_n]:
            cache = llm.get_cache()
            if cache.has_pro_analysis(d["rcept_no"]):
                d.update(cache.get(d["rcept_no"]))
                continue
            try:
                content = dart.get_disclosure_content(d["rcept_no"])
                if not content:
                    continue
                pro_result = llm.deep_analyze(
                    d["rcept_no"], d["title"],
                    d.get("submitter", ""), d.get("date", ""), content,
                )
                d.update(pro_result)
            except Exception:
                pass  # Flash 분류 결과 유지

    # 3단계: 점수 산출
    cats = [d.get("category", "neutral") for d in enriched]
    n_critical = cats.count("critical")
    n_negative = cats.count("negative")
    n_positive = cats.count("positive")

    score = 0
    parts: list[str] = []
    if n_critical > 0:
        score = -2
        parts.append(f"중대 공시 {n_critical}건")
    elif n_negative > 0:
        score = -1
        parts.append(f"부정 공시 {n_negative}건")
    if n_positive > 0:
        if score >= 0:
            score = 1
        parts.append(f"긍정 공시 {n_positive}건")

    if not parts:
        msg = f"최근 30일 routine 공시만 ({len(enriched)}건)"
    else:
        msg = ", ".join(parts)

    return enriched, {"name": "공시", "score": score, "msg": msg, "max": 1}


def overall_opinion(total: int, max_possible: int) -> str:
    """가중 종합점수로 신호 강도 표현. 매수/매도 추천이 아닌 자문 톤 유지.
    최종 판단은 사용자 몫이라는 앱 철학에 맞춰, 명령형이 아닌 서술형으로."""
    ratio = total / max_possible if max_possible > 0 else 0
    if total >= 8 and ratio >= 0.5:
        return "긍정 신호 강함 — 다수 지표가 우호적입니다. 단, 이미 가격에 반영됐을 가능성을 고려하세요."
    if total >= 4 and ratio >= 0.25:
        return "긍정 우세 — 신호는 좋지만 위험 요인도 있어 분할 접근이 안전합니다."
    if total <= -5 or ratio <= -0.3:
        return "위험 신호 강함 — 부정 지표가 우세해 보유 비중·진입 시점 점검이 필요합니다."
    if total <= -1:
        return "위험 우세 — 하락·리스크 신호가 남아 있어 신규 진입은 신중하세요."
    return "중립 — 긍정·부정 신호가 비슷해 추가 확인이 필요합니다."


def _resolve_name(code: str) -> str:
    if code in KOREAN_NAMES:
        return KOREAN_NAMES[code]
    try:
        return all_korean_stocks().get(code, code)
    except Exception:
        return code


# 발굴 시점 직전 급등 감지 — 평균 회귀 함정 회피용 하드 제외 필터.
# 점수 시뮬레이션에서 단기 ≥3 그룹이 -4.91% 손실 (3/3 전부 손실)이라는 패턴이
# 관찰돼 추가. 거래량·수급·시장상대강도 점수가 높으면 이미 상승한 종목일 가능성이
# 크고, 발굴 시점이 고점이라 다음날부터 되돌림에 맞기 쉬움.
SURGE_THRESHOLDS = {
    "d1": 6.0,   # 1일 +6% 이상 = 직전 일봉 큰 양봉, 단기 고점 가능성
    "d3": 13.0,  # 3일 누적 +13% 이상 = 단기 급등
    "d5": 20.0,  # 5일 누적 +20% 이상 = 강한 단기 랠리
    "d20": 40.0, # 20일 누적 +40% 이상 = 중기 과열
}


def _compute_recent_surge(df) -> dict:
    """발굴 시점 직전 N일 누적 수익률을 계산하고 임계값 초과 여부 반환.

    반환:
      is_surge: bool — 임계값 하나라도 넘으면 True
      metrics: {"d1": float, "d3": ..., "d5": ..., "d20": ...} — 계산 가능한 것만
      triggers: list[str] — 사람이 읽을 수 있는 트리거 설명 (UI 표시용)
    """
    if df is None or len(df) < 2:
        return {"is_surge": False, "metrics": {}, "triggers": []}

    close = df["Close"]
    last = float(close.iloc[-1])
    metrics: dict[str, float] = {}

    # 각 windows: 마지막 종가 대비 N일 전 종가 누적 수익률
    for label, n_back in [("d1", 1), ("d3", 3), ("d5", 5), ("d20", 20)]:
        if len(close) >= n_back + 1:
            past = float(close.iloc[-(n_back + 1)])
            if past > 0:
                metrics[label] = round((last / past - 1) * 100, 2)

    triggers: list[str] = []
    for label, threshold in SURGE_THRESHOLDS.items():
        val = metrics.get(label)
        if val is not None and val >= threshold:
            label_kr = {"d1": "1일", "d3": "3일", "d5": "5일", "d20": "20일"}[label]
            triggers.append(f"{label_kr} +{val:.1f}%")

    return {
        "is_surge": bool(triggers),
        "metrics": metrics,
        "triggers": triggers,
    }


# 펀더멘털 backed-out 급등 — 단기 급등에도 적자/이익 악화가 동반되면 자동 제외.
# 펀더멘털 없는 급등은 평균 회귀 위험이 더 큼 (memory: 보수적 점수 철학).
_FUNDAMENTAL_SURGE_RET5_MIN = 12.0
_FUNDAMENTAL_SURGE_OP_DECLINE_PCT = -30.0


def _check_fundamental_backed_surge(
    df, recent_surge: dict, fin: dict | None, preliminary: dict | None,
) -> dict:
    """기존 recent_surge에 펀더멘털 backed-out 트리거 추가.

    - 5일 +12% 이상 상승 + (영업적자 or 영업이익 30%+ 악화) → is_surge=True
    - 잠정실적 우선, 없으면 정기 재무 사용.
    """
    if df is None or len(df) < 6:
        return recent_surge

    close = df["Close"]
    past5 = float(close.iloc[-6])
    if past5 <= 0:
        return recent_surge
    ret5 = (float(close.iloc[-1]) / past5 - 1) * 100
    if ret5 < _FUNDAMENTAL_SURGE_RET5_MIN:
        return recent_surge

    op = None
    op_prev = None
    label = None
    if preliminary:
        op = preliminary.get("operating_income")
        op_prev = preliminary.get("operating_income_yoy")
        label = preliminary.get("period_label") or "잠정실적"
    elif fin:
        op = fin.get("operating_income")
        op_prev = fin.get("operating_income_prev")
        label = fin.get("report_label") or "정기 재무"

    if op is None:
        return recent_surge

    trigger = None
    if op < 0:
        trigger = f"{label} 영업적자 + 5일 +{ret5:.1f}% 급등"
    elif op_prev is not None and op_prev > 0:
        decline = (op / op_prev - 1) * 100
        if decline <= _FUNDAMENTAL_SURGE_OP_DECLINE_PCT:
            trigger = f"{label} 영업이익 {decline:.0f}% 악화 + 5일 +{ret5:.1f}% 급등"

    if not trigger:
        return recent_surge

    updated = dict(recent_surge)
    updated["is_surge"] = True
    updated["triggers"] = list(recent_surge.get("triggers", [])) + [trigger]
    updated["fundamental_backed_out"] = True
    return updated


def analyze(code: str, lite: bool = False, deep_top: int = 3) -> AnalysisResult:
    """
    종목 분석.

    lite=True: 빠른 스크리닝용. LLM 공시 본문 분석 + 뉴스 수집 생략.
              점수 자체는 동일하게 산출 (공시는 룰 기반 분류).
              종목당 ~3-5초.

    deep_top: LLM Pro로 본문 깊이 분석할 중요 공시 상위 N개 (lite=False 일 때만).
              스크리닝 시 0으로 호출하면 깊이 분석 스킵 → 종목당 ~10초 절약.
              개별 종목 단독 분석 시 기본 3으로 두면 자세한 근거·요약 받음.
    """
    name = _resolve_name(code)
    _empty_partials = {
        "short_term_score": 0, "short_term_max": 0,
        "mid_term_score": 0, "mid_term_max": 0,
    }
    try:
        df = fetch_price_data(code)
    except Exception as e:
        return {
            "code": code, "name": name, "last_date": "", "last_close": 0,
            "change_pct": 0, "scores": [], "total": 0, "opinion": "",
            **_empty_partials,
            "error": f"데이터 조회 실패: {e}", "history": [], "disclosures": [],
            "news": [], "preliminary": None, "trade_plan": {}, "sources": {},
        }

    if df.empty:
        return {
            "code": code, "name": name, "last_date": "", "last_close": 0,
            "change_pct": 0, "scores": [], "total": 0, "opinion": "",
            **_empty_partials,
            "error": "가격 데이터가 비어 있습니다.", "history": [], "disclosures": [],
            "news": [], "preliminary": None, "trade_plan": {}, "sources": {},
        }

    last_close = float(df["Close"].iloc[-1])
    scores: list[ScoreItem] = [
        score_trend(df),
        score_momentum(df),
        score_volume(df),
        score_risk(df),
    ]

    # 시장 상대강도 (KOSPI/KOSDAQ 대비) — 단기 매매 핵심 신호
    market_rel = score_market_relative(df, code)
    if market_rel is not None:
        scores.append(market_rel)
    market_regime = score_market_regime(code)
    if market_regime is not None:
        scores.append(market_regime)

    flow_last_date: str | None = None
    fin_report_label: str | None = None
    fin: dict | None = None  # 펀더멘털 backed-out 급등 체크용으로 외부 노출
    dart_error: str | None = None  # 사용자에게 노출할 만한 "진짜 오류"만 기록 (ETF 등 단순 미등록은 None 유지)

    # 외국인/기관 수급 (네이버 금융 스크래핑)
    try:
        flow = naver.get_flow_summary(code)
        flow_last_date = flow.get("last_date")
        supply_item = score_supply(flow)
        if supply_item is not None:
            scores.append(supply_item)
    except Exception as e:
        scores.append({
            "name": "수급",
            "score": 0,
            "msg": f"수급 조회 실패: {e}",
            "max": 0,
        })

    # DART 재무 데이터 (키 설정되어 있을 때만)
    # 실패 시 점수 카드는 추가하지 않음 — 노이즈 줄이기.
    # 진짜 오류(네트워크·API)만 dart_error 로 기록해 출처 영역에 표시.
    # ETF/SPAC 처럼 corp_code 자체가 없는 경우는 영구적이라 조용히 무시.
    if dart.is_configured():
        try:
            # Gist 캐시 우선 — DART 직접 호출 실패 시 옛 캐시 fallback.
            # Streamlit Cloud → DART 네트워크가 자주 막혀서 매번 새로 호출하면
            # 점수가 None으로 떨어짐. 6시간 TTL 안이면 캐시 그대로 사용.
            fin = dart.get_financials_cached(code)
            fin_report_label = fin.get("report_label")
            # report_label=None 인데 _fetch_acnt_all 에서 진짜 API 오류가 있었다면
            # 그걸 dart_error 로 노출 (이전엔 조용히 무시돼 사용자가 원인을 못 알아냈음)
            if fin_report_label is None:
                api_err = dart._get_last_acnt_error()
                if api_err:
                    dart_error = api_err
            # 캐시 fallback (stale) 사용 시 안내 메시지 (오류는 아니지만 사용자가 알아야 함)
            if fin.get("_cache_stale"):
                fetched = fin.get("_cache_fetched_at", "?")
                dart_error = f"DART 일시 오류 — 캐시 사용 (저장 시점: {fetched})"
            shares = get_shares_outstanding(code)
            ratios = dart.calc_per_pbr(code, last_close, shares or 0, fin) if shares else {"per": None, "pbr": None}

            for fn, args in [
                (score_value, (ratios["per"], ratios["pbr"])),
                (score_health, (fin["roe"], fin["debt_ratio"])),
            ]:
                item = fn(*args)
                if item is not None:
                    scores.append(item)
            growth_item = score_growth(fin)
            if growth_item is not None:
                scores.append(growth_item)
        except dart.DartError as e:
            msg = str(e)
            # corp_code 미등록 = ETF·SPAC·외국기업 등 → 영구적이므로 조용히
            # 그 외 = 네트워크·키 오류·HTTP 등 진짜 오류 → 사용자에게 노출
            if "등록된 회사가 아닙니다" not in msg:
                dart_error = msg
        except Exception as e:
            dart_error = f"내부 오류: {type(e).__name__}: {e}"

    # DART 공시 — 50개까지 가져와서 (임원 변동 신고가 많은 종목 대응),
    # 화면 표시·점수는 상위 15개로 제한, 잠정실적 검색은 전체 50개에서.
    disclosures: list[dict] = []
    raw_disclosures_full: list[dict] = []
    if dart.is_configured():
        try:
            raw_disclosures_full = dart.get_recent_disclosures(code, days=30, max_count=50)
            disclosures, disc_score = analyze_and_score_disclosures(
                raw_disclosures_full[:15],
                use_llm=not lite,
                deep_analysis_top_n=deep_top,
            )
            if disc_score is not None:
                scores.append(disc_score)
        except Exception:
            disclosures = []
            raw_disclosures_full = []

    # 잠정실적공시 추출 — lite 모드에서도 실행 (1종목당 LLM 1회만 호출, 부담 적음)
    # 안전 유니버스 스크리닝에서 분기 결산 신호 반영하려면 이게 필수.
    preliminary: dict | None = None
    preliminary_used_for_growth = False
    if raw_disclosures_full and llm.is_configured():
        # 50개 공시 전체에서 잠정실적 후보를 찾음 (임원 신고 등으로 묻혀도 발견)
        # 후보 필터링 + 우선순위 정렬:
        #   1. "자회사의 주요경영사항"은 자회사 데이터라 모회사 분석에 부적합 → 제외
        #   2. "정정"이 들어간 공시는 부분 데이터라 후순위 (원본 잠정실적 우선)
        #   3. 같은 우선순위 내에선 rcept_no 큰 순 (= 최신)
        candidates = [
            d for d in raw_disclosures_full
            if dart.is_preliminary_disclosure(d.get("title", ""))
            and "자회사의 주요경영사항" not in d.get("title", "")
        ]
        candidates.sort(
            key=lambda d: (
                1 if "정정" in d.get("title", "") else 0,
                -int(d.get("rcept_no", "0") or 0),
            )
        )

        for d in candidates:
            try:
                content = dart.get_disclosure_content(d.get("rcept_no", ""))
                if not content:
                    continue
                extracted = llm.extract_preliminary_results(
                    d.get("rcept_no", ""), d.get("title", ""), content,
                )
                if extracted and extracted.get("revenue"):
                    preliminary = {
                        **extracted,
                        "disclosure_date": d.get("date", ""),
                        "disclosure_title": d.get("title", ""),
                        "rcept_no": d.get("rcept_no", ""),
                    }
                    break  # 가장 신뢰도 높은 잠정실적 1개 사용
            except Exception:
                continue

    # 정식 보고서가 stale 상태이고 잠정실적이 있으면 성장성 점수를 잠정 기반으로 교체
    fin_freshness = dart.check_data_freshness(fin_report_label) if fin_report_label else {}
    if preliminary and fin_freshness.get("is_stale"):
        prelim_growth = score_growth_from_preliminary(preliminary)
        if prelim_growth is not None:
            for i, s in enumerate(scores):
                if s.get("name") == "성장성":
                    scores[i] = prelim_growth
                    preliminary_used_for_growth = True
                    break
            if not preliminary_used_for_growth:
                scores.append(prelim_growth)
                preliminary_used_for_growth = True

    total, max_possible = weighted_score(scores)

    # 단기·중기 부분합 — 시간 축이 맞는 항목만 합산
    short_term_score, short_term_max = weighted_score(scores, SHORT_TERM_ITEMS)
    mid_term_score, mid_term_max = weighted_score(scores, MID_TERM_ITEMS)

    # 뉴스 (참고용, 점수 영향 없음 — lite 모드는 생략)
    news_items: list[dict] = []
    if not lite:
        try:
            news_items = naver.get_recent_news(code, name=name, max_count=12)
        except Exception:
            pass

    sources = {
        "price_last_date": df.index[-1].strftime("%Y-%m-%d"),
        "flow_last_date": flow_last_date,
        "fin_report_label": fin_report_label,
        "fin_freshness": fin_freshness,
        "fin_cached": bool(fin and fin.get("_cached")) if fin else False,
        "fin_cache_fetched_at": fin.get("_cache_fetched_at") if fin else None,
        "fin_cache_stale": bool(fin and fin.get("_cache_stale")) if fin else False,
        "has_dart": dart.is_configured(),
        "has_llm": llm.is_configured() and not lite,
        "news_count": len(news_items),
        "lite_mode": lite,
        "preliminary_used_for_growth": preliminary_used_for_growth,
        "dart_error": dart_error,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    opinion_text = overall_opinion(total, max_possible)
    trade_plan = build_trade_plan(df, scores, total, short_term_score, max_possible)

    # 매일 점수 스냅샷 저장 (같은 날 중복은 덮어씀)
    # 항목별 점수도 함께 저장 → 검증 도구에서 어떤 항목이 예측력 있었는지 분석 가능
    try:
        history.record_snapshot(code, total, last_close, opinion_text, scores=scores)
    except Exception:
        pass  # 저장 실패해도 분석은 계속

    return {
        "code": code,
        "name": name,
        "last_date": df.index[-1].strftime("%Y.%m.%d"),
        "last_close": last_close,
        "change_pct": float(df["Change"].iloc[-1] * 100),
        "scores": scores,
        "total": total,
        "short_term_score": short_term_score,
        "short_term_max": short_term_max,
        "mid_term_score": mid_term_score,
        "mid_term_max": mid_term_max,
        "opinion": opinion_text,
        "error": None,
        "history": _build_history(df),
        "disclosures": disclosures,
        "news": news_items,
        "preliminary": preliminary,
        "trade_plan": trade_plan,
        "recent_surge": _check_fundamental_backed_surge(
            df, _compute_recent_surge(df), fin, preliminary,
        ),
        "sources": sources,
    }


def render_text(result: AnalysisResult) -> str:
    if result["error"]:
        return f"[오류] {result['name']} {result['code']}: {result['error']}"

    arrow = "▲" if result["change_pct"] >= 0 else "▼"
    lines = []
    lines.append("━" * 40)
    lines.append(f"  {result['name']}  {result['code']}")
    lines.append(f"  {result['last_date']} 기준 분석")
    lines.append("━" * 40)
    lines.append("")
    lines.append(f"  현재가: {result['last_close']:,.0f}원   {arrow} {result['change_pct']:+.2f}%")
    lines.append("")
    lines.append("  [항목별 분석]")

    for s in result["scores"]:
        if s["score"] > 0:
            mark = "▲"
        elif s["score"] < 0:
            mark = "▼"
        else:
            mark = "▬"
        sign = f"{s['score']:+d}" if s["score"] != 0 else " 0"
        lines.append(f"  {s['name']:<10} {mark} {s['msg']:<35} {sign}")

    lines.append("")
    lines.append("━" * 40)
    lines.append(f"  종합점수: {result['total']:+d}점")
    lines.append(f"  종합의견: {result['opinion']}")
    plan = result.get("trade_plan") or {}
    if plan:
        lines.append(
            f"  매매계획: {plan.get('action', '-')} "
            f"(신뢰도 {plan.get('confidence', '-')})"
        )
        if plan.get("stop_loss") and plan.get("target_1r"):
            lines.append(
                f"  손절/목표: {plan['stop_loss']:,.0f}원 / "
                f"{plan['target_1r']:,.0f}원 / {plan.get('target_2r', 0):,.0f}원"
            )
    lines.append("━" * 40)
    return "\n".join(lines)


def main():
    codes = sys.argv[1:] if len(sys.argv) > 1 else ["005930"]
    for code in codes:
        print(render_text(analyze(code)))
        print()


if __name__ == "__main__":
    main()
