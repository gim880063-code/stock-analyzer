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
    total: int
    opinion: str
    error: str | None
    history: list[dict]  # [{"Date": "YYYY-MM-DD", "종가": .., "20일선": .., "60일선": ..}, ...]
    disclosures: list[dict]  # 최근 30일 DART 공시 목록
    news: list[dict]  # 최근 종목 뉴스 (참고용, 점수 영향 없음)
    preliminary: dict | None  # 잠정실적공시에서 추출한 최신 실적 (있으면)
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
    if rsi >= 70:
        score, msg = -1, f"RSI {rsi:.0f}, 단기 과열 주의"
    elif rsi <= 30:
        score, msg = 1, f"RSI {rsi:.0f}, 단기 과매도 반등 가능"
    elif 50 < rsi < 70:
        score, msg = 1, f"RSI {rsi:.0f}, 상승 모멘텀 유효"
    elif 30 < rsi < 50:
        score, msg = 0, f"RSI {rsi:.0f}, 모멘텀 약화"
    else:
        score, msg = 0, f"RSI {rsi:.0f}, 중립"
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

    if revenue is None or revenue_prev in (None, 0):
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


def score_risk(df: pd.DataFrame) -> ScoreItem:
    close = df["Close"]
    ret5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100
    high20 = close.tail(20).max()
    drawdown = (close.iloc[-1] / high20 - 1) * 100
    high52w = close.tail(252).max() if len(close) >= 252 else close.max()
    proximity = close.iloc[-1] / high52w * 100  # 100 = 52주 고점
    rsi = calc_rsi(close)

    flags: list[str] = []
    score = 0

    if ret5 >= 15:
        score -= 2
        flags.append(f"5일 +{ret5:.1f}% 급등")
    elif ret5 >= 8:
        score -= 1
        flags.append(f"5일 +{ret5:.1f}% 단기 상승폭 확대")

    if rsi >= 80:
        score -= 1
        flags.append(f"RSI {rsi:.0f} 극단적 과열")

    if proximity >= 97:
        score -= 1
        flags.append(f"52주 고점 {proximity:.1f}% 근접")

    if drawdown <= -10:
        score -= 1
        flags.append(f"20일 고점 대비 {drawdown:.1f}% 조정")

    score = max(-2, score)  # max 항목값(0) 안에서 -2까지

    if not flags:
        msg = f"단기 변동성 양호 (5일 {ret5:+.1f}%, RSI {rsi:.0f}, 52주 고점 대비 {proximity:.0f}%)"
    else:
        msg = ", ".join(flags)

    return {"name": "가격 리스크", "score": score, "msg": msg, "max": 0}


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
    """max_possible(가능한 양의 점수 합) 대비 비율로 의견 결정"""
    ratio = total / max_possible if max_possible > 0 else 0
    if ratio >= 0.5:
        return "관심 유지 — 다수 지표가 우호적입니다."
    if ratio >= 0.2:
        return "분할 접근 가능 — 일부 위험 요인이 있어 한 번에 매수하기보다 나눠서 접근하세요."
    if ratio >= 0:
        return "중립 — 긍정·부정 신호가 비슷해 추가 확인이 필요합니다."
    if ratio >= -0.3:
        return "관망 — 위험 요인이 많아 진입 시점을 미루는 것이 안전합니다."
    return "리스크 확대 — 다수 지표가 부정적이므로 신규 진입은 신중해야 합니다."


def _resolve_name(code: str) -> str:
    if code in KOREAN_NAMES:
        return KOREAN_NAMES[code]
    try:
        return all_korean_stocks().get(code, code)
    except Exception:
        return code


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
    try:
        df = fetch_price_data(code)
    except Exception as e:
        return {
            "code": code, "name": name, "last_date": "", "last_close": 0,
            "change_pct": 0, "scores": [], "total": 0, "opinion": "",
            "error": f"데이터 조회 실패: {e}", "history": [], "disclosures": [],
            "news": [], "preliminary": None, "sources": {},
        }

    if df.empty:
        return {
            "code": code, "name": name, "last_date": "", "last_close": 0,
            "change_pct": 0, "scores": [], "total": 0, "opinion": "",
            "error": "가격 데이터가 비어 있습니다.", "history": [], "disclosures": [],
            "news": [], "preliminary": None, "sources": {},
        }

    last_close = float(df["Close"].iloc[-1])
    scores: list[ScoreItem] = [
        score_trend(df),
        score_momentum(df),
        score_volume(df),
        score_risk(df),
    ]

    flow_last_date: str | None = None
    fin_report_label: str | None = None

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
    if dart.is_configured():
        try:
            fin = dart.get_financials(code)
            fin_report_label = fin.get("report_label")
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
        except Exception as e:
            scores.append({
                "name": "재무",
                "score": 0,
                "msg": f"DART 조회 실패: {e}",
                "max": 0,
            })

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

    total = sum(s["score"] for s in scores)
    max_possible = sum(s["max"] for s in scores)

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
        "has_dart": dart.is_configured(),
        "has_llm": llm.is_configured() and not lite,
        "news_count": len(news_items),
        "lite_mode": lite,
        "preliminary_used_for_growth": preliminary_used_for_growth,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    opinion_text = overall_opinion(total, max_possible)

    # 매일 점수 스냅샷 저장 (같은 날 중복은 덮어씀)
    try:
        history.record_snapshot(code, total, last_close, opinion_text)
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
        "opinion": opinion_text,
        "error": None,
        "history": _build_history(df),
        "disclosures": disclosures,
        "news": news_items,
        "preliminary": preliminary,
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
    lines.append("━" * 40)
    return "\n".join(lines)


def main():
    codes = sys.argv[1:] if len(sys.argv) > 1 else ["005930"]
    for code in codes:
        print(render_text(analyze(code)))
        print()


if __name__ == "__main__":
    main()
