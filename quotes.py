"""시세·환율 조회 — FinanceDataReader 기반 (무료, 키 불필요).

매매기록 페이지에서 쓰는 얇은 래퍼:
  - 한국 주식: 6자리 코드 그대로 (예: 005930)
  - 미국 주식: 티커 그대로 (예: AAPL) — fdr 이 자동 판별
  - 환율: 'USD/KRW'

가격은 일별 종가라 장중 실시간이 아니다(최근 종가 기준). 모듈 내 TTL 캐시로
rerun 마다 재요청하지 않게 한다. 실패도 짧게 캐시해 죽은 티커가 화면을
느리게 만드는 걸 막는다.
"""
from __future__ import annotations

import threading
import time
from datetime import date, timedelta

import pandas as pd
import FinanceDataReader as fdr


_TTL_OK = 600.0     # 정상 응답 캐시(초)
_TTL_FAIL = 120.0   # 실패 캐시(초) — 이 동안은 재시도 안 함
_cache: dict[tuple, tuple[float, object]] = {}
_lock = threading.Lock()


def _cached_fetch(key: tuple, fetch):
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            saved_at, val = hit
            ttl = _TTL_OK if val is not None else _TTL_FAIL
            if now - saved_at < ttl:
                return val
    try:
        val = fetch()
        if val is not None and len(val) == 0:
            val = None
    except Exception:
        val = None
    with _lock:
        _cache[key] = (now, val)
    return val


def price_history(market: str, code: str, start: date) -> pd.Series | None:
    """start 이후 일별 종가 시리즈. 실패 시 None."""
    def fetch():
        df = fdr.DataReader(code, start.isoformat())
        return df["Close"].dropna()
    return _cached_fetch(("px", market, code, start.isoformat()), fetch)


_NAVER_FX_URL = "https://m.stock.naver.com/front-api/marketIndex/prices"
_NAVER_FX_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _naver_usdkrw_history(start: date) -> pd.Series | None:
    """네이버 서울 고시환율(하나은행 매매기준율) 일별 이력. 실패 시 None.

    글로벌 FX 소스와 달리 한국 휴장일이 정확히 반영된다 — 2023-10-04(추석 연휴
    직후) 삼성증권 적용 1,351.3원 vs 서울 기준율 1,354.0(2.7원 차) vs 글로벌
    1,359.2(7.9원 차) 사례에서 교체 결정. 증권사 적용 환율(최초고시±스프레드)과
    같은 계열이라 사용자 거래내역과 가장 가깝다.
    """
    import requests

    rows: dict[str, float] = {}
    start_key = start.isoformat()
    try:
        for page in range(1, 201):  # 60건/페이지 — 200페이지 ≈ 48년치 상한
            r = requests.get(
                _NAVER_FX_URL,
                params={"category": "exchange", "reutersCode": "FX_USDKRW",
                        "page": page, "pageSize": 60},
                headers=_NAVER_FX_HEADERS, timeout=15,
            )
            items = (r.json().get("result") or []) if r.ok else []
            if not items:
                break
            for it in items:
                rows[it["localTradedAt"]] = float(str(it["closePrice"]).replace(",", ""))
            if items[-1]["localTradedAt"] < start_key:
                break
        rows = {d: v for d, v in rows.items() if d >= start_key}
        if not rows:
            return None
        s = pd.Series(rows)
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception:
        return None


def usdkrw_history(start: date) -> pd.Series | None:
    """start 이후 원/달러 일별 환율 시리즈 (서울 고시환율 기준). 실패 시 None.

    1순위: 네이버 서울 매매기준율 (한국 휴장일 정확, 증권사 기준과 동일 계열).
    폴백: 글로벌 FX 일봉(fdr) — 뉴욕 기준이라 일~목 라벨이 서울 D+1 환율이라
    하루 밀어 서울 날짜에 맞춘다 (연휴 구간엔 수 원 오차 가능).
    """
    def fetch():
        s = _naver_usdkrw_history(start - timedelta(days=4))
        if s is not None and len(s) > 0:
            return s
        df = fdr.DataReader("USD/KRW", (start - timedelta(days=4)).isoformat())
        s = df["Close"].dropna()
        s.index = s.index + pd.Timedelta(days=1)
        return s
    return _cached_fetch(("fx", start.isoformat()), fetch)


def current_usdkrw() -> float | None:
    """가장 최근 원/달러 환율."""
    s = usdkrw_history(date.today() - timedelta(days=10))
    if s is None or len(s) == 0:
        return None
    return float(s.iloc[-1])


def usdkrw_at(d: date) -> float | None:
    """d 당일(휴장이면 직전 영업일) 원/달러 환율 — 과거 매매 입력 시 자동 채움용."""
    s = usdkrw_history(d - timedelta(days=10))
    if s is None:
        return None
    try:
        v = s.asof(pd.Timestamp(d))
    except Exception:
        return None
    return None if pd.isna(v) else float(v)


def current_price(market: str, code: str) -> float | None:
    """가장 최근 종가."""
    s = price_history(market, code, date.today() - timedelta(days=14))
    if s is None or len(s) == 0:
        return None
    return float(s.iloc[-1])
