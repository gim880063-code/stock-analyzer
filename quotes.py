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


def usdkrw_history(start: date) -> pd.Series | None:
    """start 이후 원/달러 일별 환율 시리즈 (서울 날짜 기준). 실패 시 None.

    글로벌 FX 일봉은 뉴욕 기준이라 일~목요일에 라벨이 찍히고, 라벨 D 봉이
    실제로는 서울 D+1 거래일의 환율이다. 하루 밀어 서울 날짜(월~금)에 맞춘다.
    검증: 서울 2023-06-29 고시 ≈1,310원 = 원본 6/28 봉 1,309.01 /
          서울 2023-07-14 ≈1,268원 = 원본 7/13 봉 1,266.83.
    """
    def fetch():
        # 하루 시프트 후에도 start 시점 값이 있도록 며칠 앞서 조회
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
