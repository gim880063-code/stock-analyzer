"""
포트폴리오 추적 — 보유 종목·수량·평균매수가 관리.

저장 위치: Gist (설정 시) + 로컬 data/portfolio.json
형식:
  {
    "005930": {"quantity": 100, "avg_price": 250000, "added_at": "2026-05-10"},
    ...
  }
"""
from datetime import datetime

import cloud_store


FILENAME = "portfolio.json"


def load_portfolio() -> dict[str, dict]:
    data = cloud_store.load(FILENAME, {})
    return data if isinstance(data, dict) else {}


def save_portfolio(portfolio: dict[str, dict]) -> None:
    cloud_store.save(FILENAME, portfolio)


def add_holding(
    code: str,
    quantity: int,
    avg_price: float,
    stop_loss: float | None = None,
    target_1r: float | None = None,
    target_2r: float | None = None,
) -> None:
    p = load_portfolio()
    entry = {
        "quantity": int(quantity),
        "avg_price": float(avg_price),
        "added_at": datetime.now().strftime("%Y-%m-%d"),
        "peak_price": float(avg_price),   # 트레일링 스톱용 고점 추적
    }
    # 매수 시점의 손절/목표가를 저장해 두면 보유 점검 때 그대로 기준이 됨.
    if stop_loss is not None:
        entry["stop_loss"] = float(stop_loss)
    if target_1r is not None:
        entry["target_1r"] = float(target_1r)
    if target_2r is not None:
        entry["target_2r"] = float(target_2r)
    p[code] = entry
    save_portfolio(p)


def update_plan(
    code: str,
    stop_loss: float | None = None,
    target_1r: float | None = None,
    target_2r: float | None = None,
) -> None:
    """기존 보유 종목에 손절/목표가만 갱신 (수량·평단 유지)."""
    p = load_portfolio()
    if code not in p:
        return
    if stop_loss is not None:
        p[code]["stop_loss"] = float(stop_loss)
    if target_1r is not None:
        p[code]["target_1r"] = float(target_1r)
    if target_2r is not None:
        p[code]["target_2r"] = float(target_2r)
    save_portfolio(p)


def update_holding(code: str, quantity: int, avg_price: float) -> None:
    p = load_portfolio()
    if code in p:
        p[code]["quantity"] = int(quantity)
        p[code]["avg_price"] = float(avg_price)
        save_portfolio(p)


def remove_holding(code: str) -> None:
    p = load_portfolio()
    if code in p:
        del p[code]
        save_portfolio(p)


def compute_pnl(holding: dict, current_price: float) -> dict:
    """현재가 기준 손익 계산."""
    qty = holding.get("quantity", 0)
    avg = holding.get("avg_price", 0)
    cost = qty * avg
    market = qty * current_price
    profit = market - cost
    pct = (current_price / avg - 1) * 100 if avg > 0 else 0
    return {
        "cost": cost,
        "market_value": market,
        "profit": profit,
        "profit_pct": pct,
    }


# ─────────── 리스크/사이징 설정 ───────────
# 포지션 사이징·트레일링 스톱에 쓰는 사용자 설정. Gist + 로컬 동기화.
SETTINGS_FILE = "risk_settings.json"
DEFAULT_SETTINGS = {
    "account_equity": 0,          # 총 투자금(원). 0이면 사이징 비활성.
    "risk_per_trade_pct": 1.0,    # 한 종목 매매에서 감수할 계좌 대비 손실 비율(%)
    "max_position_pct": 20.0,     # 한 종목 최대 비중(%)
    "trail_pct": 10.0,            # 트레일링 스톱: 고점 대비 하락 임계(%)
    "round_trip_cost_pct": 0.5,   # 왕복 거래비용(수수료+매도세+슬리피지) — 검증 net 수익용
    "risk_off_enabled": True,     # 하락장(코스피<200일선)에 신규 진입 기준 상향
    "risk_off_score_boost": 2,    # 리스크오프 시 min_score 에 더할 점수
}


def load_settings() -> dict:
    s = cloud_store.load(SETTINGS_FILE, {})
    if not isinstance(s, dict):
        s = {}
    merged = {**DEFAULT_SETTINGS, **s}
    # 숫자 형 보정 (저장 중 문자열로 들어와도 안전)
    out = {}
    for k, default in DEFAULT_SETTINGS.items():
        try:
            out[k] = type(default)(merged.get(k, default))
        except (TypeError, ValueError):
            out[k] = default
    return out


def save_settings(settings: dict) -> None:
    clean = {}
    for k, default in DEFAULT_SETTINGS.items():
        try:
            clean[k] = type(default)(settings.get(k, default))
        except (TypeError, ValueError):
            clean[k] = default
    cloud_store.save(SETTINGS_FILE, clean)
