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


def add_holding(code: str, quantity: int, avg_price: float) -> None:
    p = load_portfolio()
    p[code] = {
        "quantity": int(quantity),
        "avg_price": float(avg_price),
        "added_at": datetime.now().strftime("%Y-%m-%d"),
    }
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
