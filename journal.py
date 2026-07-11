"""매매기록·수익률 추적 — 한국+미국 주식, 환율 반영.

저장 위치: Gist(설정 시) + 로컬 data/trades.json
매매 1건 형식:
  {
    "id": "t1720680000123ab",
    "date": "2026-07-11",          # 체결일
    "market": "KR" | "US",
    "code": "005930" | "AAPL",
    "name": "삼성전자" | "AAPL",
    "side": "buy" | "sell",
    "qty": 10,
    "price": 61000.0,              # 매매 통화 기준 단가 (KR=원, US=달러)
    "fx": 1.0 | 1380.5,            # 체결 시점 원/달러 환율 (KR은 1.0)
    "fee": 0.0,                    # 수수료+세금 (매매 통화 기준)
    "note": ""
  }

수익률 계산:
  - 실현손익: 이동평균법(한국 증권사 표준). 원화 손익은 매수·매도 각각의
    체결 환율로 환산해 환차손익까지 포함한다.
  - 월간 수익률: Modified Dietz — 월말 평가액과 월중 매수(자금 투입)·매도(자금
    회수)를 기간 가중해 '주식에 들어가 있던 돈' 대비 수익률을 구한다.
    입출금 기록 없이 매매내역만으로 계산 가능한 방식.
  - 연간/누적 수익률: 월간 수익률을 기하 연결(chain-link).
"""
from __future__ import annotations

import random
import string
import time
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone

import pandas as pd

import cloud_store


TRADES_FILE = "trades.json"

KST = timezone(timedelta(hours=9))


def today_kst() -> date:
    return datetime.now(KST).date()


# ─────────── 매매내역 CRUD ───────────

def load_trades() -> list[dict]:
    data = cloud_store.load(TRADES_FILE, [])
    return data if isinstance(data, list) else []


def _new_id() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"t{int(time.time() * 1000)}{suffix}"


def normalize_trade(raw: dict) -> dict:
    """입력값 검증·형 보정. 잘못된 값이면 ValueError."""
    market = str(raw.get("market", "")).upper()
    if market not in ("KR", "US"):
        raise ValueError("market은 KR 또는 US")
    side = str(raw.get("side", "")).lower()
    if side not in ("buy", "sell"):
        raise ValueError("side는 buy 또는 sell")
    code = str(raw.get("code", "")).strip().upper() if market == "US" else str(raw.get("code", "")).strip()
    if not code:
        raise ValueError("종목코드가 비어 있음")
    try:
        d = str(raw.get("date", ""))
        datetime.strptime(d, "%Y-%m-%d")
    except ValueError:
        raise ValueError("date는 YYYY-MM-DD 형식")
    qty = float(raw.get("qty", 0))
    price = float(raw.get("price", 0))
    if qty <= 0 or price <= 0:
        raise ValueError("수량·단가는 0보다 커야 함")
    fx = 1.0 if market == "KR" else float(raw.get("fx", 0))
    if fx <= 0:
        raise ValueError("환율은 0보다 커야 함")
    fee = max(0.0, float(raw.get("fee", 0) or 0))
    return {
        "id": str(raw.get("id") or _new_id()),
        "date": d,
        "market": market,
        "code": code,
        "name": str(raw.get("name", "") or code).strip(),
        "side": side,
        "qty": qty,
        "price": price,
        "fx": fx,
        "fee": fee,
        "note": str(raw.get("note", "") or "").strip(),
    }


def add_trade(raw: dict) -> dict:
    """매매 1건 추가. 원격 최신본 확보(refresh) 후 read-modify-write."""
    trade = normalize_trade(raw)
    cloud_store.refresh()
    trades = load_trades()
    trades.append(trade)
    cloud_store.save(TRADES_FILE, trades)
    return trade


def delete_trades(ids: set[str]) -> int:
    cloud_store.refresh()
    trades = load_trades()
    kept = [t for t in trades if t.get("id") not in ids]
    removed = len(trades) - len(kept)
    if removed:
        cloud_store.save(TRADES_FILE, kept)
    return removed


def sorted_trades(trades: list[dict]) -> list[dict]:
    """날짜순 정렬. 같은 날짜는 입력 순서 유지(안정 정렬)."""
    return sorted(trades, key=lambda t: t.get("date", ""))


# ─────────── 액면분할(주식분할·병합) ───────────
# 시세 데이터(FinanceDataReader = 수정주가)는 분할을 소급 반영하므로, 분할 이전에
# 입력된 매매를 '현재 주수 기준'으로 환산해야 수량·단가가 시세와 맞는다.
# 환산은 qty×ratio, price÷ratio 라 금액(qty×price)이 보존됨 → 원금·손익 불변.

SPLITS_FILE = "splits.json"


def load_splits() -> list[dict]:
    data = cloud_store.load(SPLITS_FILE, [])
    return data if isinstance(data, list) else []


def normalize_split(raw: dict) -> dict:
    """분할 기록 검증. ratio = 1주가 몇 주가 되었나 (분할 3:1 → 3, 병합 1:10 → 0.1)."""
    market = str(raw.get("market", "")).upper()
    if market not in ("KR", "US"):
        raise ValueError("market은 KR 또는 US")
    code = str(raw.get("code", "")).strip().upper() if market == "US" else str(raw.get("code", "")).strip()
    if not code:
        raise ValueError("종목코드가 비어 있음")
    try:
        d = str(raw.get("date", ""))
        datetime.strptime(d, "%Y-%m-%d")
    except ValueError:
        raise ValueError("date는 YYYY-MM-DD 형식")
    ratio = float(raw.get("ratio", 0))
    if ratio <= 0:
        raise ValueError("비율은 0보다 커야 함")
    if abs(ratio - 1.0) < 1e-9:
        raise ValueError("비율 1은 분할이 아님")
    return {
        "id": str(raw.get("id") or _new_id()),
        "date": d,
        "market": market,
        "code": code,
        "name": str(raw.get("name", "") or code).strip(),
        "ratio": ratio,
    }


def add_split(raw: dict) -> dict:
    split = normalize_split(raw)
    cloud_store.refresh()
    splits = load_splits()
    for s in splits:
        if (s.get("market"), s.get("code"), s.get("date")) == (split["market"], split["code"], split["date"]):
            raise ValueError("같은 종목·날짜의 분할 기록이 이미 있음 (중복 반영 방지)")
    splits.append(split)
    cloud_store.save(SPLITS_FILE, splits)
    return split


def delete_splits(ids: set[str]) -> int:
    cloud_store.refresh()
    splits = load_splits()
    kept = [s for s in splits if s.get("id") not in ids]
    removed = len(splits) - len(kept)
    if removed:
        cloud_store.save(SPLITS_FILE, kept)
    return removed


def adjust_trades_for_splits(trades: list[dict], splits: list[dict]) -> list[dict]:
    """분할일 '이전' 체결을 현재 주수 기준으로 환산한 사본을 돌려준다.

    분할일 당일 이후 체결은 이미 새 기준으로 입력된 것으로 본다(증권사 앱 표기와 동일).
    여러 번 분할된 종목은 비율을 누적 적용.
    """
    if not splits:
        return list(trades)
    by_key: dict[tuple, list[dict]] = {}
    for s in splits:
        by_key.setdefault((s["market"], s["code"]), []).append(s)
    out = []
    for t in trades:
        factor = 1.0
        for s in by_key.get((t["market"], t["code"]), []):
            if t["date"] < s["date"]:
                factor *= s["ratio"]
        if abs(factor - 1.0) > 1e-9:
            t = {**t, "qty": t["qty"] * factor, "price": t["price"] / factor,
                 "split_factor": factor}
        out.append(t)
    return out


# ─────────── 포지션·실현손익 (이동평균법) ───────────

def compute_positions(trades: list[dict]) -> tuple[dict, list[dict], list[str]]:
    """매매내역 → (보유 포지션, 실현손익 이벤트, 경고).

    positions: {(market, code): {qty, avg_local, avg_krw, name, market, code}}
      - avg_local: 매매 통화 기준 평균단가 (수수료 포함)
      - avg_krw:  원화 기준 평균단가 (체결 환율 반영 → 환차손익 계산용)
    realized: 매도마다 {trade_id, date, market, code, name, qty,
                        pnl_local, pnl_krw, proceeds_krw, cost_krw}
    warnings: 보유량보다 많이 판 기록 등 데이터 이상.
    """
    positions: dict[tuple, dict] = {}
    realized: list[dict] = []
    warnings: list[str] = []

    for t in sorted_trades(trades):
        key = (t["market"], t["code"])
        pos = positions.get(key) or {
            "market": t["market"], "code": t["code"], "name": t["name"],
            "qty": 0.0, "avg_local": 0.0, "avg_krw": 0.0,
        }
        qty, price, fx, fee = t["qty"], t["price"], t["fx"], t["fee"]
        if t.get("name"):
            pos["name"] = t["name"]

        if t["side"] == "buy":
            cost_local = qty * price + fee
            cost_krw = (qty * price + fee) * fx
            new_qty = pos["qty"] + qty
            pos["avg_local"] = (pos["qty"] * pos["avg_local"] + cost_local) / new_qty
            pos["avg_krw"] = (pos["qty"] * pos["avg_krw"] + cost_krw) / new_qty
            pos["qty"] = new_qty
        else:
            sell_qty = qty
            if sell_qty > pos["qty"] + 1e-9:
                warnings.append(
                    f"{t['date']} {pos['name']}({t['code']}) 매도 {qty:g}주가 "
                    f"보유량 {pos['qty']:g}주보다 많음 — 초과분은 손익 계산에서 제외. "
                    f"이전 매수 기록을 먼저 입력하세요."
                )
                sell_qty = pos["qty"]
            if sell_qty > 0:
                pnl_local = sell_qty * (price - pos["avg_local"]) - fee
                proceeds_krw = (sell_qty * price - fee) * fx
                cost_krw = sell_qty * pos["avg_krw"]
                realized.append({
                    "trade_id": t["id"],
                    "date": t["date"],
                    "market": t["market"],
                    "code": t["code"],
                    "name": pos["name"],
                    "qty": sell_qty,
                    "pnl_local": pnl_local,
                    "pnl_krw": proceeds_krw - cost_krw,
                    "proceeds_krw": proceeds_krw,
                    "cost_krw": cost_krw,
                })
                pos["qty"] -= sell_qty
                if pos["qty"] <= 1e-9:
                    pos["qty"] = 0.0
                    pos["avg_local"] = 0.0
                    pos["avg_krw"] = 0.0
        positions[key] = pos

    open_positions = {k: v for k, v in positions.items() if v["qty"] > 0}
    return open_positions, realized, warnings


def realized_by_period(realized: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    """실현손익(원화)을 월별·연도별로 합산. → ({'2026-07': ...}, {'2026': ...})"""
    monthly: dict[str, float] = {}
    yearly: dict[str, float] = {}
    for r in realized:
        ym, y = r["date"][:7], r["date"][:4]
        monthly[ym] = monthly.get(ym, 0.0) + r["pnl_krw"]
        yearly[y] = yearly.get(y, 0.0) + r["pnl_krw"]
    return monthly, yearly


def realized_by_symbol(realized: list[dict]) -> list[dict]:
    """실현손익(원화)을 종목별로 합산. 손익 큰 순 정렬.

    반환: [{"market", "code", "name", "pnl_krw", "cost_krw", "sells", "ret"}]
      - ret: 매도분 원가 대비 실현수익률 (원가 0이면 None)
    """
    agg: dict[tuple, dict] = {}
    for r in realized:
        key = (r["market"], r["code"])
        a = agg.setdefault(key, {
            "market": r["market"], "code": r["code"], "name": r["name"],
            "pnl_krw": 0.0, "cost_krw": 0.0, "sells": 0,
        })
        a["pnl_krw"] += r["pnl_krw"]
        a["cost_krw"] += r["cost_krw"]
        a["sells"] += 1
        a["name"] = r["name"]
    out = []
    for a in agg.values():
        a["ret"] = (a["pnl_krw"] / a["cost_krw"]) if a["cost_krw"] > 0 else None
        out.append(a)
    return sorted(out, key=lambda x: x["pnl_krw"], reverse=True)


def realized_monthly_series(realized: list[dict], today: date | None = None) -> pd.Series:
    """첫 실현 달부터 이번 달까지, 빈 달을 0으로 채운 월별 실현손익(원) 시계열.

    인덱스는 월말 Timestamp — 그래프(막대·누적선)용.
    """
    monthly, _ = realized_by_period(realized)
    if not monthly:
        return pd.Series(dtype=float, name="실현손익(원)")
    today = today or today_kst()
    first = min(monthly)
    y, m = int(first[:4]), int(first[5:7])
    idx, vals = [], []
    while (y, m) <= (today.year, today.month):
        idx.append(pd.Timestamp(_month_end(y, m)))
        vals.append(monthly.get(f"{y:04d}-{m:02d}", 0.0))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return pd.Series(vals, index=idx, name="실현손익(원)")


# ─────────── 평가액·수익률 (Modified Dietz) ───────────

def _series_asof(series: pd.Series | None, d: date) -> float | None:
    """정렬된 날짜 인덱스 시리즈에서 d 이전(포함) 마지막 값. 없으면 None."""
    if series is None or len(series) == 0:
        return None
    try:
        v = series.asof(pd.Timestamp(d))
    except Exception:
        return None
    if pd.isna(v):
        # d 가 데이터 시작 전이면 첫 값으로 (상장 전 가격은 없으므로 근사)
        first = series.iloc[0]
        return float(first) if not pd.isna(first) else None
    return float(v)


def _last_trade_price(trades: list[dict], market: str, code: str, d: date) -> tuple[float, float] | None:
    """가격 데이터를 못 구할 때의 폴백 — d 이전 마지막 체결가·환율."""
    best = None
    for t in sorted_trades(trades):
        if t["market"] == market and t["code"] == code and t["date"] <= d.isoformat():
            best = (t["price"], t["fx"])
    return best


def holdings_at(trades: list[dict], d: date) -> dict[tuple, float]:
    """d 시점(포함) 보유 수량. 초과 매도는 0으로 클램프."""
    qty: dict[tuple, float] = {}
    for t in sorted_trades(trades):
        if t["date"] > d.isoformat():
            break
        key = (t["market"], t["code"])
        cur = qty.get(key, 0.0)
        qty[key] = cur + t["qty"] if t["side"] == "buy" else max(0.0, cur - t["qty"])
    return {k: v for k, v in qty.items() if v > 1e-9}


def equity_at(
    trades: list[dict],
    d: date,
    price_series_fn,
    fx_series: pd.Series | None,
) -> float:
    """d 시점 보유 주식 평가액(원). price_series_fn(market, code) -> pd.Series|None."""
    total = 0.0
    for (market, code), qty in holdings_at(trades, d).items():
        px = _series_asof(price_series_fn(market, code), d)
        fx = _series_asof(fx_series, d) if market == "US" else 1.0
        if px is None or fx is None:
            fallback = _last_trade_price(trades, market, code, d)
            if fallback is None:
                continue
            px = px if px is not None else fallback[0]
            fx = fx if fx is not None else fallback[1]
        total += qty * px * fx
    return total


def _month_end(y: int, m: int) -> date:
    return date(y, m, monthrange(y, m)[1])


def compute_monthly_returns(
    trades: list[dict],
    price_series_fn,
    fx_series: pd.Series | None,
    today: date | None = None,
) -> list[dict]:
    """첫 매매가 있는 달부터 이번 달까지 월별 수익률·손익.

    반환: [{"ym", "start_equity", "end_equity", "net_flow", "pnl_krw", "ret"}]
      - pnl_krw: 월간 손익(원) = 기말평가 − 기초평가 − 순투입
      - ret: Modified Dietz 수익률 (자본이 사실상 없던 달은 None)
    """
    ts = sorted_trades(trades)
    if not ts:
        return []
    today = today or today_kst()
    first = datetime.strptime(ts[0]["date"], "%Y-%m-%d").date()

    out: list[dict] = []
    y, m = first.year, first.month
    prev_equity = 0.0
    while (y, m) <= (today.year, today.month):
        m_start = date(y, m, 1)
        m_end = _month_end(y, m)
        is_current = (y, m) == (today.year, today.month)
        eval_end = today if is_current else m_end
        period_days = max(1, (eval_end - m_start).days + 1)

        # 월중 자금 흐름: 매수 = +투입, 매도 = −회수 (원화, 체결 환율 기준)
        net_flow = 0.0
        weighted_flow = 0.0
        for t in ts:
            if not (m_start.isoformat() <= t["date"] <= eval_end.isoformat()):
                continue
            if t["side"] == "buy":
                f = (t["qty"] * t["price"] + t["fee"]) * t["fx"]
            else:
                f = -(t["qty"] * t["price"] - t["fee"]) * t["fx"]
            t_day = datetime.strptime(t["date"], "%Y-%m-%d").date()
            elapsed = (t_day - m_start).days
            w = (period_days - elapsed) / period_days
            net_flow += f
            weighted_flow += w * f

        end_equity = equity_at(ts, eval_end, price_series_fn, fx_series)
        pnl = end_equity - prev_equity - net_flow
        denom = prev_equity + weighted_flow
        ret = (pnl / denom) if denom > 1.0 else None

        out.append({
            "ym": f"{y:04d}-{m:02d}",
            "start_equity": prev_equity,
            "end_equity": end_equity,
            "net_flow": net_flow,
            "pnl_krw": pnl,
            "ret": ret,
        })
        prev_equity = end_equity
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def yearly_returns(monthly: list[dict]) -> list[dict]:
    """월간 수익률을 연도별 기하 연결. [{"year", "pnl_krw", "ret"}]"""
    by_year: dict[str, dict] = {}
    for row in monthly:
        year = row["ym"][:4]
        agg = by_year.setdefault(year, {"year": year, "pnl_krw": 0.0, "growth": 1.0, "has_ret": False})
        agg["pnl_krw"] += row["pnl_krw"]
        if row["ret"] is not None:
            agg["growth"] *= 1.0 + row["ret"]
            agg["has_ret"] = True
    return [
        {"year": v["year"], "pnl_krw": v["pnl_krw"],
         "ret": (v["growth"] - 1.0) if v["has_ret"] else None}
        for v in sorted(by_year.values(), key=lambda x: x["year"])
    ]


def cumulative_curve(monthly: list[dict]) -> pd.Series:
    """월간 수익률 기하 연결 → 누적 수익률(%) 시계열 (인덱스: 월말 Timestamp)."""
    growth = 1.0
    idx, vals = [], []
    for row in monthly:
        if row["ret"] is not None:
            growth *= 1.0 + row["ret"]
        y, m = int(row["ym"][:4]), int(row["ym"][5:7])
        idx.append(pd.Timestamp(_month_end(y, m)))
        vals.append((growth - 1.0) * 100.0)
    return pd.Series(vals, index=idx, name="누적 수익률(%)")
