"""journal.py 회귀 테스트 — 이동평균법 실현손익, 환차손익, Modified Dietz 수익률."""
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import journal  # noqa: E402


def _t(d, market, code, side, qty, price, fx=1.0, fee=0.0, name=""):
    return {
        "id": f"id-{d}-{code}-{side}-{qty}",
        "date": d, "market": market, "code": code, "name": name or code,
        "side": side, "qty": float(qty), "price": float(price),
        "fx": float(fx), "fee": float(fee), "note": "",
    }


class FakeStore:
    def __init__(self):
        self.data = {}

    def load(self, filename, default):
        return self.data.get(filename, default)

    def save(self, filename, data):
        self.data[filename] = data

    def refresh(self):
        return True


class NormalizeTests(unittest.TestCase):
    def test_valid_kr(self):
        t = journal.normalize_trade({
            "date": "2026-07-01", "market": "kr", "code": "005930",
            "side": "BUY", "qty": 10, "price": 61000,
        })
        self.assertEqual(t["market"], "KR")
        self.assertEqual(t["side"], "buy")
        self.assertEqual(t["fx"], 1.0)  # 한국 주식은 환율 강제 1.0
        self.assertTrue(t["id"])

    def test_us_ticker_uppercased_and_fx_required(self):
        t = journal.normalize_trade({
            "date": "2026-07-01", "market": "US", "code": "aapl",
            "side": "buy", "qty": 1, "price": 300, "fx": 1400,
        })
        self.assertEqual(t["code"], "AAPL")
        with self.assertRaises(ValueError):
            journal.normalize_trade({
                "date": "2026-07-01", "market": "US", "code": "AAPL",
                "side": "buy", "qty": 1, "price": 300, "fx": 0,
            })

    def test_invalid_inputs(self):
        base = {"date": "2026-07-01", "market": "KR", "code": "005930",
                "side": "buy", "qty": 10, "price": 1000}
        for bad in ({"qty": 0}, {"price": -1}, {"date": "07/01"},
                    {"side": "hold"}, {"market": "JP"}, {"code": ""}):
            with self.assertRaises(ValueError):
                journal.normalize_trade({**base, **bad})


class PositionTests(unittest.TestCase):
    def test_average_cost_with_fee(self):
        trades = [
            _t("2026-01-05", "KR", "005930", "buy", 10, 10000, fee=100),
            _t("2026-02-05", "KR", "005930", "buy", 10, 12000),
            _t("2026-03-05", "KR", "005930", "sell", 5, 13000, fee=50),
        ]
        pos, realized, warnings = journal.compute_positions(trades)
        p = pos[("KR", "005930")]
        # 평단 = (10*10000+100 + 10*12000) / 20 = 11005
        self.assertAlmostEqual(p["avg_local"], 11005.0)
        self.assertAlmostEqual(p["qty"], 15.0)
        self.assertEqual(len(realized), 1)
        # 실현 = 5*(13000-11005) - 50 = 9925
        self.assertAlmostEqual(realized[0]["pnl_local"], 9925.0)
        self.assertAlmostEqual(realized[0]["pnl_krw"], 9925.0)
        self.assertEqual(warnings, [])

    def test_us_fx_gain_included_in_krw_pnl(self):
        trades = [
            _t("2026-01-05", "US", "AAPL", "buy", 10, 100, fx=1300),
            _t("2026-02-05", "US", "AAPL", "sell", 5, 110, fx=1400),
        ]
        _, realized, _ = journal.compute_positions(trades)
        r = realized[0]
        self.assertAlmostEqual(r["pnl_local"], 50.0)          # 달러 기준 5*(110-100)
        # 원화: 5*110*1400 - 5*100*1300 = 770000-650000 = 주가차익+환차익
        self.assertAlmostEqual(r["pnl_krw"], 120000.0)

    def test_oversell_clamped_with_warning(self):
        trades = [
            _t("2026-01-05", "KR", "005930", "buy", 10, 10000),
            _t("2026-02-05", "KR", "005930", "sell", 20, 11000),
        ]
        pos, realized, warnings = journal.compute_positions(trades)
        self.assertNotIn(("KR", "005930"), pos)  # 전량 매도 처리
        self.assertAlmostEqual(realized[0]["qty"], 10.0)
        self.assertAlmostEqual(realized[0]["pnl_local"], 10000.0)
        self.assertEqual(len(warnings), 1)

    def test_full_sell_resets_average(self):
        trades = [
            _t("2026-01-05", "KR", "005930", "buy", 10, 10000),
            _t("2026-02-05", "KR", "005930", "sell", 10, 11000),
            _t("2026-03-05", "KR", "005930", "buy", 5, 20000),
        ]
        pos, _, _ = journal.compute_positions(trades)
        self.assertAlmostEqual(pos[("KR", "005930")]["avg_local"], 20000.0)

    def test_realized_by_symbol_aggregates_and_sorts(self):
        trades = [
            _t("2026-01-05", "KR", "005930", "buy", 10, 10000, name="삼성전자"),
            _t("2026-02-05", "KR", "005930", "sell", 5, 12000, name="삼성전자"),
            _t("2026-02-06", "KR", "005930", "sell", 5, 13000, name="삼성전자"),
            _t("2026-01-05", "US", "AAPL", "buy", 10, 100, fx=1300),
            _t("2026-03-05", "US", "AAPL", "sell", 10, 90, fx=1300),
        ]
        _, realized, _ = journal.compute_positions(trades)
        by_sym = journal.realized_by_symbol(realized)
        self.assertEqual([a["code"] for a in by_sym], ["005930", "AAPL"])  # 이익 큰 순
        samsung = by_sym[0]
        self.assertEqual(samsung["sells"], 2)
        self.assertAlmostEqual(samsung["pnl_krw"], 5 * 2000 + 5 * 3000)
        self.assertAlmostEqual(samsung["ret"], 25000 / 100000)  # 매도분 원가 대비
        aapl = by_sym[1]
        self.assertAlmostEqual(aapl["pnl_krw"], 10 * (90 - 100) * 1300)
        self.assertAlmostEqual(aapl["ret"], -0.1)

    def test_realized_monthly_series_fills_gaps(self):
        realized = [
            {"date": "2026-01-10", "pnl_krw": 100.0},
            {"date": "2026-04-10", "pnl_krw": -30.0},
        ]
        s = journal.realized_monthly_series(realized, today=date(2026, 5, 15))
        self.assertEqual(len(s), 5)  # 1~5월, 빈 달 0
        self.assertAlmostEqual(s.iloc[0], 100.0)
        self.assertAlmostEqual(s.iloc[1], 0.0)
        self.assertAlmostEqual(s.iloc[3], -30.0)
        self.assertAlmostEqual(s.cumsum().iloc[-1], 70.0)

    def test_realized_by_period(self):
        realized = [
            {"date": "2025-12-10", "pnl_krw": 100.0},
            {"date": "2026-01-10", "pnl_krw": 200.0},
            {"date": "2026-01-20", "pnl_krw": -50.0},
        ]
        monthly, yearly = journal.realized_by_period(realized)
        self.assertAlmostEqual(monthly["2026-01"], 150.0)
        self.assertAlmostEqual(yearly["2025"], 100.0)
        self.assertAlmostEqual(yearly["2026"], 150.0)


def _split(d, market, code, ratio):
    return {"id": f"s-{d}-{code}", "date": d, "market": market, "code": code,
            "name": code, "ratio": float(ratio)}


class SplitTests(unittest.TestCase):
    def test_adjusts_only_trades_before_split(self):
        trades = [
            _t("2022-01-10", "US", "TSLA", "buy", 10, 900),   # 분할 전
            _t("2022-08-25", "US", "TSLA", "buy", 3, 300),    # 분할 당일 (새 기준)
            _t("2023-01-10", "US", "TSLA", "buy", 5, 200),    # 분할 후
        ]
        adj = journal.adjust_trades_for_splits(trades, [_split("2022-08-25", "US", "TSLA", 3)])
        self.assertAlmostEqual(adj[0]["qty"], 30.0)
        self.assertAlmostEqual(adj[0]["price"], 300.0)
        self.assertAlmostEqual(adj[1]["qty"], 3.0)     # 당일 이후는 그대로
        self.assertAlmostEqual(adj[2]["price"], 200.0)
        # 금액 보존: 환산 전후 qty×price 동일
        self.assertAlmostEqual(adj[0]["qty"] * adj[0]["price"], 9000.0)

    def test_multiple_splits_compound(self):
        trades = [_t("2019-01-10", "US", "TSLA", "buy", 1, 1500)]
        splits = [
            _split("2020-08-31", "US", "TSLA", 5),
            _split("2022-08-25", "US", "TSLA", 3),
        ]
        adj = journal.adjust_trades_for_splits(trades, splits)
        self.assertAlmostEqual(adj[0]["qty"], 15.0)
        self.assertAlmostEqual(adj[0]["price"], 100.0)

    def test_merge_ratio_below_one(self):
        trades = [_t("2024-01-10", "KR", "000001", "buy", 100, 1000)]
        adj = journal.adjust_trades_for_splits(trades, [_split("2024-06-01", "KR", "000001", 0.1)])
        self.assertAlmostEqual(adj[0]["qty"], 10.0)
        self.assertAlmostEqual(adj[0]["price"], 10000.0)

    def test_positions_and_realized_after_split(self):
        # 분할 전 10주@900 매수 → 3:1 분할 → 30주@300 기준으로 15주@350 매도
        trades = [
            _t("2022-01-10", "US", "TSLA", "buy", 10, 900, fx=1300),
            _t("2022-09-10", "US", "TSLA", "sell", 15, 350, fx=1300),
        ]
        adj = journal.adjust_trades_for_splits(trades, [_split("2022-08-25", "US", "TSLA", 3)])
        pos, realized, warnings = journal.compute_positions(adj)
        self.assertEqual(warnings, [])  # 분할 환산 덕에 '보유량 초과 매도' 아님
        self.assertAlmostEqual(pos[("US", "TSLA")]["qty"], 15.0)
        self.assertAlmostEqual(pos[("US", "TSLA")]["avg_local"], 300.0)
        self.assertAlmostEqual(realized[0]["pnl_local"], 15 * (350 - 300))

    def test_equity_money_preserved_across_split(self):
        # 수정주가 시세(현재 기준)와 환산 수량이 일치해 분할 전 시점 평가액도 정확
        trades = [_t("2026-01-05", "KR", "005930", "buy", 10, 900)]
        adj = journal.adjust_trades_for_splits(trades, [_split("2026-02-15", "KR", "005930", 3)])
        px = pd.Series([300.0, 330.0], index=pd.to_datetime(["2026-01-05", "2026-02-28"]))
        monthly = journal.compute_monthly_returns(adj, lambda m, c: px, None, today=date(2026, 2, 28))
        self.assertAlmostEqual(monthly[0]["end_equity"], 9000.0)   # 1월말: 30주×300 = 투자금 그대로
        self.assertAlmostEqual(monthly[1]["ret"], 0.10, places=6)  # 2월: +10%

    def test_split_crud_and_duplicate_guard(self):
        fake = FakeStore()
        with patch.object(journal, "cloud_store", fake):
            s = journal.add_split({"market": "US", "code": "tsla", "date": "2022-08-25", "ratio": 3})
            self.assertEqual(s["code"], "TSLA")
            with self.assertRaises(ValueError):
                journal.add_split({"market": "US", "code": "TSLA", "date": "2022-08-25", "ratio": 3})
            with self.assertRaises(ValueError):
                journal.normalize_split({"market": "US", "code": "TSLA", "date": "2022-08-25", "ratio": 1})
            self.assertEqual(journal.delete_splits({s["id"]}), 1)
            self.assertEqual(journal.load_splits(), [])


class IncomeTests(unittest.TestCase):
    def test_dividend_net_krw_with_tax_and_fx(self):
        e = journal.normalize_income({
            "type": "dividend", "date": "2026-06-10", "market": "US",
            "code": "aapl", "amount": 100, "tax": 15, "fx": 1400,
        })
        self.assertEqual(e["code"], "AAPL")
        self.assertEqual(e["currency"], "USD")
        self.assertAlmostEqual(journal.income_net_krw(e), (100 - 15) * 1400)

    def test_expense_and_income_signs(self):
        tax = journal.normalize_income({
            "type": "expense", "date": "2026-05-31", "name": "해외주식 양도소득세",
            "amount": 220000, "currency": "KRW",
        })
        self.assertAlmostEqual(journal.income_net_krw(tax), -220000.0)
        etc = journal.normalize_income({
            "type": "income", "date": "2026-05-31", "name": "이자",
            "amount": 10, "currency": "USD", "fx": 1400,
        })
        self.assertAlmostEqual(journal.income_net_krw(etc), 14000.0)

    def test_income_validation(self):
        base = {"type": "dividend", "date": "2026-06-10", "market": "US",
                "code": "AAPL", "amount": 100, "tax": 15, "fx": 1400}
        for bad in ({"type": "loan"}, {"amount": 0}, {"tax": 100},
                    {"code": ""}, {"fx": 0}, {"date": "6/10"}):
            with self.assertRaises(ValueError):
                journal.normalize_income({**base, **bad})
        with self.assertRaises(ValueError):
            journal.normalize_income({"type": "expense", "date": "2026-06-10",
                                      "amount": 100, "currency": "EUR"})

    def test_incomes_by_period_and_merge(self):
        incomes = [
            journal.normalize_income({"type": "dividend", "date": "2026-06-10",
                                      "market": "KR", "code": "005930",
                                      "amount": 100000, "tax": 15400}),
            journal.normalize_income({"type": "expense", "date": "2026-06-20",
                                      "name": "출금 수수료", "amount": 5000, "currency": "KRW"}),
        ]
        monthly, yearly = journal.incomes_by_period(incomes)
        self.assertAlmostEqual(monthly["2026-06"], 84600 - 5000)
        self.assertAlmostEqual(yearly["2026"], 79600)
        merged = journal.merge_period_sums({"2026-06": 100.0}, monthly)
        self.assertAlmostEqual(merged["2026-06"], 79700.0)

    def test_dividends_by_symbol(self):
        incomes = [
            journal.normalize_income({"type": "dividend", "date": "2026-03-10",
                                      "market": "US", "code": "AAPL",
                                      "amount": 100, "tax": 15, "fx": 1400}),
            journal.normalize_income({"type": "dividend", "date": "2026-06-10",
                                      "market": "US", "code": "AAPL",
                                      "amount": 100, "tax": 15, "fx": 1500}),
            journal.normalize_income({"type": "expense", "date": "2026-06-20",
                                      "name": "세금", "amount": 1, "currency": "KRW"}),
        ]
        by_sym = journal.dividends_by_symbol(incomes)
        a = by_sym[("US", "AAPL")]
        self.assertEqual(a["count"], 2)
        self.assertAlmostEqual(a["net_krw"], 85 * 1400 + 85 * 1500)

    def test_dividend_raises_monthly_return(self):
        # 1/2 매수 10주@10,000, 가격 변동 없음, 1/15 배당 5,000원(세후) → 수익률 > 0
        trades = [_t("2026-01-02", "KR", "005930", "buy", 10, 10000)]
        px = pd.Series([10000.0, 10000.0], index=pd.to_datetime(["2026-01-02", "2026-01-31"]))
        div = journal.normalize_income({"type": "dividend", "date": "2026-01-15",
                                        "market": "KR", "code": "005930",
                                        "amount": 5000, "tax": 0})
        monthly = journal.compute_monthly_returns(
            trades, lambda m, c: px, None, today=date(2026, 1, 31), incomes=[div])
        jan = monthly[0]
        self.assertAlmostEqual(jan["pnl_krw"], 5000.0)  # 배당만큼 이익
        # denom = 100000×(30/31) − 5000×(17/31)
        expected = 5000.0 / (100000 * 30 / 31 - 5000 * 17 / 31)
        self.assertAlmostEqual(jan["ret"], expected, places=6)
        # 세금·비용은 수익률에 영향 없음
        exp = journal.normalize_income({"type": "expense", "date": "2026-01-20",
                                        "name": "수수료", "amount": 99999, "currency": "KRW"})
        monthly2 = journal.compute_monthly_returns(
            trades, lambda m, c: px, None, today=date(2026, 1, 31), incomes=[div, exp])
        self.assertAlmostEqual(monthly2[0]["ret"], expected, places=6)

    def test_expense_with_stock_link(self):
        e = journal.normalize_income({
            "type": "expense", "date": "2026-05-31", "name": "해외주식 양도소득세",
            "amount": 300000, "currency": "KRW",
            "market": "us", "code": "tsla", "stock_name": "TSLA",
        })
        self.assertEqual((e["market"], e["code"]), ("US", "TSLA"))
        with self.assertRaises(ValueError):  # 시장만 있고 종목코드 없음
            journal.normalize_income({
                "type": "expense", "date": "2026-05-31", "amount": 1,
                "currency": "KRW", "market": "US", "code": "",
            })
        # 종목 연결 없는 세금도 여전히 허용
        e2 = journal.normalize_income({
            "type": "expense", "date": "2026-05-31", "name": "출금 수수료",
            "amount": 5000, "currency": "KRW",
        })
        self.assertEqual((e2["market"], e2["code"]), ("", ""))

    def test_taxes_by_year_and_symbol(self):
        incomes = [
            # TSLA 배당 $100, 세금 $15, 환율 1400 → 원천징수 21,000원
            journal.normalize_income({"type": "dividend", "date": "2026-03-10",
                                      "market": "US", "code": "TSLA",
                                      "amount": 100, "tax": 15, "fx": 1400}),
            # TSLA 연결 양도세 300,000원
            journal.normalize_income({"type": "expense", "date": "2026-05-31",
                                      "name": "양도소득세", "amount": 300000,
                                      "currency": "KRW", "market": "US",
                                      "code": "TSLA", "stock_name": "TSLA"}),
            # 계좌 공통 수수료 5,000원
            journal.normalize_income({"type": "expense", "date": "2026-06-01",
                                      "name": "출금 수수료", "amount": 5000, "currency": "KRW"}),
            # 작년 세금 (연도 분리 확인)
            journal.normalize_income({"type": "expense", "date": "2025-05-31",
                                      "name": "양도소득세", "amount": 100000, "currency": "KRW"}),
        ]
        by_year = journal.taxes_by_year(incomes)
        self.assertAlmostEqual(by_year["2026"]["expense_krw"], 305000.0)
        self.assertAlmostEqual(by_year["2026"]["dividend_tax_krw"], 21000.0)
        self.assertAlmostEqual(by_year["2025"]["expense_krw"], 100000.0)

        by_sym = journal.taxes_by_symbol(incomes)
        tsla = next(a for a in by_sym if a["code"] == "TSLA")
        self.assertAlmostEqual(tsla["expense_krw"], 300000.0)
        self.assertAlmostEqual(tsla["dividend_tax_krw"], 21000.0)
        self.assertAlmostEqual(tsla["total_krw"], 321000.0)
        common = next(a for a in by_sym if a["code"] == "")
        self.assertEqual(common["name"], "계좌 공통")
        self.assertAlmostEqual(common["total_krw"], 105000.0)
        # 정렬: 합계 큰 순
        self.assertEqual(by_sym[0]["code"], "TSLA")

    def test_income_crud(self):
        fake = FakeStore()
        with patch.object(journal, "cloud_store", fake):
            e = journal.add_income({"type": "dividend", "date": "2026-06-10",
                                    "market": "KR", "code": "005930", "name": "삼성전자",
                                    "amount": 10000, "tax": 1540})
            self.assertEqual(len(journal.load_incomes()), 1)
            self.assertEqual(journal.delete_incomes({e["id"]}), 1)
            self.assertEqual(journal.load_incomes(), [])


def _kr_series():
    return pd.Series(
        [10000.0, 11000.0, 12100.0],
        index=pd.to_datetime(["2026-01-02", "2026-01-31", "2026-02-28"]),
    )


class ReturnTests(unittest.TestCase):
    def test_monthly_dietz_and_yearly_chain(self):
        trades = [_t("2026-01-02", "KR", "005930", "buy", 10, 10000)]
        price_fn = lambda market, code: _kr_series()  # noqa: E731
        monthly = journal.compute_monthly_returns(
            trades, price_fn, None, today=date(2026, 2, 28))
        self.assertEqual([m["ym"] for m in monthly], ["2026-01", "2026-02"])

        jan, feb = monthly
        # 1월: 기말 110,000 / 1/2 매수 100,000 (가중 30/31) → 10,000 / 96,774.19
        self.assertAlmostEqual(jan["pnl_krw"], 10000.0)
        self.assertAlmostEqual(jan["ret"], 10000.0 / (100000.0 * 30 / 31), places=6)
        # 2월: 흐름 없음 → 순수 가격 변동 10%
        self.assertAlmostEqual(feb["pnl_krw"], 11000.0)
        self.assertAlmostEqual(feb["ret"], 0.10, places=6)

        yearly = journal.yearly_returns(monthly)
        self.assertEqual(yearly[0]["year"], "2026")
        expected = (1 + jan["ret"]) * (1 + feb["ret"]) - 1
        self.assertAlmostEqual(yearly[0]["ret"], expected, places=6)

        curve = journal.cumulative_curve(monthly)
        self.assertAlmostEqual(curve.iloc[-1], expected * 100, places=4)

    def test_us_equity_uses_fx_series(self):
        trades = [_t("2026-01-02", "US", "AAPL", "buy", 1, 100, fx=1300)]
        px = pd.Series([100.0, 100.0], index=pd.to_datetime(["2026-01-02", "2026-01-31"]))
        fx = pd.Series([1300.0, 1430.0], index=pd.to_datetime(["2026-01-02", "2026-01-31"]))
        monthly = journal.compute_monthly_returns(
            trades, lambda m, c: px, fx, today=date(2026, 1, 31))
        # 주가 그대로 + 환율 10% 상승 → 원화 평가 143,000, 투입 130,000
        self.assertAlmostEqual(monthly[0]["end_equity"], 143000.0)
        self.assertAlmostEqual(monthly[0]["pnl_krw"], 13000.0)

    def test_price_fetch_failure_falls_back_to_trade_price(self):
        trades = [_t("2026-01-02", "KR", "005930", "buy", 10, 10000)]
        monthly = journal.compute_monthly_returns(
            trades, lambda m, c: None, None, today=date(2026, 1, 31))
        # 가격 없음 → 마지막 체결가로 평가 → 손익 0
        self.assertAlmostEqual(monthly[0]["end_equity"], 100000.0)
        self.assertAlmostEqual(monthly[0]["pnl_krw"], 0.0)

    def test_holdings_at_respects_date(self):
        trades = [
            _t("2026-01-02", "KR", "005930", "buy", 10, 10000),
            _t("2026-03-02", "KR", "005930", "sell", 4, 12000),
        ]
        self.assertAlmostEqual(
            journal.holdings_at(trades, date(2026, 2, 1))[("KR", "005930")], 10.0)
        self.assertAlmostEqual(
            journal.holdings_at(trades, date(2026, 3, 2))[("KR", "005930")], 6.0)


class CrudTests(unittest.TestCase):
    def test_add_and_delete_roundtrip(self):
        fake = FakeStore()
        with patch.object(journal, "cloud_store", fake):
            t1 = journal.add_trade({
                "date": "2026-07-01", "market": "KR", "code": "005930",
                "name": "삼성전자", "side": "buy", "qty": 10, "price": 61000,
            })
            t2 = journal.add_trade({
                "date": "2026-07-02", "market": "US", "code": "AAPL",
                "side": "buy", "qty": 2, "price": 315, "fx": 1500,
            })
            self.assertEqual(len(journal.load_trades()), 2)
            removed = journal.delete_trades({t1["id"]})
            self.assertEqual(removed, 1)
            remaining = journal.load_trades()
            self.assertEqual(len(remaining), 1)
            self.assertEqual(remaining[0]["id"], t2["id"])

    def test_bad_stored_data_returns_empty(self):
        fake = FakeStore()
        fake.data[journal.TRADES_FILE] = {"corrupted": True}
        with patch.object(journal, "cloud_store", fake):
            self.assertEqual(journal.load_trades(), [])


if __name__ == "__main__":
    unittest.main()
