import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analyzer  # noqa: E402
import cloud_store  # noqa: E402
import dart  # noqa: E402
import llm  # noqa: E402


class _Response:
    ok = False
    status_code = 500
    text = "server error"


class CloudStoreTests(unittest.TestCase):
    def test_failed_gist_save_keeps_current_session_cache_fresh(self):
        old_cache = cloud_store._gist_cache
        old_data_dir = cloud_store.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cloud_store.DATA_DIR = Path(tmp)
                cloud_store._gist_cache = {
                    "watchlist.json": json.dumps(["005930"], ensure_ascii=False)
                }
                with (
                    patch.object(cloud_store, "_get_credentials", return_value=("pat", "gist")),
                    patch.object(cloud_store.requests, "patch", return_value=_Response()),
                ):
                    cloud_store.save("watchlist.json", ["000660"])
                    self.assertEqual(cloud_store.load("watchlist.json", []), ["000660"])
                saved = json.loads((Path(tmp) / "watchlist.json").read_text(encoding="utf-8"))
                self.assertEqual(saved, ["000660"])
        finally:
            cloud_store._gist_cache = old_cache
            cloud_store.DATA_DIR = old_data_dir


class DartTests(unittest.TestCase):
    def test_income_statement_prefers_cumulative_amounts_when_available(self):
        rows = [{
            "account_nm": "당기순이익",
            "thstrm_amount": "1,000",
            "thstrm_add_amount": "3,000",
        }]
        self.assertEqual(
            dart._find_amount(rows, ["당기순이익"], current=True, cumulative=True),
            3000,
        )

    def test_calc_per_uses_annualized_income_when_available(self):
        ratios = dart.calc_per_pbr(
            "000000",
            current_price=100,
            shares=10,
            fin={"net_income": 50, "net_income_annualized": 200, "equity": 1000},
        )
        self.assertEqual(ratios["eps"], 20)
        self.assertEqual(ratios["per"], 5)
        self.assertEqual(ratios["pbr"], 1)

    def test_placeholder_dart_key_is_not_configured(self):
        with (
            patch.dict(os.environ, {"DART_API_KEY": "여기에_본인의_DART_API_키"}, clear=False),
            patch.object(dart, "load_dotenv", return_value=False),
        ):
            self.assertFalse(dart.is_configured())


class LlmTests(unittest.TestCase):
    def test_placeholder_gemini_key_is_not_configured(self):
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "여기에_GEMINI_API_키"}, clear=False),
            patch.object(llm, "load_dotenv", return_value=False),
        ):
            self.assertFalse(llm.is_configured())


class ScoringTests(unittest.TestCase):
    def test_weighted_score_prioritizes_price_reaction_items(self):
        scores = [
            {"name": "시장 상대강도", "score": 1, "max": 1},
            {"name": "수급", "score": 1, "max": 1},
            {"name": "가치", "score": 1, "max": 1},
        ]
        total, max_possible = analyzer.weighted_score(scores)
        self.assertEqual(total, 5)
        self.assertEqual(max_possible, 5)

    def test_market_regime_is_weighted_as_price_prediction_signal(self):
        total, max_possible = analyzer.weighted_score([
            {"name": "시장 국면", "score": 1, "max": 1},
        ])
        self.assertEqual(total, 2)
        self.assertEqual(max_possible, 2)

    def test_oversold_rsi_is_not_counted_as_buy_signal_by_itself(self):
        df = pd.DataFrame({"Close": list(range(100, 60, -1))})
        item = analyzer.score_momentum(df)
        self.assertLessEqual(item["score"], 0)
        self.assertIn("과매도", item["msg"])

    def test_opinion_thresholds_are_trading_oriented(self):
        self.assertTrue(analyzer.overall_opinion(8, 15).startswith("매수 우위"))
        self.assertTrue(analyzer.overall_opinion(-5, 15).startswith("매도"))

    def test_trade_plan_adds_stop_and_reward_targets(self):
        close = list(range(100, 130))
        df = pd.DataFrame({
            "High": [v + 2 for v in close],
            "Low": [v - 2 for v in close],
            "Close": close,
        })
        scores = [
            {"name": "시장 국면", "score": 1, "max": 1},
            {"name": "시장 상대강도", "score": 1, "max": 1},
            {"name": "수급", "score": 1, "max": 1},
            {"name": "거래량", "score": 1, "max": 1},
            {"name": "추세", "score": 1, "max": 1},
            {"name": "가격 리스크", "score": 0, "max": 0},
        ]
        total, max_possible = analyzer.weighted_score(scores)
        short_total, _ = analyzer.weighted_score(scores, analyzer.SHORT_TERM_ITEMS)
        plan = analyzer.build_trade_plan(df, scores, total, short_total, max_possible)
        self.assertEqual(plan["action"], "매수 우위")
        self.assertLess(plan["stop_loss"], plan["entry_price"])
        self.assertGreater(plan["target_1r"], plan["entry_price"])
        self.assertGreater(plan["target_2r"], plan["target_1r"])


if __name__ == "__main__":
    unittest.main()
