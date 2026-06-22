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
import holdings_monitor  # noqa: E402
import llm  # noqa: E402
import portfolio  # noqa: E402
import verifier  # noqa: E402


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

    def test_fetch_keeps_last_good_on_failure_and_refetches_after_ttl(self):
        """캐시 TTL 만료 후 재조회가 실패해도 직전 정상본을 유지(빈 값 오염 금지)."""
        old_cache, old_at, old_ttl = (
            cloud_store._gist_cache,
            cloud_store._gist_cache_at,
            cloud_store._CACHE_TTL_SEC,
        )
        try:
            cloud_store._gist_cache = None
            cloud_store._gist_cache_at = 0.0
            cloud_store._CACHE_TTL_SEC = 60.0
            calls = {"n": 0}

            class _Ok:
                ok = True

                def json(self):
                    return {"files": {"scouted.json": {"content": '{"005930": {}}'}}}

            class _Fail:
                ok = False
                status_code = 500
                text = "err"

            def fake_get(*a, **k):
                calls["n"] += 1
                return _Ok() if calls["n"] == 1 else _Fail()

            with (
                patch.object(cloud_store, "_get_credentials", return_value=("pat", "gist")),
                patch.object(cloud_store.requests, "get", side_effect=fake_get),
            ):
                first = cloud_store._fetch_gist_files()
                self.assertIn("scouted.json", first)
                cloud_store._fetch_gist_files()          # TTL 안 — 재조회 없음
                self.assertEqual(calls["n"], 1)
                cloud_store._gist_cache_at = 0.0         # TTL 만료 강제
                third = cloud_store._fetch_gist_files()  # 재조회 시도 → 실패
                self.assertEqual(calls["n"], 2)
                self.assertIn("scouted.json", third)     # 직전 정상본 유지
        finally:
            cloud_store._gist_cache = old_cache
            cloud_store._gist_cache_at = old_at
            cloud_store._CACHE_TTL_SEC = old_ttl

    def test_refresh_false_when_unreadable_true_when_ok(self):
        """refresh(): 원격을 못 읽으면 False(→ 호출측이 덮어쓰기 보류), 읽히면 True."""
        old_cache, old_at = cloud_store._gist_cache, cloud_store._gist_cache_at
        try:
            class _Ok:
                ok = True

                def json(self):
                    return {"files": {}}

            class _Fail:
                ok = False
                status_code = 500
                text = "err"

            cloud_store._gist_cache = None
            cloud_store._gist_cache_at = 0.0
            with (
                patch.object(cloud_store, "_get_credentials", return_value=("pat", "gist")),
                patch.object(cloud_store.requests, "get", return_value=_Fail()),
            ):
                self.assertFalse(cloud_store.refresh())

            cloud_store._gist_cache = None
            cloud_store._gist_cache_at = 0.0
            with (
                patch.object(cloud_store, "_get_credentials", return_value=("pat", "gist")),
                patch.object(cloud_store.requests, "get", return_value=_Ok()),
            ):
                self.assertTrue(cloud_store.refresh())
        finally:
            cloud_store._gist_cache = old_cache
            cloud_store._gist_cache_at = old_at


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

    def test_market_regime_uses_weight_one_to_avoid_market_overdominance(self):
        # 시장 국면은 모든 종목에 같은 ± 부여 → 종합점수가 시장 trend에 과도 의존하지
        # 않게 가중치 1로 고정. 가격 반응 신호 (수급/거래량/공시/시장 상대강도, 가중치 2)
        # 와 차별화.
        total, max_possible = analyzer.weighted_score([
            {"name": "시장 국면", "score": 1, "max": 1},
        ])
        self.assertEqual(total, 1)
        self.assertEqual(max_possible, 1)

    def test_oversold_rsi_is_not_counted_as_buy_signal_by_itself(self):
        df = pd.DataFrame({"Close": list(range(100, 60, -1))})
        item = analyzer.score_momentum(df)
        self.assertLessEqual(item["score"], 0)
        self.assertIn("과매도", item["msg"])

    def test_momentum_neutralized_in_total_score(self):
        # 모멘텀(RSI)은 의심 항목 + 역방향 IC(2026-06) 로 종합점수 가중치 0 (관찰용).
        # +1 모멘텀을 더해도 총점·max 가 변하지 않아야 한다.
        self.assertEqual(analyzer.SCORE_WEIGHTS["모멘텀"], 0)
        base = [{"name": "추세", "score": 1, "max": 1}]
        with_mom = base + [{"name": "모멘텀", "score": 1, "max": 1}]
        self.assertEqual(analyzer.weighted_score(with_mom), analyzer.weighted_score(base))

    def test_opinion_text_stays_advisory_not_prescriptive(self):
        # 앱 철학: 매수/매도 추천 아님. 처방형("매수하세요") 금지, 서술형 유지.
        positive = analyzer.overall_opinion(8, 15)
        negative = analyzer.overall_opinion(-5, 15)
        self.assertTrue(positive.startswith("긍정"))
        self.assertTrue(negative.startswith("위험"))
        # 처방형 키워드는 들어가면 안 됨
        for text in [positive, negative]:
            self.assertNotIn("매수하세요", text)
            self.assertNotIn("매도하세요", text)

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
        # action 라벨도 서술형 — "매수 우위" 같은 처방형 아님
        self.assertEqual(plan["action"], "긍정 신호 강함")
        self.assertLess(plan["stop_loss"], plan["entry_price"])
        self.assertGreater(plan["target_1r"], plan["entry_price"])
        self.assertGreater(plan["target_2r"], plan["target_1r"])


class VerifierConsistencyTests(unittest.TestCase):
    """가중치 정책이 바뀌어도 시뮬레이션 bucket 분석이 일관성 유지하는지."""

    def test_total_score_recomputed_when_added_scores_present(self):
        """added_scores 가 있으면 stored added_score 무시하고 현재 가중치로 재계산."""
        import verifier
        info = {
            "added_score": 999,  # 의도적으로 비현실적 값 — 무시되어야
            "added_scores": {
                "시장 상대강도": 1,  # weight 2 → 2
                "수급": 1,            # weight 2 → 2
                "추세": 1,            # weight 1 → 1
            },
        }
        self.assertEqual(verifier._extract_score(info, "total"), 5)

    def test_total_falls_back_to_stored_for_legacy_entries(self):
        """added_scores 없는 옛 엔트리는 stored added_score 그대로."""
        import verifier
        info = {"added_score": 7}
        self.assertEqual(verifier._extract_score(info, "total"), 7)


class AlignmentRiskTests(unittest.TestCase):
    """단기 신호가 동시 정렬되면 정점 매수 페널티가 종합점수에서 빠지는지."""

    def _scores(self, **overrides):
        """기본 0점 항목, overrides로 일부만 +/- 설정."""
        defaults = {
            "추세": 0, "모멘텀": 0, "거래량": 0, "수급": 0,
            "시장 상대강도": 0, "시장 국면": 0,
        }
        defaults.update(overrides)
        return [{"name": k, "score": v, "max": 1} for k, v in defaults.items()]

    def test_no_penalty_when_three_or_fewer_aligned(self):
        scores = self._scores(추세=1, 모멘텀=1, 거래량=1)  # 3개
        item = analyzer.score_alignment_risk(scores)
        self.assertEqual(item["score"], 0)

    def test_minus_one_when_four_aligned(self):
        scores = self._scores(추세=1, 모멘텀=1, 거래량=1, 수급=1)
        item = analyzer.score_alignment_risk(scores)
        self.assertEqual(item["score"], -1)

    def test_minus_two_when_five_or_more_aligned(self):
        scores = self._scores(추세=1, 모멘텀=1, 거래량=1, 수급=1, **{"시장 상대강도": 1})
        item = analyzer.score_alignment_risk(scores)
        self.assertEqual(item["score"], -2)

    def test_penalty_in_short_term_items(self):
        """정렬 위험은 단기 항목으로 분류돼야 단기 부분합도 함께 보수화됨."""
        self.assertIn("정렬 위험", analyzer.SHORT_TERM_ITEMS)

    def test_penalty_excluded_from_mid_term_items(self):
        """중기 부분합에는 들어가면 안 됨 (단기 정렬 페널티이므로)."""
        self.assertNotIn("정렬 위험", analyzer.MID_TERM_ITEMS)


class VerifierHorizonTests(unittest.TestCase):
    """horizon별 수익률 계산이 영업일 기준으로 정확한지."""

    def _make_series(self, start="2026-01-02", n=30, prices=None):
        idx = pd.bdate_range(start=start, periods=n)
        if prices is None:
            prices = list(range(100, 100 + n))  # 100, 101, 102, ...
        return pd.Series(prices, index=idx, dtype=float)

    def test_horizon_5d_returns_close_at_5_trading_days_out(self):
        """5d horizon은 발굴일 + 5 영업일 시점 종가를 반환."""
        import verifier
        series = self._make_series(prices=[100.0 + i for i in range(30)])
        start, end, days, _ = verifier._horizon_return(series, "2026-01-02", "5d")
        self.assertEqual(start, 100.0)
        self.assertEqual(end, 105.0)  # 5영업일 후
        self.assertEqual(days, 5)

    def test_horizon_returns_none_when_window_not_reached(self):
        """발굴일 + horizon이 데이터 끝을 넘으면 None — 보유기간 부족."""
        import verifier
        series = self._make_series(n=10)  # 10영업일치만
        start, end, days, _ = verifier._horizon_return(series, "2026-01-02", "20d")
        self.assertIsNone(start)
        self.assertIsNone(days)

    def test_horizon_all_uses_last_close(self):
        """horizon='all' 은 발굴일 ~ 마지막 거래일."""
        import verifier
        series = self._make_series(n=10, prices=[100.0, 101, 102, 103, 104, 105, 106, 107, 108, 110])
        start, end, days, _ = verifier._horizon_return(series, "2026-01-02", "all")
        self.assertEqual(start, 100.0)
        self.assertEqual(end, 110.0)
        self.assertEqual(days, 9)

    def test_start_date_snaps_to_next_trading_day(self):
        """발굴일이 주말이면 다음 영업일로 자동 조정."""
        import verifier
        series = self._make_series(start="2026-01-05", n=10)  # 월요일부터
        # 2026-01-03(토) 은 비영업일 → 다음 영업일 2026-01-05 가 start 가 돼야 함
        start, end, days, _ = verifier._horizon_return(series, "2026-01-03", "5d")
        self.assertIsNotNone(start)
        self.assertEqual(days, 5)


class VerifierBucketTests(unittest.TestCase):
    """분위수 기반 버킷이 점수 분포에 적응하는지."""

    def test_quantile_buckets_splits_into_thirds(self):
        """충분한 데이터에서 상/중/하 3분할."""
        import verifier
        scores = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        buckets = verifier._quantile_buckets(scores)
        self.assertEqual(len(buckets), 3)
        # 모든 점수가 어딘가에 잡혀야 함
        assigned = [
            any(pred(s) for _, pred in buckets) for s in scores
        ]
        self.assertTrue(all(assigned))

    def test_quantile_buckets_collapses_when_distribution_too_narrow(self):
        """점수가 한 값에 너무 몰리면 단일 '전체' 버킷."""
        import verifier
        scores = [5, 5, 5, 5, 5, 5]
        buckets = verifier._quantile_buckets(scores)
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0][0], "전체")

    def test_quantile_buckets_handles_too_few_scores(self):
        """3개 미만이면 단일 '전체' 버킷."""
        import verifier
        buckets = verifier._quantile_buckets([7, 8])
        self.assertEqual(len(buckets), 1)


class VerifierStatsTests(unittest.TestCase):
    """bucket_stats가 절대/초과수익 통계 모두 산출하는지."""

    def test_bucket_stats_separates_abs_and_excess_returns(self):
        import verifier
        rows = [
            {"added_score": 8, "return_pct": 5.0, "excess_return_pct": 2.0},
            {"added_score": 9, "return_pct": -2.0, "excess_return_pct": 3.0},
            {"added_score": 7, "return_pct": 1.0, "excess_return_pct": -1.0},
            {"added_score": 3, "return_pct": -5.0, "excess_return_pct": -8.0},
            {"added_score": 4, "return_pct": 0.0, "excess_return_pct": -2.0},
            {"added_score": 2, "return_pct": -3.0, "excess_return_pct": -4.0},
        ]
        stats = verifier._bucket_stats(rows, score_key="added_score")
        # 어떤 버킷이 잡히든 평균/중앙값/승률/초과수익 키가 모두 있어야 함
        for label, s in stats.items():
            self.assertIn("avg_return", s)
            self.assertIn("median_return", s)
            self.assertIn("win_rate", s)
            self.assertIn("avg_excess", s)
            self.assertIn("excess_win_rate", s)


class PositionSizeTests(unittest.TestCase):
    def test_basic_risk_sizing(self):
        out = analyzer.position_size(
            entry_price=100, stop_loss=90, account_equity=1_000_000,
            risk_per_trade_pct=1.0, max_position_pct=20.0,
        )
        self.assertTrue(out["ok"])
        self.assertEqual(out["shares"], 1000)        # 10,000원 리스크 / 10원 손절폭
        self.assertFalse(out["capped"])
        self.assertEqual(out["position_pct"], 10.0)  # 100,000 / 1,000,000

    def test_capped_by_max_position(self):
        out = analyzer.position_size(
            entry_price=100, stop_loss=99, account_equity=1_000_000,
            risk_per_trade_pct=1.0, max_position_pct=20.0,
        )
        self.assertTrue(out["ok"])
        self.assertTrue(out["capped"])
        self.assertEqual(out["shares"], 2000)        # 최대비중 20% = 200,000 / 100원
        self.assertEqual(out["position_pct"], 20.0)

    def test_no_stop_is_not_ok(self):
        out = analyzer.position_size(100, None, 1_000_000)
        self.assertFalse(out["ok"])
        self.assertEqual(out["shares"], 0)

    def test_stop_above_entry_is_not_ok(self):
        out = analyzer.position_size(100, 105, 1_000_000)
        self.assertFalse(out["ok"])

    def test_tiny_account_one_share_exceeds_risk(self):
        out = analyzer.position_size(
            entry_price=100, stop_loss=50, account_equity=1000,
            risk_per_trade_pct=1.0,
        )
        self.assertFalse(out["ok"])
        self.assertIn("1주", out["note"])


class HoldingsMonitorTests(unittest.TestCase):
    def _analysis(self, last_close, action="중립", total=0):
        return {
            "last_close": last_close, "total": total, "name": "테스트",
            "trade_plan": {"action": action},
        }

    def _levels(self, ev):
        return {a["level"] for a in ev["alerts"]}

    def test_stop_breach_raises_high_alert(self):
        holding = {"quantity": 10, "avg_price": 100, "stop_loss": 95}
        ev = holdings_monitor.evaluate_holding("X", "테스트", holding, self._analysis(94))
        self.assertIn("high", self._levels(ev))
        self.assertTrue(any("손절" in a["msg"] for a in ev["alerts"]))

    def test_default_stop_used_when_no_stop_stored(self):
        # 손절선 미저장 → 평단 -8% = 92 기준. 91이면 손절 조건.
        holding = {"quantity": 10, "avg_price": 100}
        ev = holdings_monitor.evaluate_holding("X", "테스트", holding, self._analysis(91))
        self.assertIn("high", self._levels(ev))
        self.assertTrue(any("기본 손절선" in a["msg"] for a in ev["alerts"]))

    def test_target_reached_is_info_not_high(self):
        holding = {"quantity": 10, "avg_price": 100, "target_1r": 110}
        ev = holdings_monitor.evaluate_holding("X", "테스트", holding, self._analysis(112))
        self.assertIn("info", self._levels(ev))
        self.assertNotIn("high", self._levels(ev))
        self.assertTrue(any("1R 목표" in a["msg"] for a in ev["alerts"]))

    def test_trailing_stop_medium_alert(self):
        holding = {"quantity": 10, "avg_price": 100, "peak_price": 130}
        ev = holdings_monitor.evaluate_holding(
            "X", "테스트", holding, self._analysis(117), trail_pct=10.0,
        )
        self.assertIn("medium", self._levels(ev))
        self.assertTrue(any("고점 대비" in a["msg"] for a in ev["alerts"]))

    def test_risk_signal_deterioration(self):
        holding = {"quantity": 10, "avg_price": 100}
        ev = holdings_monitor.evaluate_holding(
            "X", "테스트", holding, self._analysis(100, action="위험 우세", total=-3),
        )
        self.assertIn("medium", self._levels(ev))
        self.assertTrue(any("근거 약화" in a["msg"] for a in ev["alerts"]))

    def test_healthy_holding_no_alerts(self):
        holding = {"quantity": 10, "avg_price": 100, "stop_loss": 95,
                   "target_1r": 200, "peak_price": 120}
        ev = holdings_monitor.evaluate_holding(
            "X", "테스트", holding, self._analysis(120, action="긍정 우세", total=5),
        )
        self.assertEqual(ev["alerts"], [])


class RiskSettingsTests(unittest.TestCase):
    def test_load_settings_fills_defaults_and_coerces_types(self):
        with patch.object(portfolio.cloud_store, "load", return_value={"account_equity": "5000000"}):
            s = portfolio.load_settings()
        self.assertEqual(s["account_equity"], 5_000_000)        # 문자열 → int
        self.assertEqual(s["risk_per_trade_pct"], 1.0)          # 기본값 채움
        self.assertEqual(s["max_position_pct"], 20.0)
        self.assertEqual(s["risk_off_enabled"], True)
        self.assertEqual(s["risk_off_score_boost"], 2)


class ScoreNameAliasTests(unittest.TestCase):
    def test_old_name_mapped_to_current(self):
        import history
        out = history._normalize_scores({"재무": 2, "추세": 1})
        self.assertEqual(out, {"재무 건전성": 2, "추세": 1})

    def test_current_name_wins_when_both_present(self):
        import history
        out = history._normalize_scores({"재무": 1, "재무 건전성": 2})
        self.assertEqual(out["재무 건전성"], 2)
        self.assertNotIn("재무", out)

    def test_no_alias_unchanged(self):
        import history
        d = {"추세": 1, "재무 건전성": 2}
        self.assertEqual(history._normalize_scores(d), d)


class TrailingStopTests(unittest.TestCase):
    def _s(self, vals):
        return pd.Series([float(v) for v in vals])

    def test_stops_on_drawdown(self):
        import verifier
        # 100 진입 → 고점 120 → 120의 -10%(108) 이하인 105에서 청산
        ret, days, stopped = verifier._trailing_stop_return(
            self._s([100, 110, 120, 105, 130]), 0, 4, 10.0)
        self.assertTrue(stopped)
        self.assertEqual(days, 3)
        self.assertAlmostEqual(ret, 5.0)

    def test_no_stop_holds_to_end(self):
        import verifier
        ret, days, stopped = verifier._trailing_stop_return(
            self._s([100, 102, 101, 103, 108]), 0, 4, 10.0)
        self.assertFalse(stopped)
        self.assertEqual(days, 4)
        self.assertAlmostEqual(ret, 8.0)

    def test_invalid_returns_none(self):
        import verifier
        self.assertEqual(
            verifier._trailing_stop_return(self._s([100, 101]), 0, 0, 10.0),
            (None, None, False))


class MarketRegimeTests(unittest.TestCase):
    def _series(self, values):
        return pd.Series([float(v) for v in values])

    def test_risk_off_when_below_long_ma(self):
        s = self._series([120] * 200 + [90])
        self.assertTrue(analyzer._regime_from_series(s)["risk_off"])

    def test_risk_on_when_above_long_ma(self):
        s = self._series([100] * 200 + [120])
        self.assertFalse(analyzer._regime_from_series(s)["risk_off"])

    def test_insufficient_data_holds_judgement(self):
        r = analyzer._regime_from_series(self._series([1, 2, 3]))
        self.assertFalse(r["risk_off"])
        self.assertIn("보류", r["label"])

    def test_effective_min_score_raises_only_when_risk_off_and_enabled(self):
        on = {"risk_off_enabled": True, "risk_off_score_boost": 2}
        self.assertEqual(analyzer.effective_min_score(5, {"risk_off": True}, on), (7, 2))
        self.assertEqual(analyzer.effective_min_score(5, {"risk_off": False}, on), (5, 0))
        self.assertEqual(
            analyzer.effective_min_score(5, {"risk_off": True}, {"risk_off_enabled": False}),
            (5, 0),
        )


class ItemICTests(unittest.TestCase):
    """rank-IC(순위상관) 계산기 검증 — verify_item_scores 항목별 예측력의 핵심."""

    def test_perfect_monotonic(self):
        xs = [1, 2, 3, 4, 5] * 10
        self.assertEqual(verifier._spearman_ic(xs, xs), 1.0)
        self.assertEqual(verifier._spearman_ic(xs, [5, 4, 3, 2, 1] * 10), -1.0)

    def test_ties_handled(self):
        # 거친 정수 점수(-1/0/+1)도 완전 단조면 동점 보정 후 +1
        xs = [-1, 0, 1] * 20
        self.assertEqual(verifier._spearman_ic(xs, xs), 1.0)

    def test_insufficient_returns_none(self):
        self.assertIsNone(verifier._spearman_ic([1, 2] * 5, [1, 2] * 5))   # 표본 < IC_MIN_OBS
        self.assertIsNone(verifier._spearman_ic([1] * 40, list(range(40))))  # x 분산 없음

    def test_noise_near_zero(self):
        xs = list(range(60))
        ys = [(i * 37) % 11 for i in range(60)]   # 사실상 무관
        ic = verifier._spearman_ic(xs, ys)
        self.assertIsNotNone(ic)
        self.assertLess(abs(ic), 0.3)


class WalkForwardAggregateTests(unittest.TestCase):
    """verifier._aggregate_walk_forward — 날짜별 IC 시계열 집계의 정확성."""

    @staticmethod
    def _make(rel, seed=42):
        import random
        random.seed(seed)
        by_date = {}
        for di in range(12):
            rows = []
            for s in range(-5, 6):          # 점수 -5..5
                for _ in range(3):          # 점수당 3종목 → 날짜별 33종목
                    rows.append((float(s), rel * s + random.gauss(0, 3)))
            by_date[f"2026-05-{di + 1:02d}"] = rows
        return by_date

    def test_positive_signal_detected(self):
        r = verifier._aggregate_walk_forward(self._make(1.0))
        self.assertEqual(r["n_periods"], 12)
        self.assertFalse(r["insufficient"])
        self.assertGreater(r["mean_ic"], 0.3)
        self.assertGreater(r["t_stat"], 5)
        self.assertGreater(r["mean_spread"], 0)

    def test_inverted_signal_detected(self):
        r = verifier._aggregate_walk_forward(self._make(-1.0))
        self.assertLess(r["mean_ic"], -0.3)
        self.assertLess(r["t_stat"], -5)
        self.assertLess(r["mean_spread"], 0)

    def test_null_signal_not_significant(self):
        r = verifier._aggregate_walk_forward(self._make(0.0))
        self.assertLess(abs(r["t_stat"]), 3)   # 무신호 → 통계적으로 유의하지 않음

    def test_small_cross_section_excluded(self):
        # 한 날짜 종목 수가 WF_MIN_CROSS_SECTION 미만이면 그 날짜는 집계에서 빠진다.
        r = verifier._aggregate_walk_forward({"2026-05-01": [(1.0, 2.0)] * 5})
        self.assertEqual(r["n_periods"], 0)
        self.assertTrue(r["insufficient"])


if __name__ == "__main__":
    unittest.main()
