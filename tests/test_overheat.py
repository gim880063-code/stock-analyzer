"""과열장 대응 — 과열 판정 / 관찰 대상 선정 / scouted kind 분리 회귀 테스트."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analyzer  # noqa: E402
import scouted  # noqa: E402


def _series(last: float, ma_level: float, n: int = 210) -> pd.Series:
    """마지막 한 점만 last, 나머지는 ma_level — rolling(200) 평균 ≈ ma_level."""
    return pd.Series([ma_level] * (n - 1) + [last], dtype=float)


class RegimeOverheatTests(unittest.TestCase):
    def test_overheated_far_above_ma(self):
        r = analyzer._regime_from_series(_series(200, 100), ma_window=200)
        self.assertFalse(r["risk_off"])
        self.assertTrue(r["overheated"])
        self.assertGreaterEqual(r["gap_pct"], analyzer.OVERHEAT_GAP_PCT)

    def test_normal_near_ma(self):
        r = analyzer._regime_from_series(_series(105, 100), ma_window=200)
        self.assertFalse(r["risk_off"])
        self.assertFalse(r["overheated"])

    def test_risk_off_below_ma(self):
        r = analyzer._regime_from_series(_series(90, 100), ma_window=200)
        self.assertTrue(r["risk_off"])
        self.assertFalse(r["overheated"])

    def test_data_shortage_has_overheated_key(self):
        r = analyzer._regime_from_series(pd.Series([1.0, 2.0]), ma_window=200)
        self.assertIn("overheated", r)
        self.assertFalse(r["overheated"])


class ObservationTargetTests(unittest.TestCase):
    def _result(self, code, total, rel_score=0, surge=False, d20=None, vol_ok=True):
        scores = [{"name": "시장 상대강도", "score": rel_score, "msg": "", "max": 1}]
        if surge:
            msg = "5일 +30.0% 강한 급등"
            if not vol_ok:
                msg += ", 거래량 0.50배 — 급등에 거래량 미동반"
            scores.append({"name": "가격 리스크", "score": -2, "msg": msg, "max": 0})
        metrics = {"d20": d20} if d20 is not None else {}
        rs = {"is_surge": surge, "metrics": metrics, "triggers": []}
        return {"code": code, "name": code, "total": total, "last_close": 100.0,
                "scores": scores, "recent_surge": rs}

    def test_picks_top_unpassed(self):
        fresh = [self._result("A", 3), self._result("B", 2), self._result("C", 1)]
        out = analyzer.select_observation_targets(
            [], fresh, {"overheated": False}, passed_codes=set(), top_n=2, mom_n=0,
        )
        self.assertEqual([r["code"] for r in out], ["A", "B"])

    def test_excludes_passed_codes(self):
        fresh = [self._result("A", 5), self._result("B", 2)]
        out = analyzer.select_observation_targets(
            [], fresh, {"overheated": False}, passed_codes={"A"}, top_n=5,
        )
        self.assertEqual([r["code"] for r in out], ["B"])

    def test_overheated_adds_leaders_by_momentum(self):
        screened = [
            self._result("L1", 1, rel_score=1, surge=True, d20=60),
            self._result("L2", 1, rel_score=0, surge=True, d20=80),
            self._result("W", 1, rel_score=-1, surge=True, d20=90),   # 상대강도<0 → 제외
            self._result("V", 1, rel_score=1, surge=True, d20=70, vol_ok=False),  # 거래량 미동반 → 제외
            self._result("N", 1, rel_score=1, surge=False, d20=99),   # surge 아님 → 제외
        ]
        out = analyzer.select_observation_targets(
            screened, [], {"overheated": True}, passed_codes=set(), top_n=0, mom_n=5,
        )
        codes = [r["code"] for r in out]
        self.assertEqual(codes, ["L2", "L1"])  # d20 큰 순

    def test_not_overheated_skips_leaders(self):
        screened = [self._result("L1", 1, rel_score=1, surge=True, d20=60)]
        out = analyzer.select_observation_targets(
            screened, [], {"overheated": False}, passed_codes=set(), top_n=0, mom_n=5,
        )
        self.assertEqual(out, [])


class ScoutedKindTests(unittest.TestCase):
    def test_observed_does_not_overwrite_picked(self):
        store = {"005930": {"added_at": "2026-06-01", "added_score": 5, "kind": "picked"}}
        with patch.object(scouted, "load_scouted", return_value=store), \
             patch.object(scouted, "save_scouted"):
            added, skipped = scouted.add_observed_from_analysis(
                [{"code": "005930", "total": 3}], universe="safe",
            )
        self.assertEqual((added, skipped), (0, 1))
        self.assertEqual(store["005930"]["kind"], "picked")

    def test_observed_added_for_new_code(self):
        store: dict = {}
        with patch.object(scouted, "load_scouted", return_value=store), \
             patch.object(scouted, "save_scouted"):
            added, skipped = scouted.add_observed_from_analysis(
                [{"code": "000660", "total": 2, "last_close": 100.0}], universe="safe",
            )
        self.assertEqual(added, 1)
        self.assertEqual(store["000660"]["kind"], "observed")

    def test_picked_promotes_existing_observed(self):
        store = {"005930": {"added_at": "2026-06-01", "added_score": 2, "kind": "observed"}}
        with patch.object(scouted, "load_scouted", return_value=store), \
             patch.object(scouted, "save_scouted"):
            scouted.add_many_from_analysis(
                [{"code": "005930", "total": 5}], universe="safe", kind="picked",
            )
        self.assertEqual(store["005930"]["kind"], "picked")     # 승격
        self.assertEqual(store["005930"]["added_at"], "2026-06-01")  # 추적 시작점 보존

    def test_entry_kind_legacy_defaults_picked(self):
        self.assertEqual(scouted._entry_kind({"added_at": "2026-01-01"}), "picked")
        self.assertEqual(scouted._entry_kind({"kind": "observed"}), "observed")
        self.assertEqual(scouted._entry_kind({"kind": "adaptive"}), "adaptive")


class AdaptivePickTests(unittest.TestCase):
    def _result(self, code, total, relative=None, surge=True, vol_ok=True,
                rsi_extreme=False, fin=0, growth=0, d20=30):
        scores = []
        if relative is not None:
            rel_score = 1 if relative >= 5 else (-1 if relative <= -5 else 0)
            scores.append({"name": "시장 상대강도", "score": rel_score,
                           "relative": relative, "msg": "", "max": 1})
        rmsg = "5일 +30.0% 강한 급등"
        if not vol_ok:
            rmsg += ", 거래량 0.50배 — 급등에 거래량 미동반"
        if rsi_extreme:
            rmsg += ", RSI 85 극단적 과열"
        scores.append({"name": "가격 리스크", "score": -2, "msg": rmsg, "max": 0})
        scores.append({"name": "재무 건전성", "score": fin, "msg": "", "max": 2})
        scores.append({"name": "성장성", "score": growth, "msg": "", "max": 2})
        rs = {"is_surge": surge, "metrics": {"d20": d20}, "triggers": []}
        return {"code": code, "name": code, "total": total, "last_close": 100.0,
                "scores": scores, "recent_surge": rs}

    def test_picks_market_beating_sorted_by_total(self):
        screened = [
            self._result("A", 3, relative=10),
            self._result("B", 5, relative=20),
            self._result("C", 1, relative=8),
        ]
        out = analyzer.select_adaptive_picks(screened, {"overheated": True}, set(), max_n=3)
        self.assertEqual([r["code"] for r in out], ["B", "A", "C"])  # total 내림차순

    def test_excludes_over_excess(self):
        screened = [self._result("X", 5, relative=30)]  # 시장 대비 +30%p → 추격
        self.assertEqual(analyzer.select_adaptive_picks(screened, {"overheated": True}, set()), [])

    def test_excludes_below_market(self):
        screened = [self._result("X", 5, relative=2)]  # 시장 못 이김
        self.assertEqual(analyzer.select_adaptive_picks(screened, {"overheated": True}, set()), [])

    def test_excludes_weak_volume_rsi_fundamentals(self):
        screened = [
            self._result("V", 5, relative=10, vol_ok=False),
            self._result("R", 5, relative=10, rsi_extreme=True),
            self._result("F", 5, relative=10, fin=-1, growth=-1),
        ]
        self.assertEqual(analyzer.select_adaptive_picks(screened, {"overheated": True}, set()), [])

    def test_not_overheated_returns_empty(self):
        screened = [self._result("A", 5, relative=10)]
        self.assertEqual(analyzer.select_adaptive_picks(screened, {"overheated": False}, set()), [])

    def test_respects_max_n_and_passed_codes(self):
        screened = [self._result(c, 5, relative=10) for c in ("A", "B", "C", "D")]
        out = analyzer.select_adaptive_picks(screened, {"overheated": True}, {"A"}, max_n=2)
        codes = [r["code"] for r in out]
        self.assertNotIn("A", codes)
        self.assertEqual(len(codes), 2)


class ScoutedRankPromotionTests(unittest.TestCase):
    def test_promotes_observed_to_adaptive_to_picked(self):
        store = {"X": {"added_at": "2026-06-01", "added_score": 1, "kind": "observed"}}
        with patch.object(scouted, "load_scouted", return_value=store), \
             patch.object(scouted, "save_scouted"):
            scouted.add_adaptive_from_analysis([{"code": "X", "total": 3}])
            self.assertEqual(store["X"]["kind"], "adaptive")
            scouted.add_many_from_analysis([{"code": "X", "total": 5}], kind="picked")
            self.assertEqual(store["X"]["kind"], "picked")
            self.assertEqual(store["X"]["added_at"], "2026-06-01")  # 추적 시작점 보존

    def test_no_demotion(self):
        store = {"X": {"added_at": "2026-06-01", "added_score": 5, "kind": "picked"}}
        with patch.object(scouted, "load_scouted", return_value=store), \
             patch.object(scouted, "save_scouted"):
            scouted.add_observed_from_analysis([{"code": "X", "total": 1}])
            scouted.add_adaptive_from_analysis([{"code": "X", "total": 2}])
            self.assertEqual(store["X"]["kind"], "picked")  # 강등 없음


if __name__ == "__main__":
    unittest.main()
