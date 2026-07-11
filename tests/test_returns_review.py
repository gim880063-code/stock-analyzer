"""2026-07 수익률 점검 회귀 테스트 — 업종 상한, 무손실 아카이브, 시간 손절."""
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analyzer  # noqa: E402
import history  # noqa: E402
import holdings_monitor  # noqa: E402
import screening_history  # noqa: E402
import verifier  # noqa: E402


class FakeStore:
    def __init__(self):
        self.data = {}

    def load(self, filename, default):
        return self.data.get(filename, default)

    def save(self, filename, data):
        self.data[filename] = data

    def refresh(self):
        return True


def _r(code, name, total):
    return {"code": code, "name": name, "total": total, "last_close": 1000}


class SectorCapTests(unittest.TestCase):
    IND = {"A1": "278", "A2": "278", "A3": "278", "B1": "300", "C1": None}

    def _fn(self, code):
        return self.IND.get(code)

    def test_third_same_sector_demoted(self):
        results = [_r("A1", "반도체1", 7), _r("A2", "반도체2", 6),
                   _r("A3", "반도체3", 5), _r("B1", "인터넷1", 5)]
        kept, demoted = analyzer.apply_sector_cap(results, industry_fn=self._fn)
        self.assertEqual([r["code"] for r in kept], ["A1", "A2", "B1"])
        self.assertEqual(len(demoted), 1)
        self.assertEqual(demoted[0]["result"]["code"], "A3")
        self.assertIn("업종 집중 상한", demoted[0]["reason"])
        self.assertIn("반도체1", demoted[0]["reason"])  # 통과 종목명 명시

    def test_unknown_sector_not_capped(self):
        results = [_r("A1", "a", 7), _r("A2", "b", 6), _r("C1", "c", 5), _r("A3", "d", 4)]
        kept, demoted = analyzer.apply_sector_cap(results, industry_fn=self._fn)
        self.assertIn("C1", [r["code"] for r in kept])  # 업종 미확인 → 통과 유지
        self.assertEqual([d["result"]["code"] for d in demoted], ["A3"])

    def test_small_list_untouched(self):
        results = [_r("A1", "a", 7), _r("A2", "b", 6)]
        kept, demoted = analyzer.apply_sector_cap(results, industry_fn=self._fn)
        self.assertEqual(len(kept), 2)
        self.assertEqual(demoted, [])

    def test_score_order_priority(self):
        # 점수 내림차순 입력에서 상위 2개가 남아야 함
        results = [_r("A3", "high", 9), _r("A1", "mid", 6), _r("A2", "low", 4)]
        kept, demoted = analyzer.apply_sector_cap(results, industry_fn=self._fn)
        self.assertEqual([r["code"] for r in kept], ["A3", "A1"])
        self.assertEqual(demoted[0]["result"]["code"], "A2")


class ScoreHistoryArchiveTests(unittest.TestCase):
    def test_aged_entries_moved_not_deleted(self):
        fake = FakeStore()
        old_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        mid_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        fake.data[history.FILENAME] = {
            "005930": [
                {"date": old_date, "total": 3, "close": 100.0, "opinion": ""},
                {"date": mid_date, "total": 5, "close": 110.0, "opinion": ""},
            ],
        }
        with patch.object(history, "cloud_store", fake):
            history.record_snapshot("005930", 6, 120.0, "매수")
            main = fake.data[history.FILENAME]["005930"]
            self.assertEqual([e["total"] for e in main], [5, 6])  # 오늘+최근만 본체
            archive = fake.data[history.ARCHIVE_FILENAME]["005930"]
            self.assertEqual([e["date"] for e in archive], [old_date])  # 삭제 아닌 이동
            # load_all(include_archive=True) 로 병합 조회
            merged = history.load_all(include_archive=True)["005930"]
            self.assertEqual(len(merged), 3)
            self.assertEqual(merged[0]["date"], old_date)

    def test_archive_merge_no_duplicates(self):
        fake = FakeStore()
        old_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        entry = {"date": old_date, "total": 3, "close": 100.0, "opinion": ""}
        fake.data[history.FILENAME] = {"005930": [dict(entry)]}
        fake.data[history.ARCHIVE_FILENAME] = {"005930": [dict(entry)]}  # 이미 아카이브됨
        with patch.object(history, "cloud_store", fake):
            history.record_snapshot("005930", 6, 120.0, "매수")
            archive = fake.data[history.ARCHIVE_FILENAME]["005930"]
            self.assertEqual(len(archive), 1)  # 중복 병합 안 됨


class ScreeningHistoryArchiveTests(unittest.TestCase):
    def test_old_dates_archived(self):
        fake = FakeStore()
        old_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
        fake.data[screening_history.FILENAME] = {
            old_date: {"codes": ["005930"], "passed": {}, "dropped": {}, "params": {}, "ran_at": None},
        }
        with patch.object(screening_history, "cloud_store", fake):
            screening_history.record_today(["000660"])
            main = fake.data[screening_history.FILENAME]
            self.assertNotIn(old_date, main)  # 본체에선 이동
            archive = fake.data[screening_history.ARCHIVE_FILENAME]
            self.assertIn(old_date, archive)  # 아카이브에 보존
            self.assertEqual(archive[old_date]["codes"], ["005930"])


def _leader(code, total=5, rel=10.0, vol_ok=True, fund=1):
    """적응 통과 게이트를 통과하는 건전 주도주 형태의 분석 결과."""
    return {
        "code": code, "name": code, "total": total,
        "recent_surge": {"is_surge": True, "metrics": {"d20": 15.0}},
        "scores": [
            {"name": "시장 상대강도", "score": 1, "max": 1, "relative": rel},
            {"name": "가격 리스크", "score": 0, "max": 0,
             "msg": "" if vol_ok else "거래량 미동반 급등"},
            {"name": "재무 건전성", "score": fund, "max": 1},
            {"name": "성장성", "score": 0, "max": 1},
        ],
    }


NORMAL = {"risk_off": False, "overheated": False, "sharp_drop": False}
OVERHEATED = {"risk_off": False, "overheated": True, "sharp_drop": False}


class AdaptiveExpansionTests(unittest.TestCase):
    def test_normal_regime_needs_expansion(self):
        screened = [_leader("A"), _leader("B")]
        self.assertEqual(analyzer.select_adaptive_picks(screened, NORMAL), [])
        picks = analyzer.select_adaptive_picks(
            screened, NORMAL, expansion={"expand": True, "max_n": 5})
        self.assertEqual([r["code"] for r in picks], ["A", "B"])

    def test_defense_regimes_block_even_when_expanded(self):
        screened = [_leader("A")]
        exp = {"expand": True, "max_n": 5}
        for regime in ({"risk_off": True, "overheated": False, "sharp_drop": False},
                       {"risk_off": False, "overheated": True, "sharp_drop": True}):
            self.assertEqual(
                analyzer.select_adaptive_picks(screened, regime, expansion=exp), [])

    def test_overheated_legacy_path_unchanged(self):
        screened = [_leader(f"C{i}", total=10 - i) for i in range(5)]
        picks = analyzer.select_adaptive_picks(screened, OVERHEATED)
        self.assertEqual(len(picks), analyzer.ADAPTIVE_MAX_N)  # 기본 3
        picks_exp = analyzer.select_adaptive_picks(
            screened, OVERHEATED, expansion={"expand": True, "max_n": 5})
        self.assertEqual(len(picks_exp), 5)  # 확대 시 5

    def test_gates_still_apply_when_expanded(self):
        screened = [
            _leader("OK"),
            _leader("NOVOL", vol_ok=False),        # 거래량 미동반
            _leader("WEAKFUND", fund=-1),          # 펀더 악화
            _leader("TOOHOT", rel=30.0),           # 시장대비 과다 초과
        ]
        picks = analyzer.select_adaptive_picks(
            screened, NORMAL, expansion={"expand": True, "max_n": 5})
        self.assertEqual([r["code"] for r in picks], ["OK"])


class ExpansionStateTests(unittest.TestCase):
    """확대 판정은 적응(adaptive) 트랙 단독 — 관찰은 탈락 대조군이라 합산 금지
    (2026-07 실측: 관찰이 가장 나빠서 합산하면 적응의 진짜 성과가 희석됨)."""

    def _patch(self, stats_by_kind):
        def fake_verify(score_type="total", horizon="all", min_hold_days=5, kind="picked"):
            n, e = stats_by_kind.get(kind, (0, None))
            return {"total_count": n, "overall_avg_excess": e}
        return patch.object(verifier, "verify_scouted", fake_verify)

    def test_expands_when_edge_proven(self):
        with self._patch({"adaptive": (25, 2.0), "picked": (20, 0.5)}):
            s = verifier.adaptive_expansion_state()
        self.assertTrue(s["expand"])
        self.assertEqual(s["max_n"], verifier.EXPANDED_MAX_N)
        self.assertEqual(s["n_adaptive"], 25)
        self.assertAlmostEqual(s["avg_adaptive_excess"], 2.0, places=2)

    def test_observed_track_ignored(self):
        # 관찰이 아무리 좋아도(대조군) 적응 표본이 모자라면 확대 안 함
        with self._patch({"adaptive": (7, 5.0), "observed": (100, 9.9), "picked": (20, 0.0)}):
            s = verifier.adaptive_expansion_state()
        self.assertFalse(s["expand"])
        self.assertIn("표본 부족", s["reason"])
        self.assertEqual(s["n_adaptive"], 7)

    def test_no_expand_negative_excess(self):
        with self._patch({"adaptive": (25, -0.5), "picked": (20, 0.0)}):
            s = verifier.adaptive_expansion_state()
        self.assertFalse(s["expand"])
        self.assertIn("≤ 0", s["reason"])

    def test_no_expand_insufficient_edge_vs_picked(self):
        with self._patch({"adaptive": (25, 1.5), "picked": (20, 1.0)}):
            s = verifier.adaptive_expansion_state()
        self.assertFalse(s["expand"])
        self.assertIn("우위 부족", s["reason"])

    def test_never_raises(self):
        with patch.object(verifier, "verify_scouted", side_effect=RuntimeError("boom")):
            s = verifier.adaptive_expansion_state()
        self.assertFalse(s["expand"])


class MoneySummaryTests(unittest.TestCase):
    def test_money_math(self):
        rows_by_kind = {
            "picked": [
                {"return_pct": 10.0, "trail_return_pct": 5.0, "excess_return_pct": 8.0},
                {"return_pct": -20.0, "trail_return_pct": -10.0, "excess_return_pct": -22.0},
            ],
            "adaptive": [
                {"return_pct": 4.0, "trail_return_pct": None, "excess_return_pct": None},
            ],
        }

        def fake_verify(score_type="total", horizon="all", min_hold_days=5, kind="picked"):
            return {"rows": rows_by_kind.get(kind, []), "total_count": len(rows_by_kind.get(kind, []))}

        with patch.object(verifier, "verify_scouted", fake_verify):
            m = verifier.money_summary(horizon="all", per_stock=1_000_000)
        self.assertEqual(m["n"], 3)  # 관찰(observed)은 아예 조회 안 함
        self.assertEqual(m["invested"], 3_000_000)
        # 보유: 1.10 + 0.80 + 1.04 = 2.94백만
        self.assertEqual(m["hold_value"], 2_940_000)
        # 손절: 1.05 + 0.90 + (trail 없음→보유 1.04) = 2.99백만
        self.assertEqual(m["trail_value"], 2_990_000)
        # 시장: (10-8=2%) 1.02 + (-20+22=2%) 1.02 + (초과 없음→원금 1.00) = 3.04백만
        self.assertEqual(m["market_value"], 3_040_000)
        self.assertEqual(m["wins"], 2)

    def test_none_when_no_rows(self):
        with patch.object(verifier, "verify_scouted",
                          lambda **k: {"rows": [], "total_count": 0}):
            self.assertIsNone(verifier.money_summary())


class FocusGateTests(unittest.TestCase):
    def test_focus_subscore_weighted(self):
        r = {"scores": [
            {"name": "시장 상대강도", "score": 1, "max": 1},   # 가중 2
            {"name": "재무 건전성", "score": -1, "max": 1},    # 가중 1
            {"name": "가치", "score": -1, "max": 1},           # 가중 1
            {"name": "거래량", "score": 1, "max": 1},          # 집중 항목 아님
        ]}
        self.assertEqual(analyzer.focus_subscore(r), 2 - 1 - 1)  # = 0 (게이트 통과)

    def test_focus_subscore_negative(self):
        r = {"scores": [
            {"name": "시장 상대강도", "score": -1, "max": 1},
            {"name": "재무 건전성", "score": 1, "max": 1},
            {"name": "가치", "score": -1, "max": 1},
        ]}
        self.assertEqual(analyzer.focus_subscore(r), -2)  # 게이트에서 제외 대상

    def test_focus_subscore_none_when_no_items(self):
        # 집중 항목 데이터가 하나도 없으면 None — 게이트가 막지 않음 (보수적)
        r = {"scores": [{"name": "거래량", "score": 1, "max": 1}]}
        self.assertIsNone(analyzer.focus_subscore(r))


class TimeStopTests(unittest.TestCase):
    def _eval(self, added_days_ago, avg, current):
        holding = {
            "quantity": 10, "avg_price": avg,
            "added_at": (datetime.now() - timedelta(days=added_days_ago)).strftime("%Y-%m-%d"),
        }
        analysis = {"last_close": current, "total": 2, "trade_plan": {"action": "보유"}}
        return holdings_monitor.evaluate_holding("005930", "삼성전자", holding, analysis)

    def test_stagnant_long_hold_alerts(self):
        r = self._eval(120, avg=10000, current=9800)
        msgs = [a["msg"] for a in r["alerts"]]
        self.assertTrue(any("시간 손절 점검" in m for m in msgs), msgs)

    def test_profitable_hold_no_time_alert(self):
        r = self._eval(120, avg=10000, current=12000)
        self.assertFalse(any("시간 손절" in a["msg"] for a in r["alerts"]))

    def test_recent_hold_no_time_alert(self):
        r = self._eval(30, avg=10000, current=9500)
        self.assertFalse(any("시간 손절" in a["msg"] for a in r["alerts"]))


if __name__ == "__main__":
    unittest.main()
