# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v89_prospective_cohort import analyze_v89_cohort, load_score_cohort, run_v89


def score(draw: int, model_success: bool, random_success: bool, *, timing: str = "verified_commit_date_precedes_draw_date"):
    groups = {}
    for name, success in (
        ("V5_raw_probability", model_success),
        ("V8.1_batch_quota", model_success),
        ("V8.5_low_overlap", model_success),
        ("uniform_random_control", random_success),
    ):
        groups[name] = {
            "batch_at_least_3_hits": success,
            "mean_hits_per_ticket": 0.9 if success else 0.7,
            "ticket_3plus_rate": 0.1 if success else 0.0,
        }
    return {
        "target_draw": draw,
        "score_id": f"score-{draw}",
        "actual_main_numbers": [1, 2, 3, 4, 5, 6],
        "pre_draw_history_fingerprint_verified": True,
        "commitment_timing": {"status": timing},
        "strategy_metrics": groups,
    }


class V89ProspectiveCohortTests(unittest.TestCase):
    def test_paired_cohort_counts_and_holm_adjustment(self):
        records = [{"target_draw": i, "score": score(i, True, False)} for i in range(1, 21)]
        report = analyze_v89_cohort(records, minimum_inferential_draws=20)
        self.assertEqual(report["scored_draws"], 20)
        self.assertTrue(report["inferentially_ready"])
        for comparison in report["comparisons"]:
            self.assertEqual(comparison["paired_wins"], 20)
            self.assertEqual(comparison["paired_losses"], 0)
            self.assertLess(comparison["holm_adjusted_p"], 0.01)

    def test_single_draw_is_not_inferentially_ready(self):
        report = analyze_v89_cohort(
            [{"target_draw": 1253, "score": score(1253, True, False)}],
            minimum_inferential_draws=100,
        )
        self.assertFalse(report["inferentially_ready"])
        self.assertEqual(report["scored_draws"], 1)

    def test_unverified_timing_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text(json.dumps(score(1253, True, False, timing="same_calendar_day_order_unverified")))
            (root / "b.json").write_text(json.dumps(score(1254, True, False)))
            accepted, excluded = load_score_cohort([root / "a.json", root / "b.json"])
            self.assertEqual([x["target_draw"] for x in accepted], [1254])
            self.assertEqual(excluded[0]["reason"], "commitment_timing_not_verified")

    def test_duplicate_draw_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a.json", "b.json"):
                (root / name).write_text(json.dumps(score(1253, True, False)))
            with self.assertRaisesRegex(ValueError, "多个评分文件"):
                load_score_cohort([root / "a.json", root / "b.json"])

    def test_report_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores = root / "scores"
            scores.mkdir()
            (scores / "one.json").write_text(json.dumps(score(1253, True, False)))
            output = root / "report.json"
            run_v89(scores, output)
            with self.assertRaises(FileExistsError):
                run_v89(scores, output)


if __name__ == "__main__":
    unittest.main()
