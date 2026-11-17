import unittest

import numpy as np
import pandas as pd

from v77_prospective_evidence import (
    UNIFORM_BRIER,
    UNIFORM_LOG_LOSS,
    V77_END_DRAW,
    V77_START_DRAW,
    build_v77_evidence_report,
)


def make_summary(start=V77_START_DRAW, count=50, hit_delta=0.05):
    rows = []
    for offset in range(count):
        draw_no = start + offset
        rows.append({
            "目标期号": draw_no,
            "V5.1注数": 10,
            "随机注数": 10,
            "V5.1平均命中数": 0.80 + hit_delta,
            "随机平均命中数": 0.80,
            "平均命中差值": hit_delta,
            "V5.1单注3+率": 0.025,
            "随机单注3+率": 0.024,
            "V5.1整批至少一注3+": 0.22,
            "随机整批至少一注3+": 0.21,
            "V5.1最佳命中数": 3,
            "随机最佳命中数": 3,
            "V5.1_Brier": UNIFORM_BRIER - 0.00001,
            "V5.1_LogLoss": UNIFORM_LOG_LOSS - 0.00001,
        })
    return pd.DataFrame(rows)


class V77ProspectiveEvidenceTests(unittest.TestCase):
    def test_empty_and_partial_windows_are_descriptive_only(self):
        for source in (pd.DataFrame(), make_summary(count=1), make_summary(count=49)):
            report, per_draw, metrics = build_v77_evidence_report(source)
            self.assertFalse(report["analysis_ready"])
            self.assertIsNone(report["primary_endpoint"]["bootstrap_95_ci"])
            self.assertEqual(report["status"], "collecting_fixed_horizon")

    def test_complete_fixed_window_runs_one_primary_bootstrap_interval(self):
        report, per_draw, metrics = build_v77_evidence_report(
            make_summary(hit_delta=0.05),
            bootstrap_repetitions=500,
        )
        self.assertTrue(report["analysis_ready"])
        self.assertEqual(report["status"], "complete_fixed_horizon")
        self.assertEqual(report["scored_unique_draws"], 50)
        self.assertEqual(report["target_window"]["start_draw"], V77_START_DRAW)
        self.assertEqual(report["target_window"]["end_draw"], V77_END_DRAW)
        interval = report["primary_endpoint"]["bootstrap_95_ci"]
        self.assertAlmostEqual(interval["lower"], 0.05, places=12)
        self.assertAlmostEqual(interval["upper"], 0.05, places=12)
        self.assertEqual(len(per_draw), 50)
        self.assertIn("Brier分数", set(metrics["指标"]))

    def test_duplicate_target_is_never_resolved_by_picking_best_row(self):
        source = make_summary(count=50)
        duplicate = source.iloc[[0]].copy()
        duplicate["V5.1平均命中数"] = 6.0
        source = pd.concat([source, duplicate], ignore_index=True)
        report, per_draw, _ = build_v77_evidence_report(
            source,
            bootstrap_repetitions=500,
        )
        self.assertFalse(report["analysis_ready"])
        self.assertEqual(report["status"], "protocol_violation_duplicate_target")
        self.assertEqual(report["duplicate_draws_excluded"], [V77_START_DRAW])
        self.assertEqual(report["scored_unique_draws"], 49)
        self.assertIsNone(report["primary_endpoint"]["bootstrap_95_ci"])
        self.assertNotIn(V77_START_DRAW, per_draw["目标期号"].tolist())

    def test_out_of_window_draws_are_ignored_and_reported(self):
        source = make_summary(count=50)
        outside = make_summary(start=V77_END_DRAW + 1, count=1)
        source = pd.concat([source, outside], ignore_index=True)
        report, per_draw, _ = build_v77_evidence_report(
            source,
            bootstrap_repetitions=500,
        )
        self.assertTrue(report["analysis_ready"])
        self.assertEqual(report["out_of_window_draws_ignored"], [V77_END_DRAW + 1])
        self.assertEqual(len(per_draw), 50)

    def test_pending_duplicate_lock_also_invalidates_target(self):
        source = make_summary(count=50)
        duplicate_locks = [
            {"event_type": "v51_forward_lock", "payload": {"metadata": {"target_draw": V77_START_DRAW}}},
            {"event_type": "v51_forward_lock", "payload": {"metadata": {"target_draw": V77_START_DRAW}}},
        ]
        report, per_draw, _ = build_v77_evidence_report(
            source,
            lock_events=duplicate_locks,
            bootstrap_repetitions=500,
        )
        self.assertFalse(report["analysis_ready"])
        self.assertEqual(report["duplicate_draws_excluded"], [V77_START_DRAW])
        self.assertEqual(report["scored_unique_draws"], 49)
        self.assertNotIn(V77_START_DRAW, per_draw["目标期号"].tolist())

    def test_invalid_metric_ranges_are_rejected(self):
        source = make_summary(count=2)
        source.loc[0, "V5.1单注3+率"] = 1.5
        with self.assertRaises(ValueError):
            build_v77_evidence_report(source)

    def test_unequal_ticket_counts_are_rejected(self):
        source = make_summary(count=2)
        source.loc[0, "随机注数"] = 9
        with self.assertRaises(ValueError):
            build_v77_evidence_report(source)

    def test_bootstrap_is_deterministic(self):
        source = make_summary(hit_delta=0.02)
        report_a, _, _ = build_v77_evidence_report(source, bootstrap_repetitions=500, bootstrap_seed=12)
        report_b, _, _ = build_v77_evidence_report(source, bootstrap_repetitions=500, bootstrap_seed=12)
        self.assertEqual(
            report_a["primary_endpoint"]["bootstrap_95_ci"],
            report_b["primary_endpoint"]["bootstrap_95_ci"],
        )


if __name__ == "__main__":
    unittest.main()
