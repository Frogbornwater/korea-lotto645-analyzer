import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from v77_prospective_evidence import (
    UNIFORM_BRIER,
    UNIFORM_LOG_LOSS,
    V77_END_DRAW,
    V77_START_DRAW,
    V77_BOOTSTRAP_REPLICATES,
    V77_BOOTSTRAP_SEED,
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

def make_lock_events(start=V77_START_DRAW, count=50):
    config = {
        "model": "V5.1联合概率选号",
        "engine_version": "5.0.0-predictive-research",
        "training_window": 500,
        "ridge_alpha": 1.0,
        "pair_strength": 0.5,
        "requested_overlap": 4,
        "effective_overlap": 4,
        "ticket_count": 10,
        "seed": 5050,
    }
    events = []
    for offset in range(count):
        draw_no = start + offset
        events.append({
            "event_type": "v51_forward_lock",
            "payload": {
                "metadata": {
                    "target_draw": draw_no,
                    "engine_version": config["engine_version"],
                    "training_window": config["training_window"],
                    "ridge_alpha": config["ridge_alpha"],
                    "pair_strength": config["pair_strength"],
                    "requested_overlap": config["requested_overlap"],
                    "ticket_count": config["ticket_count"],
                    "seed": config["seed"],
                    "random_seed": config["seed"] ^ 0x5A17,
                },
                "configuration": dict(config),
            },
        })
    return events


class V77ProspectiveEvidenceTests(unittest.TestCase):
    def test_committed_protocol_file_matches_code_constants(self):
        protocol_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "forward_lock_protocols"
            / "v77_v51_forward_1251_1300.json"
        )
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["protocol_id"], "v77-v51-forward-50-draws")
        self.assertEqual(protocol["target_draw_window"]["start"], V77_START_DRAW)
        self.assertEqual(protocol["target_draw_window"]["end"], V77_END_DRAW)
        self.assertEqual(protocol["target_draw_window"]["count"], 50)
        self.assertEqual(
            protocol["primary_endpoint"]["inference"]["repetitions"],
            V77_BOOTSTRAP_REPLICATES,
        )
        self.assertEqual(
            protocol["primary_endpoint"]["inference"]["seed"],
            V77_BOOTSTRAP_SEED,
        )
        self.assertIn("Do not stop early", " ".join(protocol["protocol_guards"]))
        self.assertIn("independently verifiable hash-only", protocol["timestamp_anchor_policy"]["requirement"])

    def test_empty_and_partial_windows_are_descriptive_only(self):
        for source in (pd.DataFrame(), make_summary(count=1), make_summary(count=49)):
            report, per_draw, metrics = build_v77_evidence_report(source)
            self.assertIn("self-reported", report["timestamp_anchor_policy"]["draws_1252_to_1300"])
            self.assertFalse(report["analysis_ready"])
            self.assertIsNone(report["primary_endpoint"]["bootstrap_95_ci"])
            self.assertEqual(report["status"], "collecting_fixed_horizon")

    def test_complete_fixed_window_runs_one_primary_bootstrap_interval(self):
        report, per_draw, metrics = build_v77_evidence_report(
            make_summary(hit_delta=0.05),
            lock_events=make_lock_events(),
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
            lock_events=make_lock_events() + make_lock_events(start=V77_END_DRAW + 1, count=1),
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

    def test_model_configuration_must_match_committed_1251_lock(self):
        events = make_lock_events()
        events[1]["payload"]["configuration"]["pair_strength"] = 1.0
        report, _, _ = build_v77_evidence_report(
            make_summary(count=50),
            lock_events=events,
            bootstrap_repetitions=500,
        )
        self.assertFalse(report["analysis_ready"])
        self.assertEqual(report["status"], "protocol_violation_configuration_mismatch")
        self.assertEqual(report["configuration_mismatch_draws"], [V77_START_DRAW + 1])
        self.assertIsNone(report["primary_endpoint"]["bootstrap_95_ci"])

    def test_all_target_draws_need_a_lock_record(self):
        report, _, _ = build_v77_evidence_report(
            make_summary(count=50),
            lock_events=make_lock_events(count=49),
            bootstrap_repetitions=500,
        )
        self.assertFalse(report["analysis_ready"])
        self.assertEqual(report["missing_lock_draws"], [V77_END_DRAW])

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
        locks = make_lock_events()
        report_a, _, _ = build_v77_evidence_report(source, lock_events=locks, bootstrap_repetitions=500, bootstrap_seed=12)
        report_b, _, _ = build_v77_evidence_report(source, lock_events=locks, bootstrap_repetitions=500, bootstrap_seed=12)
        self.assertEqual(
            report_a["primary_endpoint"]["bootstrap_95_ci"],
            report_b["primary_endpoint"]["bootstrap_95_ci"],
        )


if __name__ == "__main__":
    unittest.main()
