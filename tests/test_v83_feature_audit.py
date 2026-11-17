# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v83_feature_audit import (
    FIRST_ORDER_FEATURES,
    _holm_adjust,
    evaluate_v83_feature_audit,
)


def make_history(n: int = 110, seed: int = 8301) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(1, n + 1):
        numbers = sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False))
        rows.append({
            "draw_no": draw,
            "date": "",
            "n1": numbers[0],
            "n2": numbers[1],
            "n3": numbers[2],
            "n4": numbers[3],
            "n5": numbers[4],
            "n6": numbers[5],
            "bonus": next(x for x in range(1, 46) if x not in numbers),
        })
    return pd.DataFrame(rows)


class V83FeatureAuditTests(unittest.TestCase):
    def test_holm_adjustment_is_monotone_and_bounded(self):
        raw = [0.01, 0.04, 0.03, 0.2]
        adjusted = _holm_adjust(raw)
        self.assertEqual(len(adjusted), len(raw))
        self.assertTrue(all(0.0 <= p <= 1.0 for p in adjusted))
        ordered = sorted(zip(raw, adjusted))
        self.assertTrue(all(
            ordered[i][1] <= ordered[i + 1][1] + 1e-12
            for i in range(len(ordered) - 1)
        ))

    def test_feature_list_excludes_pair_features(self):
        self.assertEqual(len(FIRST_ORDER_FEATURES), 8)
        self.assertNotIn("pair_mean_log_lift", FIRST_ORDER_FEATURES)
        self.assertNotIn("pair_sd_log_lift", FIRST_ORDER_FEATURES)

    def test_small_rolling_audit_returns_all_ablation_and_placebo_metrics(self):
        history = make_history(110, seed=8302)
        report, detail = evaluate_v83_feature_audit(
            history,
            test_draws=12,
            training_window=50,
            min_history=50,
            bootstrap_repetitions=200,
            block_length=3,
            seed=8303,
        )
        self.assertEqual(report["test_draws"], 12)
        self.assertEqual(report["target_start_draw"], 99)
        self.assertEqual(report["target_end_draw"], 110)
        self.assertEqual(len(detail), 12)
        self.assertEqual(len(report["feature_comparisons"]), 8 * 4)
        self.assertEqual(len(report["feature_summaries"]), 8)
        self.assertEqual(report["bootstrap"]["feature_tests"], 32)
        self.assertFalse(report["evidence_gate"]["production_model_change_allowed"])
        self.assertFalse(report["evidence_gate"]["prediction_superiority_proven"])
        self.assertEqual(
            report["status"],
            "historical_feature_ablation_not_independent_or_prospective_evidence",
        )


if __name__ == "__main__":
    unittest.main()
