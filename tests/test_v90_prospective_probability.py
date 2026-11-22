# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v90_prospective_probability import (
    UNIFORM_BRIER, UNIFORM_LOGLOSS, analyze_v90_probability,
)


def record(draw, *, brier=0.11, logloss=0.48, top6=1, top10=2, top15=3):
    return {
        "target_draw": draw,
        "score": {
            "v5_probability_metrics": {
                "brier": brier,
                "logloss": logloss,
                "top6_hits": top6,
                "top10_hits": top10,
                "top15_hits": top15,
            }
        }
    }


class V90ProspectiveProbabilityTests(unittest.TestCase):
    def test_uniform_reference_values_are_finite_and_correct(self):
        self.assertGreater(UNIFORM_BRIER, 0)
        self.assertGreater(UNIFORM_LOGLOSS, 0)

    def test_aggregates_probability_and_top_k_metrics(self):
        rows = [record(i, brier=0.09, logloss=0.4, top6=2, top10=3, top15=4) for i in range(10, 14)]
        report = analyze_v90_probability(rows, bootstrap_reps=1000)
        self.assertEqual(report["scored_draws"], 4)
        self.assertAlmostEqual(report["probability_metrics"]["brier"]["mean"], 0.09)
        self.assertAlmostEqual(report["ranking_metrics"]["top_k_hits"]["6"]["mean_hits"], 2.0)
        self.assertEqual(len(report["probability_metrics"]["brier"]["difference_bootstrap_95ci"]), 2)

    def test_rejects_missing_probability_metrics(self):
        with self.assertRaisesRegex(ValueError, "缺少 v5_probability_metrics"):
            analyze_v90_probability([{"target_draw": 1, "score": {}}], bootstrap_reps=1000)

    def test_rejects_duplicate_or_unsorted_draws(self):
        with self.assertRaisesRegex(ValueError, "唯一且按升序"):
            analyze_v90_probability([record(2), record(2)], bootstrap_reps=1000)
        with self.assertRaisesRegex(ValueError, "唯一且按升序"):
            analyze_v90_probability([record(3), record(2)], bootstrap_reps=1000)

    def test_rejects_nonfinite_metrics(self):
        with self.assertRaisesRegex(ValueError, "无效"):
            analyze_v90_probability([record(1, brier=float("nan"))], bootstrap_reps=1000)


if __name__ == "__main__":
    unittest.main()
