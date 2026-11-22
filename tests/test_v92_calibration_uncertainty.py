# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v91_probability_vector import _canonical_vector_hash
from v92_calibration_uncertainty import analyze_v92_uncertainty


def make_record(draw, p=None, actual=None):
    p = p if p is not None else [6/45] * 45
    actual = actual if actual is not None else [1, 2, 3, 4, 5, 6]
    return {
        "target_draw": draw,
        "score": {
            "actual_main_numbers": actual,
            "probability_forecast": {
                "source": "immutable_pre_draw_lock_bundle",
                "vector_length": 45,
                "marginal_inclusion_probabilities": p,
                "vector_sha256": _canonical_vector_hash(p),
            },
        },
    }


class V92CalibrationUncertaintyTests(unittest.TestCase):
    def test_uniform_forecast_has_zero_calibration_gap_and_ece(self):
        result = analyze_v92_uncertainty([make_record(1)], bootstrap_reps=1000)
        self.assertAlmostEqual(result["metrics"]["calibration_in_the_large_gap_observed_minus_predicted"], 0.0, places=12)
        self.assertAlmostEqual(result["metrics"]["expected_calibration_error_equal_width"], 0.0, places=12)
        self.assertFalse(result["calibration_uncertainty_ready"])
        self.assertEqual(result["metrics"]["ece_95ci"], [0.0, 0.0])

    def test_draw_cluster_bootstrap_is_deterministic(self):
        rows = [make_record(i) for i in range(1, 5)]
        a = analyze_v92_uncertainty(rows, bootstrap_reps=1000)
        b = analyze_v92_uncertainty(rows, bootstrap_reps=1000)
        self.assertEqual(a["metrics"]["ece_95ci"], b["metrics"]["ece_95ci"])
        self.assertEqual(a["metrics"]["calibration_in_the_large_gap_95ci"], b["metrics"]["calibration_in_the_large_gap_95ci"])

    def test_rejects_duplicate_or_unsorted_draws(self):
        with self.assertRaisesRegex(ValueError, "唯一且按升序"):
            analyze_v92_uncertainty([make_record(2), make_record(1)], bootstrap_reps=1000)

    def test_rejects_empty_and_too_small_bootstrap(self):
        with self.assertRaises(ValueError):
            analyze_v92_uncertainty([], bootstrap_reps=1000)
        with self.assertRaisesRegex(ValueError, "至少1000"):
            analyze_v92_uncertainty([make_record(1)], bootstrap_reps=999)


if __name__ == "__main__":
    unittest.main()
