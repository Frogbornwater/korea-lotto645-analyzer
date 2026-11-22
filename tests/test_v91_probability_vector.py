# -*- coding: utf-8 -*-
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v91_probability_vector import analyze_v91_calibration, _canonical_vector_hash


def record(draw=1):
    p = [6/45] * 45
    actual = [1, 2, 3, 4, 5, 6]
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


class V91ProbabilityVectorTests(unittest.TestCase):
    def test_uniform_vector_scores_match_uniform_reference(self):
        result = analyze_v91_calibration([record()], bootstrap_reps=1000)
        self.assertAlmostEqual(result["metrics"]["brier_difference_vs_uniform"], 0.0, places=12)
        self.assertAlmostEqual(result["metrics"]["logloss_difference_vs_uniform"], 0.0, places=12)
        self.assertEqual(result["probability_vector_count"], 45)
        self.assertEqual(sum(x["count"] for x in result["reliability_bins"]), 45)

    def test_rejects_bad_vector_hash(self):
        row = record()
        row["score"]["probability_forecast"]["vector_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "哈希"):
            analyze_v91_calibration([row], bootstrap_reps=1000)

    def test_rejects_wrong_probability_sum(self):
        row = record()
        p = row["score"]["probability_forecast"]["marginal_inclusion_probabilities"]
        p[0] += .1
        row["score"]["probability_forecast"]["vector_sha256"] = _canonical_vector_hash(p)
        with self.assertRaisesRegex(ValueError, "边际概率约束"):
            analyze_v91_calibration([row], bootstrap_reps=1000)

    def test_rejects_duplicate_or_unsorted_draws(self):
        with self.assertRaisesRegex(ValueError, "唯一且按升序"):
            analyze_v91_calibration([record(2), record(1)], bootstrap_reps=1000)

    def test_empty_and_small_bootstrap_rejected(self):
        with self.assertRaises(ValueError):
            analyze_v91_calibration([], bootstrap_reps=1000)
        with self.assertRaises(ValueError):
            analyze_v91_calibration([record()], bootstrap_reps=999)


if __name__ == "__main__":
    unittest.main()
