import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v63_probability_ranking_audit import (
    bootstrap_ci,
    chronological_folds,
    rank_metrics,
)


class V63ProbabilityRankingAuditTests(unittest.TestCase):
    def test_folds_are_chronological_and_disjoint(self):
        self.assertEqual(
            chronological_folds(),
            [(850, 899), (900, 949), (950, 999), (1000, 1049)],
        )

    def test_top_k_counts_and_ndcg_are_bounded(self):
        probabilities = np.zeros(45, dtype=float)
        probabilities[:6] = np.linspace(0.9, 0.4, 6)
        actual = [1, 2, 3, 4, 5, 6]
        random_order = np.arange(1, 46)
        metrics = rank_metrics(probabilities, actual, random_order)
        self.assertEqual(metrics["top6_hits"], 6.0)
        self.assertEqual(metrics["top10_hits"], 6.0)
        self.assertEqual(metrics["top15_hits"], 6.0)
        self.assertAlmostEqual(metrics["ndcg_at_6"], 1.0)
        self.assertTrue(0.0 <= metrics["random_ndcg_at_6"] <= 1.0)

    def test_invalid_actual_numbers_rejected(self):
        with self.assertRaises(ValueError):
            rank_metrics(np.ones(45), [1, 2, 3, 4, 5, 5], np.arange(1, 46))

    def test_invalid_random_permutation_rejected(self):
        with self.assertRaises(ValueError):
            rank_metrics(np.ones(45), [1, 2, 3, 4, 5, 6], np.ones(45))

    def test_bootstrap_ci_contains_constant_value(self):
        low, high = bootstrap_ci(np.full(20, 0.25), repetitions=500, seed=3)
        self.assertAlmostEqual(low, 0.25)
        self.assertAlmostEqual(high, 0.25)


if __name__ == "__main__":
    unittest.main()
