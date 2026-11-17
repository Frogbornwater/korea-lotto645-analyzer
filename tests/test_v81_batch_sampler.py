# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v81_batch_sampler import (
    _allocate_rows,
    dependent_round_appearances,
    generate_batch_quota_portfolio,
)


def make_marginals(seed: int = 8101, dispersion: float = 0.9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, dispersion, 45)
    low, high = -20.0, 20.0
    for _ in range(100):
        middle = (low + high) / 2.0
        values = 1.0 / (1.0 + np.exp(-np.clip(logits + middle, -40.0, 40.0)))
        if float(values.sum()) < 6.0:
            low = middle
        else:
            high = middle
    probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(logits + (low + high) / 2.0, -40.0, 40.0))
    )
    return probabilities * (6.0 / float(probabilities.sum()))


class V81BatchQuotaPortfolioTests(unittest.TestCase):
    def test_portfolio_has_ten_unique_six_number_tickets_and_floor_ceil_counts(self):
        probabilities = make_marginals(seed=8102, dispersion=1.0)
        tickets, report = generate_batch_quota_portfolio(
            probabilities, ticket_count=10, sample_size=6, seed=8102028
        )
        self.assertEqual(len(tickets), 10)
        self.assertEqual(len({tuple(ticket) for ticket in tickets}), 10)
        self.assertTrue(all(len(ticket) == 6 and len(set(ticket)) == 6 for ticket in tickets))
        self.assertTrue(all(ticket == sorted(ticket) for ticket in tickets))
        self.assertEqual(sum(len(ticket) for ticket in tickets), 60)
        self.assertTrue(report["batch_counts_are_floor_or_ceil"])
        self.assertTrue(report["batch_expected_appearances_preserved"])
        self.assertFalse(report["per_ticket_marginals_identical"])
        self.assertEqual(report["unique_ticket_count"], 10)

    def test_each_number_count_is_floor_or_ceil_of_expected_batch_usage(self):
        probabilities = make_marginals(seed=8103, dispersion=1.15)
        counts = dependent_round_appearances(
            probabilities, ticket_count=10, sample_size=6, seed=8103001
        )
        expected = 10.0 * probabilities
        self.assertEqual(int(counts.sum()), 60)
        self.assertTrue(np.all(counts >= np.floor(expected)))
        self.assertTrue(np.all(counts <= np.ceil(expected)))
        self.assertLessEqual(float(np.max(np.abs(counts - expected))), 1.0 + 1e-10)

    def test_dependent_rounding_preserves_expected_counts_across_random_seeds(self):
        probabilities = make_marginals(seed=8104, dispersion=0.8)
        batches = np.vstack([
            dependent_round_appearances(
                probabilities, ticket_count=10, sample_size=6, seed=seed
            )
            for seed in range(800, 1600)
        ])
        empirical_mean = batches.mean(axis=0)
        expected_mean = 10.0 * probabilities
        self.assertLess(float(np.max(np.abs(empirical_mean - expected_mean))), 0.065)
        self.assertLess(float(np.mean(np.abs(empirical_mean - expected_mean))), 0.02)

    def test_portfolio_is_reproducible_and_reports_natural_overlap(self):
        probabilities = make_marginals(seed=8105)
        first, first_report = generate_batch_quota_portfolio(
            probabilities, ticket_count=10, seed=8105001
        )
        second, second_report = generate_batch_quota_portfolio(
            probabilities, ticket_count=10, seed=8105001
        )
        self.assertEqual(first, second)
        self.assertEqual(first_report["realized_appearances"], second_report["realized_appearances"])
        self.assertGreaterEqual(first_report["max_pairwise_overlap"], 0)
        self.assertLessEqual(first_report["max_pairwise_overlap"], 6)
        self.assertEqual(first_report["number_coverage"], len(set(n for t in first for n in t)))

    def test_refuses_a_batch_quota_that_cannot_support_ten_unique_tickets(self):
        counts = np.zeros(45, dtype=int)
        counts[:6] = 10
        with self.assertRaisesRegex(ValueError, "可形成的不同组合最多1组"):
            _allocate_rows(
                counts, ticket_count=10, sample_size=6,
                rng=np.random.default_rng(81),
            )

    def test_invalid_probability_vectors_are_rejected(self):
        with self.assertRaises(ValueError):
            dependent_round_appearances(np.full(45, 0.1), ticket_count=10)
        with self.assertRaises(ValueError):
            dependent_round_appearances(np.r_[0.0, np.full(44, 6.0 / 44.0)], ticket_count=10)


if __name__ == "__main__":
    unittest.main()
