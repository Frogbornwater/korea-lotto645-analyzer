# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v80_probability_sampling import (
    fit_conditional_bernoulli,
    generate_probability_faithful_tickets,
    sample_conditional_bernoulli,
    summarize_ticket_batch,
)


def make_marginals(seed: int = 803, dispersion: float = 0.9) -> np.ndarray:
    """Build valid, nonuniform 45-number marginals summing to six."""
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
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits + (low + high) / 2.0, -40.0, 40.0)))
    return probabilities * (6.0 / float(probabilities.sum()))


class V80ProbabilitySamplingTests(unittest.TestCase):
    def test_uniform_marginals_produce_equal_odds(self):
        probabilities = np.full(45, 6.0 / 45.0)
        fitted = fit_conditional_bernoulli(probabilities)
        self.assertLess(fitted.max_abs_error, 1e-8)
        self.assertLess(float(np.ptp(fitted.log_odds)), 1e-8)
        tickets = sample_conditional_bernoulli(fitted, count=12, seed=80)
        self.assertEqual(len(tickets), 12)
        self.assertTrue(all(len(ticket) == 6 and len(set(ticket)) == 6 for ticket in tickets))
        self.assertTrue(all(ticket == sorted(ticket) for ticket in tickets))

    def test_nonuniform_requested_marginals_are_reproduced_analytically(self):
        probabilities = make_marginals(seed=2028, dispersion=1.0)
        fitted = fit_conditional_bernoulli(probabilities)
        self.assertAlmostEqual(float(fitted.target_marginals.sum()), 6.0, places=8)
        self.assertLess(fitted.max_abs_error, 1e-7)
        self.assertTrue(np.allclose(fitted.fitted_marginals, fitted.target_marginals, atol=1e-7))
        self.assertGreater(float(np.ptp(fitted.log_odds)), 0.1)

    def test_monte_carlo_marginals_track_requested_probabilities(self):
        probabilities = make_marginals(seed=904, dispersion=0.85)
        fitted = fit_conditional_bernoulli(probabilities)
        tickets = sample_conditional_bernoulli(fitted, count=6000, seed=5904)
        counts = np.zeros(45, dtype=float)
        for ticket in tickets:
            counts[np.asarray(ticket, dtype=int) - 1] += 1.0
        empirical = counts / float(len(tickets))
        # Six thousand fixed-size draws should be close to the analytically
        # calibrated marginals; this tolerance is deliberately Monte-Carlo aware.
        self.assertLess(float(np.max(np.abs(empirical - probabilities))), 0.035)
        self.assertLess(float(np.mean(np.abs(empirical - probabilities))), 0.008)
        self.assertAlmostEqual(float(empirical.sum()), 6.0, places=10)

    def test_sampling_is_reproducible_and_always_returns_six_distinct_numbers(self):
        probabilities = make_marginals(seed=77)
        fitted = fit_conditional_bernoulli(probabilities)
        first = sample_conditional_bernoulli(fitted, count=20, seed=123456)
        second = sample_conditional_bernoulli(fitted, count=20, seed=123456)
        self.assertEqual(first, second)
        self.assertTrue(all(len(ticket) == 6 and len(set(ticket)) == 6 for ticket in first))
        self.assertTrue(all(1 <= number <= 45 for ticket in first for number in ticket))

    def test_batch_report_exposes_portfolio_sampling_deviation(self):
        probabilities = np.full(45, 6.0 / 45.0)
        tickets = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        report = summarize_ticket_batch(tickets, probabilities)
        self.assertEqual(report["ticket_count"], 2)
        self.assertEqual(report["sample_size"], 6)
        self.assertAlmostEqual(report["realized_frequency_sum"], 6.0)
        self.assertGreater(report["max_absolute_marginal_deviation"], 0.0)
        self.assertIn("小批量组合", report["interpretation"])

    def test_generate_wrapper_returns_fit_and_batch_diagnostics(self):
        probabilities = make_marginals(seed=505)
        tickets, fitted, report = generate_probability_faithful_tickets(
            probabilities, count=10, seed=5050
        )
        self.assertEqual(len(tickets), 10)
        self.assertLess(fitted.max_abs_error, 1e-7)
        self.assertEqual(report["ticket_count"], 10)
        self.assertEqual(len(report["realized_marginals"]), 45)

    def test_invalid_probability_vectors_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_conditional_bernoulli(np.full(45, 0.1))  # sum is not six
        with self.assertRaises(ValueError):
            fit_conditional_bernoulli(np.r_[np.zeros(1), np.full(44, 6.0 / 44.0)])
        with self.assertRaises(ValueError):
            fit_conditional_bernoulli(np.full((5, 9), 6.0 / 45.0))


if __name__ == "__main__":
    unittest.main()
