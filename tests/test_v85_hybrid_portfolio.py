# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v81_batch_sampler import dependent_round_appearances
from v85_hybrid_portfolio import generate_probability_weighted_portfolio, V85_VERSION
from v85_hybrid_backtest import run_v85_walk_forward


def make_probabilities(seed: int = 8501, dispersion: float = 0.18) -> np.ndarray:
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
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits + (low + high) / 2.0, -40.0, 40.0)))
    return probs * (6.0 / float(probs.sum()))


def synthetic_history(n: int = 100, seed: int = 8502) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(1, n + 1):
        numbers = sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False))
        rows.append({
            "draw_no": draw, "date": "",
            "n1": numbers[0], "n2": numbers[1], "n3": numbers[2],
            "n4": numbers[3], "n5": numbers[4], "n6": numbers[5],
            "bonus": next(x for x in range(1, 46) if x not in numbers),
        })
    return pd.DataFrame(rows)


class V85HybridPortfolioTests(unittest.TestCase):
    def test_graph17_allocation_preserves_identical_v81_quota(self):
        probabilities = make_probabilities()
        seed = 8503
        tickets, report = generate_probability_weighted_portfolio(probabilities, seed=seed)
        expected_counts = dependent_round_appearances(
            probabilities, ticket_count=10, sample_size=6, seed=seed
        )
        self.assertEqual(len(tickets), 10)
        self.assertEqual(report["sampler_version"], V85_VERSION)
        self.assertEqual(report["realized_appearances"], expected_counts.tolist())
        self.assertTrue(report["batch_counts_are_floor_or_ceil"])
        self.assertEqual(report["number_coverage"], 45)
        self.assertEqual(report["max_pairwise_overlap"], 1)
        self.assertEqual(report["pairwise_overlap_histogram"]["1"], 15)
        self.assertEqual(report["pairwise_overlap_histogram"]["2"], 0)

    def test_generation_is_reproducible(self):
        probabilities = make_probabilities(seed=8504)
        first, first_report = generate_probability_weighted_portfolio(probabilities, seed=8505)
        second, second_report = generate_probability_weighted_portfolio(probabilities, seed=8505)
        self.assertEqual(first, second)
        self.assertEqual(first_report["realized_appearances"], second_report["realized_appearances"])

    def test_generic_incidence_path_handles_zero_one_two_quotas(self):
        probabilities = np.full(45, 6.0 / 45.0)
        # 18 numbers appear twice, 24 once, and 3 not at all; total is 60.
        counts = np.asarray([2] * 18 + [1] * 24 + [0] * 3, dtype=int)
        from v85_hybrid_portfolio import _generic_incidence_portfolio, _portfolio_metrics
        tickets = _generic_incidence_portfolio(counts, probabilities, seed=8506)
        realized = np.zeros(45, dtype=int)
        for ticket in tickets:
            realized[np.asarray(ticket, dtype=int) - 1] += 1
        self.assertEqual(realized.tolist(), counts.tolist())
        metrics = _portfolio_metrics(tickets)
        self.assertEqual(metrics["max_pairwise_overlap"], 1)
        self.assertEqual(metrics["number_coverage"], 42)

    def test_small_prequential_run_compares_all_five_methods(self):
        history = synthetic_history(100)
        result = run_v85_walk_forward(
            history, test_draws=4, training_window=50, min_history=30,
            bootstrap_repetitions=200, block_length=2, seed=8507,
        )
        report = result["report"]
        detail = result["detail"]
        self.assertEqual(report["target_start_draw"], 97)
        self.assertEqual(report["target_end_draw"], 100)
        self.assertEqual(len(detail), 4)
        self.assertEqual(len(report["primary_comparisons"]), 4)
        self.assertTrue(report["structure_compliance"]["all_v85_batches_have_pairwise_overlap_at_most_one"])
        self.assertFalse(report["evidence_gate"]["predictive_superiority_proven"])
        self.assertFalse(report["evidence_gate"]["production_model_change_allowed"])


if __name__ == "__main__":
    unittest.main()
