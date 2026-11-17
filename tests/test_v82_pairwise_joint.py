# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v82_pairwise_joint import (
    PAIR_COUNT,
    fit_pair_residuals,
    forecast_pair_probabilities,
    pair_baseline_from_marginals,
    permute_pair_labels,
    score_pair_probabilities,
    evaluate_v82_pairwise_research,
)


def make_history(n: int = 160, seed: int = 8201) -> pd.DataFrame:
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


class V82PairwiseJointTests(unittest.TestCase):
    def test_pair_baseline_satisfies_exact_first_order_degree_constraints(self):
        probabilities = np.full(45, 6.0 / 45.0)
        pair = pair_baseline_from_marginals(probabilities)
        self.assertEqual(pair.shape, (45, 45))
        self.assertTrue(np.allclose(pair, pair.T))
        self.assertTrue(np.allclose(np.diag(pair), 0.0))
        self.assertTrue(np.allclose(pair.sum(axis=1), 5.0 * probabilities, atol=1e-9))
        self.assertAlmostEqual(float(pair.sum() / 2.0), 15.0, places=8)
        self.assertTrue(np.all(pair <= np.minimum.outer(probabilities, probabilities) + 1e-9))

    def test_history_pair_residuals_are_symmetric_and_preserve_row_sums(self):
        history = make_history(220, seed=8202)
        fit = fit_pair_residuals(history, prior_draws=1000.0)
        self.assertEqual(fit.residual_log_odds.shape, (45, 45))
        self.assertTrue(np.allclose(fit.residual_log_odds, fit.residual_log_odds.T))
        self.assertTrue(np.allclose(np.diag(fit.residual_log_odds), 0.0))
        self.assertTrue(np.isfinite(fit.residual_log_odds).all())
        self.assertTrue(np.allclose(
            fit.posterior_pair_probabilities.sum(axis=1),
            5.0 * fit.historical_marginals,
            atol=1e-8,
        ))

    def test_zero_strength_returns_exact_first_order_baseline(self):
        probabilities = np.random.default_rng(8203).dirichlet(np.ones(45) * 50) * 6.0
        residual = np.zeros((45, 45), dtype=float)
        adjusted, baseline, diagnostics = forecast_pair_probabilities(
            probabilities, residual, strength=0.0
        )
        self.assertTrue(np.allclose(adjusted, baseline, atol=1e-12))
        self.assertLess(diagnostics["max_row_sum_error"], 1e-12)

    def test_pair_adjustment_preserves_all_first_order_degree_constraints(self):
        probabilities = np.random.default_rng(8204).dirichlet(np.ones(45) * 35) * 6.0
        rng = np.random.default_rng(8205)
        residual = rng.normal(0.0, 0.15, size=(45, 45))
        residual = (residual + residual.T) / 2.0
        np.fill_diagonal(residual, 0.0)
        adjusted, baseline, diagnostics = forecast_pair_probabilities(
            probabilities, residual, strength=0.5
        )
        self.assertLess(diagnostics["max_row_sum_error"], 1e-7)
        self.assertTrue(np.allclose(adjusted, adjusted.T, atol=1e-10))
        self.assertTrue(np.allclose(adjusted.sum(axis=1), 5.0 * probabilities, atol=1e-7))
        self.assertAlmostEqual(float(adjusted.sum() / 2.0), 15.0, places=6)
        self.assertTrue(np.all(adjusted >= 0.0))
        self.assertTrue(np.all(adjusted <= np.minimum.outer(probabilities, probabilities) + 1e-9))
        self.assertGreater(float(np.mean(np.abs(adjusted - baseline))), 0.0)

    def test_placebo_is_reproducible_and_keeps_pair_matrix_symmetric(self):
        history = make_history(180, seed=8206)
        fit = fit_pair_residuals(history)
        a, perm_a = permute_pair_labels(fit, seed=8207)
        b, perm_b = permute_pair_labels(fit, seed=8207)
        self.assertTrue(np.array_equal(perm_a, perm_b))
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.allclose(a, a.T))
        self.assertTrue(np.allclose(np.diag(a), 0.0))

    def test_pair_scoring_counts_exactly_fifteen_observed_pairs(self):
        probabilities = np.full(45, 6.0 / 45.0)
        pair = pair_baseline_from_marginals(probabilities)
        score = score_pair_probabilities(pair, [1, 2, 3, 4, 5, 6])
        self.assertEqual(score["observed_pairs"], 15)
        self.assertEqual(PAIR_COUNT, 990)
        self.assertAlmostEqual(score["forecast_pair_sum"], 15.0, places=8)
        self.assertGreater(score["pair_brier"], 0.0)
        self.assertGreater(score["pair_log_loss"], 0.0)

    def test_rolling_research_is_prequential_and_keeps_production_gate_closed(self):
        history = make_history(110, seed=8208)
        report = evaluate_v82_pairwise_research(
            history,
            test_draws=12,
            training_window=50,
            min_history=50,
            bootstrap_repetitions=200,
            block_length=3,
            seed=8209,
        )
        self.assertEqual(report["test_draws"], 12)
        self.assertEqual(report["target_start_draw"], 99)
        self.assertEqual(report["target_end_draw"], 110)
        self.assertEqual(report["pair_feature_exclusions"], list(("pair_mean_log_lift", "pair_sd_log_lift"))) if "pair_feature_exclusions" in report else None
        self.assertFalse(report["evidence_gate"]["production_model_change_allowed"])
        self.assertFalse(report["evidence_gate"]["prediction_superiority_proven"])
        self.assertEqual(report["status"], "historical_pairwise_screen_not_independent_or_prospective_evidence")


if __name__ == "__main__":
    unittest.main()
