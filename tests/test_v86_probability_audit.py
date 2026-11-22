# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v86_probability_audit import (
    _select_temperature,
    _temperature_transform,
    run_v86_probability_audit,
)


def synthetic_history(n: int = 100, seed: int = 8601) -> pd.DataFrame:
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


class V86ProbabilityAuditTests(unittest.TestCase):
    def test_temperature_scaling_preserves_rank_and_six_probability_mass(self):
        rng = np.random.default_rng(8602)
        scores = rng.normal(0.0, 0.4, 45)
        p = 1.0 / (1.0 + np.exp(-scores))
        # Normalize with the same intercept constraint used by the model.
        from v50_predictive_engine import _calibrate_to_six
        p = _calibrate_to_six(scores)
        for temperature in (0.5, 0.8, 1.0, 1.5, 2.0, 3.0):
            adjusted = _temperature_transform(p, temperature)
            self.assertAlmostEqual(float(adjusted.sum()), 6.0, places=8)
            self.assertEqual(np.argsort(-adjusted).tolist(), np.argsort(-p).tolist())
        self.assertTrue(np.allclose(_temperature_transform(p, 1.0), p, atol=1e-8))

    def test_temperature_selection_is_deterministic_and_uses_paired_rows(self):
        rng = np.random.default_rng(8603)
        probs, labels = [], []
        for _ in range(20):
            p = rng.uniform(0.08, 0.19, 45)
            p *= 6.0 / p.sum()
            y = np.zeros(45)
            y[rng.choice(45, 6, replace=False)] = 1.0
            probs.append(p)
            labels.append(y)
        first = _select_temperature(probs, labels)
        second = _select_temperature(probs, labels)
        self.assertEqual(first, second)
        self.assertIn(first[0], (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0))

    def test_small_prequential_audit_excludes_warmup_from_scored_rows(self):
        history = synthetic_history(100)
        result = run_v86_probability_audit(
            history, test_draws=4, warmup_draws=10, calibration_window=10,
            training_window=50, min_history=30, bootstrap_repetitions=200,
            block_length=2, seed=8604,
        )
        report = result["report"]
        detail = result["detail"]
        self.assertEqual(report["target_start_draw"], 97)
        self.assertEqual(report["target_end_draw"], 100)
        self.assertEqual(report["warmup_forecasts"], 10)
        self.assertEqual(len(detail), 4)
        self.assertEqual(len(report["primary_comparisons"]), 5)
        self.assertEqual(sum(report["selected_temperature_counts"].values()), 4)
        self.assertTrue(report["evidence_gate"]["calibration_is_strictly_prequential"])
        self.assertFalse(report["evidence_gate"]["predictive_superiority_proven"])
        self.assertFalse(report["evidence_gate"]["production_model_change_allowed"])
        self.assertEqual(len(result["reliability"]), 20)

    def test_rejects_insufficient_warmup(self):
        history = synthetic_history(100)
        with self.assertRaises(ValueError):
            run_v86_probability_audit(
                history, test_draws=4, warmup_draws=5, calibration_window=10,
                training_window=50, min_history=30,
            )


if __name__ == "__main__":
    unittest.main()
