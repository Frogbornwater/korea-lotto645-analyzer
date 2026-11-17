# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd

from scripts.v75_model_family_research import (
    CONFIGS, CONFIG_BY_KEY, V4_BASELINE_KEY, V5_BASELINE_KEY, UNIFORM_KEY,
    frequency_probabilities, rank_configurations, uniform_random_expectations,
    moving_block_ci,
)


class V75ModelFamilyResearchTests(unittest.TestCase):
    def test_predeclared_grid_has_31_unique_configs_and_fixed_controls(self):
        self.assertEqual(len(CONFIGS), 31)
        self.assertEqual(len(CONFIG_BY_KEY), 31)
        self.assertIn(V4_BASELINE_KEY, CONFIG_BY_KEY)
        self.assertIn(V5_BASELINE_KEY, CONFIG_BY_KEY)
        self.assertIn(UNIFORM_KEY, CONFIG_BY_KEY)

    def test_rolling_and_exponential_frequency_probabilities_sum_to_six(self):
        rng = np.random.default_rng(2026)
        matrix = np.vstack([rng.choice(np.arange(1, 46), size=6, replace=False) for _ in range(300)])
        rolling = frequency_probabilities(matrix, 250, window=100, prior=80)
        ewma = frequency_probabilities(matrix, 250, half_life=60, prior=175)
        self.assertEqual(rolling.shape, (45,))
        self.assertEqual(ewma.shape, (45,))
        self.assertTrue(np.all(rolling > 0))
        self.assertTrue(np.all(ewma > 0))
        self.assertAlmostEqual(float(rolling.sum()), 6.0, places=9)
        self.assertAlmostEqual(float(ewma.sum()), 6.0, places=9)

    def test_frequency_function_requires_exactly_one_window_type(self):
        matrix = np.ones((100, 6), dtype=np.int16)
        with self.assertRaises(ValueError):
            frequency_probabilities(matrix, 90, prior=80)
        with self.assertRaises(ValueError):
            frequency_probabilities(matrix, 90, window=50, half_life=30, prior=80)

    def test_prior_only_selector_can_switch_model_families(self):
        rows = []
        for key, top10, brier, logloss, hits in (
            (UNIFORM_KEY, 1.33, .11555, .39267, .80),
            (V5_BASELINE_KEY, 1.30, .11570, .39310, .79),
            (V4_BASELINE_KEY, 1.36, .11560, .39270, .81),
        ):
            for _ in range(25):
                rows.append({
                    "config_key": key,
                    "top10_hits": top10,
                    "Brier": brier,
                    "LogLoss": logloss,
                    "mean_hits": hits,
                })
        chosen, ranking = rank_configurations(pd.DataFrame(rows))
        self.assertEqual(chosen["key"], V4_BASELINE_KEY)
        self.assertGreaterEqual(len(ranking), 3)

    def test_uniform_random_baselines_are_exact(self):
        baseline = uniform_random_expectations(10)
        self.assertAlmostEqual(baseline["top10_hits"], 4.0 / 3.0, places=12)
        self.assertAlmostEqual(baseline["mean_hits"], 0.8, places=12)
        self.assertAlmostEqual(baseline["ticket_3plus"], 0.023834078570323606, places=12)
        self.assertAlmostEqual(baseline["batch_3plus"], 0.21433688902346237, places=12)
        self.assertAlmostEqual(baseline["ticket_4plus"], 0.0013934826753885176, places=12)
        self.assertAlmostEqual(baseline["batch_4plus"], 0.013847777540374473, places=12)

    def test_bonferroni_interval_is_wider_than_nominal_interval(self):
        values = pd.Series([0.0, 1.0, -1.0, 0.5] * 30).to_numpy()
        nominal = moving_block_ci(values, 91, alpha=0.05)
        adjusted = moving_block_ci(values, 91, alpha=0.005)
        self.assertLessEqual(adjusted[0], nominal[0])
        self.assertGreaterEqual(adjusted[1], nominal[1])


if __name__ == "__main__":
    unittest.main()
