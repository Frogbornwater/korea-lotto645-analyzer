# -*- coding: utf-8 -*-
import unittest

import pandas as pd

from scripts.v74_recent1000_research import (
    BASELINE, CONFIGS, TARGET_DRAWS, BLOCK_SIZE, MIN_TRAINING_DRAWS, _rank_configurations, config_key, moving_block_ci, uniform_random_expectations,
)


class V74Recent1000ResearchTests(unittest.TestCase):
    def test_research_expands_to_800_targets_in_100_draw_blocks(self):
        self.assertEqual(TARGET_DRAWS, 1000)
        self.assertEqual(MIN_TRAINING_DRAWS, 60)
        self.assertEqual(BLOCK_SIZE, 100)

    def test_grid_is_predeclared_and_includes_baseline(self):
        self.assertEqual(len(CONFIGS), 24)
        self.assertIn(config_key(BASELINE), {config_key(x) for x in CONFIGS})

    def test_first_block_fallback_is_baseline_rankable(self):
        chosen, ranking = _rank_configurations(pd.DataFrame())
        self.assertEqual(chosen, BASELINE)
        self.assertEqual(ranking, [])

    def test_prior_only_ranker_uses_prior_rows(self):
        rows = []
        for key, top10, brier, logloss, hits in (
            ("w300_a0.3_p0", 1.4, .116, .394, .81),
            ("w500_a1_p0.5", 1.3, .115, .393, .82),
        ):
            for _ in range(20):
                rows.append({"config_key": key, "top10_hits": top10, "Brier": brier,
                             "LogLoss": logloss, "mean_hits": hits})
        chosen, ranking = _rank_configurations(pd.DataFrame(rows))
        self.assertEqual(config_key(chosen), "w500_a1_p0.5")
        self.assertEqual(len(ranking), 2)

    def test_uniform_baselines_are_exact_6_of_45_expectations(self):
        baseline = uniform_random_expectations(10)
        self.assertAlmostEqual(baseline["top6_hits"], 0.8, places=12)
        self.assertAlmostEqual(baseline["top10_hits"], 4.0 / 3.0, places=12)
        self.assertAlmostEqual(baseline["top15_hits"], 2.0, places=12)
        self.assertAlmostEqual(baseline["mean_hits"], 0.8, places=12)
        self.assertAlmostEqual(baseline["ticket_3plus"], 0.023834078570323606, places=12)
        self.assertAlmostEqual(baseline["batch_3plus"], 0.21433688902346237, places=12)
        self.assertAlmostEqual(baseline["ticket_4plus"], 0.0013934826753885176, places=12)
        self.assertAlmostEqual(baseline["batch_4plus"], 0.013847777540374473, places=12)

    def test_block_bootstrap_is_reproducible(self):
        values = pd.Series([0.0, 1.0, -1.0, 0.5] * 20).to_numpy()
        self.assertEqual(moving_block_ci(values, 77), moving_block_ci(values, 77))

    def test_bonferroni_interval_is_wider_than_nominal_interval(self):
        values = pd.Series([0.0, 1.0, -1.0, 0.5] * 30).to_numpy()
        nominal = moving_block_ci(values, 91, alpha=0.05)
        adjusted = moving_block_ci(values, 91, alpha=0.005)
        self.assertLessEqual(adjusted[0], nominal[0])
        self.assertGreaterEqual(adjusted[1], nominal[1])


if __name__ == "__main__":
    unittest.main()
