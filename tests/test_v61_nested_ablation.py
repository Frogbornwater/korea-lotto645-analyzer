# -*- coding: utf-8 -*-
import unittest

import pandas as pd

from scripts.v61_nested_ablation import chronological_folds, summarize_variant


class TestV61NestedAblation(unittest.TestCase):
    def test_folds_are_chronological_and_non_overlapping(self):
        self.assertEqual(
            chronological_folds(),
            [(850, 899), (900, 949), (950, 999), (1000, 1049)],
        )

    def test_rejects_overlapping_or_unsorted_folds(self):
        with self.assertRaises(ValueError):
            chronological_folds((850, 890), 50)
        with self.assertRaises(ValueError):
            chronological_folds((900, 850), 50)

    def test_rejects_invalid_fold_size(self):
        with self.assertRaises(ValueError):
            chronological_folds((850,), 0)

    def test_summary_uses_paired_draw_level_differences(self):
        rows = []
        for fold, start in enumerate((850, 900, 950, 1000), 1):
            for draw in range(start, start + 50):
                rows.append({
                    "fold": fold,
                    "V5_mean_hits": 0.82,
                    "随机_mean_hits": 0.80,
                    "V5_batch_3plus": 0.20,
                    "随机_batch_3plus": 0.22,
                    "V5_Brier": 0.116,
                    "随机_Brier": 0.115,
                    "V5_LogLoss": 0.394,
                    "随机_LogLoss": 0.393,
                })
        result = summarize_variant(pd.DataFrame(rows), "test")
        self.assertEqual(result["test_draws"], 200)
        self.assertAlmostEqual(result["mean_hits_improvement_vs_random"], 0.02)
        self.assertAlmostEqual(result["batch_3plus_improvement_vs_random"], -0.02)
        self.assertAlmostEqual(result["Brier_improvement_vs_random"], -0.001)
        self.assertAlmostEqual(result["LogLoss_improvement_vs_random"], -0.001)


if __name__ == "__main__":
    unittest.main()
