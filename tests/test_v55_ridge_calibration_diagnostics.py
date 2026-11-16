import unittest

import numpy as np

from scripts.v55_ridge_calibration_diagnostics import probability_reliability
from v50_predictive_engine import fit_predict_next


def synthetic_history(n=90, seed=551):
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(1, n + 1):
        nums = sorted(rng.choice(np.arange(1, 46), size=6, replace=False).tolist())
        rows.append({
            "draw_no": draw, "date": "", "n1": nums[0], "n2": nums[1],
            "n3": nums[2], "n4": nums[3], "n5": nums[4], "n6": nums[5],
            "bonus": int(next(x for x in range(1, 46) if x not in nums)),
        })
    import pandas as pd
    return pd.DataFrame(rows)


class V55RidgeCalibrationDiagnosticsTests(unittest.TestCase):
    def test_reliability_bins_cover_all_observations(self):
        probabilities = np.linspace(0.05, 0.30, 100)
        outcomes = (np.arange(100) % 6 == 0).astype(float)
        result = probability_reliability(probabilities, outcomes, bins=10)
        self.assertEqual(result["observations"], 100)
        self.assertEqual(sum(row["count"] for row in result["bins"]), 100)
        self.assertEqual(len(result["bins"]), 10)
        self.assertGreaterEqual(result["equal_count_ECE"], 0.0)
        self.assertLessEqual(result["equal_count_ECE"], 1.0)

    def test_reliability_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            probability_reliability(np.array([0.1, 0.2]), np.array([0, 2]), bins=2)
        with self.assertRaises(ValueError):
            probability_reliability(np.array([0.1]), np.array([0]), bins=0)

    def test_ridge_mean_loss_equivalent_is_reported(self):
        result = fit_predict_next(
            synthetic_history(), training_window=50, min_history=40, ridge_alpha=1.0
        )
        diagnostics = result["diagnostics"]
        self.assertAlmostEqual(
            diagnostics["ridge_alpha_mean_loss_equivalent"],
            diagnostics["ridge_alpha"] / diagnostics["training_rows"],
        )
        self.assertIn("sum_log_loss_plus_alpha_over_2_l2", diagnostics["ridge_penalty_convention"])


if __name__ == "__main__":
    unittest.main()
