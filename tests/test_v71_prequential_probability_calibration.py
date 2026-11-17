import unittest
import numpy as np
import pandas as pd

from scripts.v71_prequential_probability_calibration import (
    UNIFORM, calibrate_probabilities, brier_vector, choose_shrinkage, run_audit,
)


class V71CalibrationTests(unittest.TestCase):
    def test_shrinkage_preserves_sum_and_bounds(self):
        p = np.full(45, UNIFORM)
        p[0] += 0.1
        p[1:] -= 0.1 / 44
        q = calibrate_probabilities(p, 0.4)
        self.assertAlmostEqual(float(q.sum()), 6.0)
        self.assertTrue(np.all((q >= 0) & (q <= 1)))
        self.assertTrue(np.allclose(q, UNIFORM + 0.4 * (p - UNIFORM)))

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_probabilities(np.full(45, 0.2), 0.5)

    def test_choose_lambda_uses_prior_rows(self):
        rows = []
        p = np.full(45, UNIFORM)
        p[0] += 0.08
        p[1:] -= 0.08 / 44
        for i in range(5):
            rows.append({"V5_45号码概率_JSON": __import__("json").dumps(p.tolist()),
                         "实际主号码": __import__("json").dumps([1, 2, 3, 4, 5, 6])})
        lam, info = choose_shrinkage(pd.DataFrame(rows))
        self.assertEqual(info["training_draws"], 5)
        self.assertIn(lam, np.linspace(0, 1, 21))

    def test_audit_rejects_wrong_target_range(self):
        with self.assertRaises(ValueError):
            run_audit(pd.DataFrame(columns=["target_draw", "fold", "V5_45号码概率_JSON", "实际主号码"]))


if __name__ == "__main__":
    unittest.main()
