import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v64_exact_null_holm_audit import exact_null_baselines, holm_adjust, sign_flip_pvalue


class V64ExactNullHolmTests(unittest.TestCase):
    def test_exact_uniform_expectations(self):
        values = exact_null_baselines()
        self.assertAlmostEqual(values["top6_hits"], 0.8)
        self.assertAlmostEqual(values["top10_hits"], 4.0 / 3.0)
        self.assertAlmostEqual(values["top15_hits"], 2.0)
        self.assertAlmostEqual(values["ndcg_at_6"], 6.0 / 45.0)
        self.assertAlmostEqual(values["Brier"], 0.11555555555555556)
        self.assertAlmostEqual(values["LogLoss"], 0.3926744672275521)

    def test_holm_adjustment_is_monotone_and_bounded(self):
        result = holm_adjust({"a": 0.001, "b": 0.02, "c": 0.5})
        self.assertAlmostEqual(result["a"], 0.003)
        self.assertAlmostEqual(result["b"], 0.04)
        self.assertAlmostEqual(result["c"], 0.5)
        self.assertTrue(all(0 <= p <= 1 for p in result.values()))

    def test_sign_flip_detects_large_consistent_effect(self):
        self.assertLess(sign_flip_pvalue([1.0] * 20, repetitions=2000, seed=4), 0.01)

    def test_sign_flip_accepts_zero_effect(self):
        self.assertAlmostEqual(sign_flip_pvalue([0.0] * 20, repetitions=2000, seed=4), 1.0)


if __name__ == "__main__":
    unittest.main()
