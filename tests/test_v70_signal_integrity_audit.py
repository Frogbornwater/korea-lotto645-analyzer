import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v70_signal_integrity_audit import (
    improvement_from_null, per_draw_metrics, ranking_hits,
)


class V70SignalIntegrityTests(unittest.TestCase):
    def test_ranking_hits(self):
        p = np.arange(1, 46, dtype=float)
        self.assertEqual(ranking_hits(p, [40, 41, 42, 43, 44, 45], 6), 6)
        self.assertEqual(ranking_hits(p, [1, 2, 3, 4, 5, 6], 6), 0)

    def test_improvement_direction(self):
        nulls = {"top6_hits": 0.8, "Brier": 0.1155, "LogLoss": 0.3927}
        got = improvement_from_null(
            {"top6_hits": 1.0, "Brier": 0.1100, "LogLoss": 0.4000}, nulls
        )
        self.assertAlmostEqual(got["top6_hits"], 0.2)
        self.assertAlmostEqual(got["Brier"], 0.0055)
        self.assertAlmostEqual(got["LogLoss"], -0.0073)

    def test_metrics_reject_bad_probability_vector(self):
        import json
        row = {
            "V5_45号码概率_JSON": json.dumps([1.0 / 45.0] * 45),
            "实际主号码": json.dumps([1, 2, 3, 4, 5, 6]),
            "V5_Brier": 0.1,
            "V5_LogLoss": 0.3,
        }
        result = per_draw_metrics(row)
        self.assertEqual(result["top6_hits"], 0.0)
        row["V5_45号码概率_JSON"] = json.dumps([0.2] * 45)
        with self.assertRaises(ValueError):
            per_draw_metrics(row)


if __name__ == "__main__":
    unittest.main()
