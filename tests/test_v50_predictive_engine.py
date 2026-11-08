import unittest

import numpy as np
import pandas as pd

from v50_predictive_engine import (
    BASE_RATE, _calibrate_to_six, build_number_features,
    fit_predict_next, generate_tickets, build_v50_forward_lock,
)
from v50_backtest import run_v50_walk_forward


def synthetic_history(n=100, seed=71):
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(1, n + 1):
        nums = sorted(rng.choice(np.arange(1, 46), size=6, replace=False).tolist())
        rows.append({
            "draw_no": draw, "date": "", "n1": nums[0], "n2": nums[1],
            "n3": nums[2], "n4": nums[3], "n5": nums[4], "n6": nums[5],
            "bonus": int(next(x for x in range(1, 46) if x not in nums)),
        })
    return pd.DataFrame(rows)


class V50PredictiveEngineTests(unittest.TestCase):
    def test_feature_shape_and_finite_values(self):
        frame = synthetic_history(80)
        x, pair = build_number_features(frame)
        self.assertEqual(x.shape, (45, 10))
        self.assertEqual(pair.shape, (45, 45))
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.isfinite(pair).all())
        self.assertTrue(np.allclose(pair, pair.T))
        self.assertTrue(np.allclose(np.diag(pair), 0.0))

    def test_calibration_sums_to_six(self):
        probs = _calibrate_to_six(np.linspace(-3.0, 3.0, 45))
        self.assertEqual(probs.shape, (45,))
        self.assertAlmostEqual(float(probs.sum()), 6.0, places=7)
        self.assertTrue(np.all((probs > 0) & (probs < 1)))

    def test_next_draw_prediction_and_tickets(self):
        frame = synthetic_history(90)
        result = fit_predict_next(frame, training_window=50, min_history=40, ridge_alpha=1.0)
        self.assertEqual(result["target_draw"], 91)
        self.assertAlmostEqual(result["probability_sum"], 6.0, places=6)
        tickets = generate_tickets(
            result["number_probabilities"]["预测入选概率"].to_numpy()[
                np.argsort(result["number_probabilities"]["号码"].to_numpy())],
            result["pair_scores"], count=10, seed=123,
        )
        self.assertEqual(len(tickets), 10)
        self.assertEqual(len({tuple(x) for x in tickets}), 10)
        self.assertTrue(all(len(x) == 6 and len(set(x)) == 6 and min(x) >= 1 and max(x) <= 45 for x in tickets))
        lock, metadata = build_v50_forward_lock(frame, result, tickets_per_group=10, seed=42)
        self.assertEqual(len(lock), 40)
        self.assertEqual(metadata["target_draw"], 91)
        self.assertEqual(lock["策略"].value_counts().to_dict(), {
            "V5联合概率模型": 10, "V5频率精简模型": 10,
            "V4冻结长期频率模型": 10, "均匀随机对照": 10,
        })
        self.assertEqual(lock["目标期号"].nunique(), 1)

    def test_walk_forward_is_reproducible_and_has_random_control(self):
        frame = synthetic_history(100)
        a = run_v50_walk_forward(
            frame, test_draws=3, training_window=50, min_history=30,
            tickets_per_draw=5, seed=99, bootstrap_reps=100,
        )
        b = run_v50_walk_forward(
            frame, test_draws=3, training_window=50, min_history=30,
            tickets_per_draw=5, seed=99, bootstrap_reps=100,
        )
        self.assertEqual(a["test_draws"], 3)
        self.assertEqual(a["evidence_status"], "retrospective_only_not_prospective_evidence")
        self.assertEqual(a["detail"]["V5_mean_hits"].tolist(), b["detail"]["V5_mean_hits"].tolist())
        self.assertEqual(a["detail"]["随机_mean_hits"].tolist(), b["detail"]["随机_mean_hits"].tolist())
        self.assertIn("随机基准平均", a["summary"].columns.tolist())
        self.assertTrue(np.allclose(a["detail"]["V5_概率和"], 6.0, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
