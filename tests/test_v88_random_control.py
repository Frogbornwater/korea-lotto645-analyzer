# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd

from lotto_core import NUMBER_COLS
from v88_random_control import _score_random_controls, run_v88_random_control


def synthetic_history(n=50):
    rng = np.random.default_rng(881)
    rows = []
    for draw in range(1, n + 1):
        numbers = sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False))
        row = {"draw_no": draw, "date": ""}
        row.update({col: numbers[i] for i, col in enumerate(NUMBER_COLS)})
        row["bonus"] = next(x for x in range(1, 46) if x not in numbers)
        rows.append(row)
    return pd.DataFrame(rows)


class V88RandomControlTests(unittest.TestCase):
    def test_random_portfolios_have_valid_metrics_and_are_reproducible(self):
        a = _score_random_controls({1, 2, 3, 4, 5, 6}, repetitions=250, rng=np.random.default_rng(88))
        b = _score_random_controls({1, 2, 3, 4, 5, 6}, repetitions=250, rng=np.random.default_rng(88))
        for key in a:
            self.assertEqual(a[key].shape, (250,))
            np.testing.assert_array_equal(a[key], b[key])
        self.assertTrue(np.all((a["batch_3plus"] == 0) | (a["batch_3plus"] == 1)))
        self.assertTrue(np.all(a["best_hits"] <= 6))

    def test_end_to_end_per_draw_random_null(self):
        history = synthetic_history()
        detail = {"target_draw": [50]}
        for method in ("V5", "V8.1", "V8.5"):
            for metric in ("batch_3plus", "batch_4plus", "mean_hits", "ticket_3plus", "ticket_4plus", "best_hits"):
                detail[f"{method}_{metric}"] = [0.0]
        result = run_v88_random_control(
            history, pd.DataFrame(detail), random_portfolios_per_draw=100,
            seed=8801, bootstrap_repetitions=100, block_length=1,
        )
        self.assertEqual(result["report"]["test_draws"], 1)
        self.assertEqual(result["report"]["random_portfolios_per_draw"], 100)
        self.assertEqual(len(result["per_draw"]), 1)
        self.assertIn("V5_batch3plus_empirical_percentile", result["per_draw"].columns)
        self.assertEqual(result["report"]["inference_design"]["statistical_unit"], "target draw, not Monte Carlo portfolio")

    def test_rejects_too_few_random_controls(self):
        frame = synthetic_history(1)
        with self.assertRaises(ValueError):
            run_v88_random_control(frame, pd.DataFrame(), random_portfolios_per_draw=10)


if __name__ == "__main__":
    unittest.main()
