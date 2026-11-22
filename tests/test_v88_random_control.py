# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd

from v88_random_control import _score_random_controls, run_v88_random_control


class V88RandomControlTests(unittest.TestCase):
    def test_random_portfolios_have_valid_metrics_and_are_reproducible(self):
        a = _score_random_controls({1, 2, 3, 4, 5, 6}, repetitions=250, rng=np.random.default_rng(88))
        b = _score_random_controls({1, 2, 3, 4, 5, 6}, repetitions=250, rng=np.random.default_rng(88))
        for key in a:
            self.assertEqual(a[key].shape, (250,))
            np.testing.assert_array_equal(a[key], b[key])
        self.assertTrue(np.all((a["batch_3plus"] == 0) | (a["batch_3plus"] == 1)))
        self.assertTrue(np.all(a["best_hits"] <= 6))

    def test_rejects_too_few_random_controls(self):
        frame = pd.DataFrame({
            "draw_no": [1], "date": ["2020-01-01"],
            "n1": [1], "n2": [2], "n3": [3], "n4": [4], "n5": [5], "n6": [6],
            "bonus": [7],
        })
        with self.assertRaises(ValueError):
            run_v88_random_control(frame, pd.DataFrame(), random_portfolios_per_draw=10)


if __name__ == "__main__":
    unittest.main()
