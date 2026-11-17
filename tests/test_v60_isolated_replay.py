# -*- coding: utf-8 -*-
import unittest

import pandas as pd

from scripts.v60_isolated_200_replay import (
    build_window_sensitivity,
    isolated_window_bounds,
)


class TestV60IsolatedReplay(unittest.TestCase):
    def test_main_window_bounds_are_inclusive(self):
        self.assertEqual(isolated_window_bounds(1249, 200), (1050, 1249))
        self.assertEqual(isolated_window_bounds(1249, 150), (1100, 1249))
        self.assertEqual(isolated_window_bounds(1249, 100), (1150, 1249))

    def test_invalid_window_rejected(self):
        with self.assertRaises(ValueError):
            isolated_window_bounds(99, 100)
        with self.assertRaises(ValueError):
            isolated_window_bounds(100, 0)

    def test_nested_sensitivity_windows(self):
        rows = []
        for draw in range(1050, 1250):
            rows.append({
                "目标期号": draw,
                "V5_mean_hits": 1.0,
                "V4_mean_hits": 0.9,
                "随机_mean_hits": 0.8,
                "V5_ticket_3plus": 0.02,
                "V4_ticket_3plus": 0.01,
                "随机_ticket_3plus": 0.01,
                "V5_ticket_4plus": 0.001,
                "V4_ticket_4plus": 0.001,
                "随机_ticket_4plus": 0.001,
                "V5_batch_3plus": 0.1,
                "V4_batch_3plus": 0.08,
                "随机_batch_3plus": 0.07,
                "V5_batch_4plus": 0.01,
                "V4_batch_4plus": 0.01,
                "随机_batch_4plus": 0.01,
                "V5_Brier": 0.15,
                "V4_Brier": 0.16,
                "随机_Brier": 0.17,
                "V5_LogLoss": 0.5,
                "V4_LogLoss": 0.51,
                "随机_LogLoss": 0.52,
            })
        result = build_window_sensitivity(pd.DataFrame(rows))
        self.assertEqual(set(result["窗口期数"]), {100, 150, 200})
        self.assertEqual(len(result), 21)
        main = result[(result["窗口期数"] == 200) & (result["指标"] == "每注平均命中数")].iloc[0]
        self.assertAlmostEqual(main["V5相对随机优势_正值较好"], 0.2)


if __name__ == "__main__":
    unittest.main()
