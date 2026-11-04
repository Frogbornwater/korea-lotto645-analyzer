# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd

from lotto_core import (
    COLUMNS,
    MODEL_NAMES,
    extract_records,
    generate_candidates,
    quality_report,
    run_backtest,
    theoretical_random_metrics,
    validate_frame,
)


def make_history(n=120, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for draw_no in range(1, n + 1):
        nums = sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False))
        bonus = next(x for x in range(1, 46) if x not in nums)
        rows.append({
            "draw_no": draw_no, "date": "2025-01-04",
            **{f"n{i}": nums[i - 1] for i in range(1, 7)},
            "bonus": bonus,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


class LottoCoreTests(unittest.TestCase):
    def test_parse_official_round_record(self):
        payload = {
            "resultCode": None,
            "data": {"list": [{
                "ltEpsd": 1201,
                "tm1WnNo": 7, "tm2WnNo": 9, "tm3WnNo": 24,
                "tm4WnNo": 27, "tm5WnNo": 35, "tm6WnNo": 36,
                "bnsWnNo": 37, "ltRflYmd": "20251206",
            }]},
        }
        records = extract_records(payload)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["draw_no"], 1201)
        self.assertEqual([records[0][f"n{i}"] for i in range(1, 7)], [7, 9, 24, 27, 35, 36])
        self.assertEqual(records[0]["date"], "2025-12-06")
        self.assertEqual(records[0]["bonus"], 37)

    def test_parse_nested_latest_payload(self):
        record = {
            "ltEpsd": 1202,
            "tm1WnNo": 1, "tm2WnNo": 2, "tm3WnNo": 3,
            "tm4WnNo": 4, "tm5WnNo": 5, "tm6WnNo": 6,
            "bnsWnNo": 7, "ltRflYmd": "20251213",
        }
        payload = {"data": {"result": {"pstLtEpstInfo": {"lt645": [record]}}}}
        self.assertEqual(extract_records(payload)[0]["draw_no"], 1202)

    def test_reject_conflicting_duplicate_round(self):
        frame = make_history(3)
        duplicate = frame.iloc[[1]].copy()
        duplicate.loc[:, "n1"] = 45
        duplicate.loc[:, "n6"] = 44
        with self.assertRaises(ValueError):
            validate_frame(pd.concat([frame, duplicate], ignore_index=True))

    def test_sequence_gap_is_reported(self):
        frame = make_history(5).drop(index=2).reset_index(drop=True)
        report = quality_report(frame)
        self.assertFalse(report["sequence_complete"])
        self.assertEqual(report["missing_draws"], [3])


    def test_missing_early_rounds_are_reported(self):
        frame = make_history(5).iloc[2:].reset_index(drop=True)
        report = quality_report(frame)
        self.assertFalse(report["sequence_complete"])
        self.assertEqual(report["missing_draws"][:2], [1, 2])

    def test_all_models_generate_valid_tickets_and_relax_transparently(self):
        history = make_history(120)
        generated, relaxed = generate_candidates(
            history, count=10, model="近期频率模型",
            window=50, max_overlap=0, seed=123,
        )
        self.assertTrue(relaxed)
        self.assertEqual(len(generated), 10)
        self.assertTrue((generated["模型"] == "近期频率模型").all())
        self.assertTrue((generated["请求跨组最大重号"] == 0).all())
        self.assertTrue((generated["实际允许最大重号"] >= 1).all())
        for nums in generated["号码列表"]:
            self.assertEqual(len(nums), 6)
            self.assertEqual(len(set(nums)), 6)
            self.assertTrue(all(1 <= n <= 45 for n in nums))
        self.assertLessEqual(
            int(generated["实际跨组最大重号"].iloc[0]),
            int(generated["实际允许最大重号"].iloc[0]),
        )

    def test_every_displayed_model_is_backtested(self):
        history = make_history(90)
        _, summary = run_backtest(
            history, lookback=50, test_draws=6,
            tickets_per_draw=1, n_seeds=1, seed=99,
        )
        self.assertEqual(set(summary["策略"]), set(MODEL_NAMES))

    def test_random_baseline_probability(self):
        metrics = theoretical_random_metrics()
        self.assertEqual(metrics["jackpot_probability"], 1 / 8145060)
        self.assertAlmostEqual(metrics["mean_hits_per_ticket"], 0.8, places=10)
        self.assertGreater(metrics["p3plus"], 0)
        self.assertLess(metrics["p3plus"], 0.1)


if __name__ == "__main__":
    unittest.main()
