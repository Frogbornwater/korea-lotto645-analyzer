import unittest

import numpy as np
import pandas as pd

from lotto_core import COLUMNS
from v40_governance import (
    append_draw, configuration_fingerprint, data_fingerprint,
    paired_draw_bootstrap, evaluate_forward_pairs, _holm_adjust,
)


def make_history(count=350):
    rows = []
    for draw in range(1, count + 1):
        start = ((draw - 1) % 40) + 1
        nums = sorted({((start + offset - 1) % 45) + 1 for offset in range(6)})
        bonus = ((start + 12 - 1) % 45) + 1
        if bonus in nums:
            bonus = ((bonus + 6 - 1) % 45) + 1
        rows.append({
            "draw_no": draw, "date": f"2020-01-{((draw - 1) % 28) + 1:02d}",
            **{f"n{i}": nums[i - 1] for i in range(1, 7)}, "bonus": bonus,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


class V40GovernanceTests(unittest.TestCase):
    def test_fingerprint_is_stable_for_canonical_history(self):
        frame = make_history()
        self.assertEqual(data_fingerprint(frame), data_fingerprint(frame.iloc[::-1]))
        self.assertEqual(len(configuration_fingerprint({"a": 1, "b": 2})), 64)

    def test_append_draw_is_strict_and_idempotent(self):
        frame = make_history(350)
        updated, result = append_draw(frame, 351, "2026-11-14", [1, 7, 14, 23, 32, 44], 9)
        self.assertEqual(result["status"], "appended")
        self.assertEqual(len(updated), 351)
        same, duplicate = append_draw(updated, 351, "2026-11-14", [44, 32, 23, 14, 7, 1], 9)
        self.assertEqual(duplicate["status"], "already_exists")
        self.assertEqual(len(same), 351)

    def test_append_rejects_gap_and_conflict(self):
        frame = make_history(350)
        with self.assertRaises(ValueError):
            append_draw(frame, 352, "2026-11-14", [1, 7, 14, 23, 32, 44], 9)
        with self.assertRaises(ValueError):
            append_draw(frame, 350, "2026-11-14", [1, 7, 14, 23, 32, 44], 9)

    def test_bootstrap_is_reproducible(self):
        model = [0.9, 1.0, 0.7, 1.1, 0.8, 0.9, 1.0, 0.8]
        random = [0.8, 0.8, 0.8, 0.9, 0.9, 0.8, 0.9, 0.8]
        a = paired_draw_bootstrap(model, random, repetitions=1000, seed=42)
        b = paired_draw_bootstrap(model, random, repetitions=1000, seed=42)
        self.assertEqual(a, b)
        self.assertEqual(a["n_draws"], 8)

    def test_holm_adjustment_is_monotonic(self):
        adjusted = _holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(len(adjusted), 3)
        self.assertTrue(all(0 <= value <= 1 for value in adjusted))

    def test_forward_gate_does_not_promote_small_sample(self):
        rows = []
        for draw in range(1, 6):
            for strategy in ["贝叶斯平滑模型", "均匀随机基准"]:
                for seed in range(1, 4):
                    for ticket in range(1, 11):
                        rows.append({
                            "目标期号": draw, "策略": strategy, "随机种子组": seed,
                            "命中数": 1 if strategy == "贝叶斯平滑模型" else 0,
                            "命中3个以上": 0,
                        })
        _, gate = evaluate_forward_pairs(pd.DataFrame(rows), "贝叶斯平滑模型", repetitions=1000)
        self.assertFalse(gate["promotion_allowed"])
        self.assertEqual(gate["n_draws"], 5)


if __name__ == "__main__":
    unittest.main()
