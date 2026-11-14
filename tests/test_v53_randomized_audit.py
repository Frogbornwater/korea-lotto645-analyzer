import json
import unittest

import numpy as np
import pandas as pd

from v53_randomized_audit import run_multi_seed_random_audit


def make_detail(draws=24, ticket_count=10):
    rng = np.random.default_rng(431)
    rows = []
    for target in range(1001, 1001 + draws):
        actual = sorted(rng.choice(np.arange(1, 46), size=6, replace=False).tolist())
        v5 = []
        v4 = []
        for _ in range(ticket_count):
            v5.append(sorted(rng.choice(np.arange(1, 46), size=6, replace=False).tolist()))
            v4.append(sorted(rng.choice(np.arange(1, 46), size=6, replace=False).tolist()))
        probabilities = np.full(45, 6.0 / 45.0)
        rows.append({
            "目标期号": target,
            "训练结束期": target - 1,
            "训练期数": 500,
            "实际主号码": json.dumps(actual),
            "V5_号码列表": "|".join(str(ticket) for ticket in v5),
            "V4_号码列表": "|".join(str(ticket) for ticket in v4),
            "V5_45号码概率_JSON": json.dumps(probabilities.tolist()),
            "V4_45号码概率_JSON": json.dumps(probabilities.tolist()),
        })
    return pd.DataFrame(rows)


class V53RandomizedAuditTests(unittest.TestCase):
    def test_repeated_random_audit_is_reproducible(self):
        detail = make_detail()
        first = run_multi_seed_random_audit(detail, random_repetitions=40, master_seed=20261114)
        second = run_multi_seed_random_audit(detail, random_repetitions=40, master_seed=20261114)
        pd.testing.assert_frame_equal(first["summary"], second["summary"])
        pd.testing.assert_frame_equal(first["random_distribution"], second["random_distribution"])
        self.assertEqual(first["manifest"]["target_draws"], 24)
        self.assertEqual(first["manifest"]["tickets_per_draw"], 10)
        self.assertEqual(len(first["random_distribution"]), 40)

    def test_rejects_future_training_alignment(self):
        detail = make_detail()
        detail.loc[5, "训练结束期"] = detail.loc[5, "目标期号"]
        with self.assertRaisesRegex(ValueError, "未来数据泄漏"):
            run_multi_seed_random_audit(detail, random_repetitions=10)

    def test_rejects_missing_actual_numbers(self):
        detail = make_detail().drop(columns=["实际主号码"])
        with self.assertRaisesRegex(ValueError, "缺少V5.3审计字段"):
            run_multi_seed_random_audit(detail, random_repetitions=10)

    def test_empirical_p_values_and_holm_values_are_valid(self):
        result = run_multi_seed_random_audit(make_detail(), random_repetitions=25, master_seed=7)
        ticket_summary = result["summary"].iloc[:6]
        self.assertTrue(ticket_summary["单侧经验p值"].between(1 / 26, 1).all())
        self.assertTrue(ticket_summary["Holm校正p值"].between(1 / 26, 1).all())
        self.assertEqual(result["manifest"]["evidence_status"], "retrospective_exploratory_only")


if __name__ == "__main__":
    unittest.main()
