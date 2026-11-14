import unittest

import numpy as np
import pandas as pd

from lotto_core import COLUMNS
from v50_predictive_engine import fit_predict_next
from v51_smart_picker import generate_constrained_tickets
from v52_rolling_validation import (
    V51_LOCK_EVENT_TYPE,
    build_v51_forward_lock_event,
    evaluate_v51_forward_lock_events,
    event_tickets_frame,
)


def synthetic_history(n=240, seed=52):
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(1, n + 1):
        nums = sorted(rng.choice(np.arange(1, 46), size=6, replace=False).tolist())
        bonus = int(next(x for x in range(1, 46) if x not in nums))
        rows.append({
            "draw_no": draw, "date": "", "n1": nums[0], "n2": nums[1],
            "n3": nums[2], "n4": nums[3], "n5": nums[4], "n6": nums[5],
            "bonus": bonus,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


class V52RollingValidationTests(unittest.TestCase):
    def test_lock_event_contains_v51_and_random_tickets_and_full_probabilities(self):
        history = synthetic_history(240)
        prediction = fit_predict_next(history, training_window=180, min_history=180, ridge_alpha=1.0)
        probs = prediction["number_probabilities"].sort_values("号码")["预测入选概率"].to_numpy()
        tickets, effective, relaxed = generate_constrained_tickets(
            probs, prediction["pair_scores"], count=10, seed=99,
            pair_strength=0.5, max_overlap=4,
        )
        candidate = pd.DataFrame([
            {"号码": " · ".join(f"{n:02d}" for n in ticket), "模型相对得分": float(100-i)}
            for i, ticket in enumerate(tickets)
        ])
        event = build_v51_forward_lock_event(
            history, prediction, candidate, seed=99, pair_strength=0.5,
            requested_overlap=4, effective_overlap=effective,
        )
        self.assertEqual(event["event_type"], V51_LOCK_EVENT_TYPE)
        self.assertEqual(len(event["payload"]["number_probabilities"]), 45)
        locked = event_tickets_frame(event)
        self.assertEqual(locked["策略"].value_counts().to_dict(), {
            "V5.1联合概率选号": 10, "均匀随机基准": 10,
        })
        self.assertEqual(event["payload"]["metadata"]["target_draw"], 241)

    def test_forward_score_waits_then_scores_only_matching_manual_history(self):
        history = synthetic_history(240)
        prediction = fit_predict_next(history, training_window=180, min_history=180, ridge_alpha=1.0)
        probs = prediction["number_probabilities"].sort_values("号码")["预测入选概率"].to_numpy()
        tickets, effective, _ = generate_constrained_tickets(
            probs, prediction["pair_scores"], count=10, seed=101,
            pair_strength=0.5, max_overlap=4,
        )
        candidate = pd.DataFrame([{"号码": " ".join(map(str, ticket))} for ticket in tickets])
        event = build_v51_forward_lock_event(
            history, prediction, candidate, seed=101, pair_strength=0.5,
            requested_overlap=4, effective_overlap=effective,
        )
        detail, summary, status = evaluate_v51_forward_lock_events(history, [event])
        self.assertTrue(summary.empty)
        self.assertEqual(status.iloc[0]["状态"], "等待开奖数据")

        next_row = synthetic_history(241).tail(1)
        extended = pd.concat([history, next_row], ignore_index=True)
        detail, summary, status = evaluate_v51_forward_lock_events(extended, [event])
        self.assertEqual(status.iloc[0]["状态"], "已评分")
        self.assertEqual(len(detail), 20)
        self.assertEqual(len(summary), 1)
        self.assertIn("V5.1平均命中数", summary.columns)

    def test_lock_rejects_wrong_history_or_non_next_target(self):
        history = synthetic_history(240)
        prediction = fit_predict_next(history, training_window=180, min_history=180, ridge_alpha=1.0)
        candidate = pd.DataFrame([{"号码": "1 2 3 4 5 6"}])
        bad = dict(prediction)
        bad["history_last_draw"] = 239
        with self.assertRaises(ValueError):
            build_v51_forward_lock_event(
                history, bad, candidate, seed=1, pair_strength=0.5,
                requested_overlap=4, effective_overlap=4,
            )


if __name__ == "__main__":
    unittest.main()
