import unittest

import numpy as np
import pandas as pd

from v94_multiscale_smart_picker import (
    ENGINE_VERSION, fit_v94_multiscale_next, generate_v94_tickets,
)


def _history(rows=1001):
    rng = np.random.default_rng(9401)
    records = []
    for draw in range(1, rows + 1):
        numbers = sorted(rng.choice(np.arange(1, 46), size=6, replace=False).tolist())
        bonus_pool = sorted(set(range(1, 46)) - set(numbers))
        bonus = int(rng.choice(bonus_pool))
        records.append({
            "draw_no": draw, "date": "",
            **{f"n{i}": numbers[i - 1] for i in range(1, 7)},
            "bonus": bonus,
        })
    return pd.DataFrame(records)


class V94MultiScaleSmartPickerTests(unittest.TestCase):
    def test_probability_vector_and_target_draw(self):
        prediction = fit_v94_multiscale_next(_history())
        self.assertEqual(prediction["engine_version"], ENGINE_VERSION)
        self.assertEqual(prediction["target_draw"], 1002)
        self.assertEqual(prediction["history_last_draw"], 1001)
        self.assertEqual(len(prediction["number_probabilities"]), 45)
        self.assertAlmostEqual(prediction["probability_sum"], 6.0, places=8)

    def test_generates_ten_valid_diversified_tickets(self):
        prediction = fit_v94_multiscale_next(_history())
        frame, effective_overlap, relaxed = generate_v94_tickets(
            prediction, count=10, seed=9402, max_overlap=4
        )
        self.assertEqual(len(frame), 10)
        self.assertTrue(frame["号码"].is_unique)
        tickets = [
            [int(value) for value in value_text.split(" · ")]
            for value_text in frame["号码"]
        ]
        self.assertTrue(all(
            len(ticket) == 6 and len(set(ticket)) == 6
            and all(1 <= number <= 45 for number in ticket)
            for ticket in tickets
        ))
        self.assertTrue(all(
            len(set(left).intersection(right)) <= effective_overlap
            for index, left in enumerate(tickets)
            for right in tickets[index + 1:]
        ))
        self.assertEqual(relaxed, effective_overlap > 4)

    def test_rejects_short_or_gapped_history(self):
        with self.assertRaisesRegex(ValueError, "至少需要1001期"):
            fit_v94_multiscale_next(_history(1000))
        gapped = _history()
        gapped.loc[500, "draw_no"] = 9999
        with self.assertRaisesRegex(ValueError, "期号存在缺失"):
            fit_v94_multiscale_next(gapped)


if __name__ == "__main__":
    unittest.main()
