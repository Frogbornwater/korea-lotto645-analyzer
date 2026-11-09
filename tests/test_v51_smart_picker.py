import unittest

import numpy as np

from v51_smart_picker import generate_constrained_tickets


class V51SmartPickerTests(unittest.TestCase):
    def test_reproducible_unique_tickets_and_overlap_contract(self):
        probabilities = np.full(45, 6.0 / 45.0)
        pair_scores = np.zeros((45, 45), dtype=float)
        a = generate_constrained_tickets(
            probabilities, pair_scores, count=10, seed=987,
            pair_strength=0.0, max_overlap=4,
        )
        b = generate_constrained_tickets(
            probabilities, pair_scores, count=10, seed=987,
            pair_strength=0.0, max_overlap=4,
        )
        tickets, effective, relaxed = a
        self.assertEqual(tickets, b[0])
        self.assertEqual(len(tickets), 10)
        self.assertEqual(len({tuple(ticket) for ticket in tickets}), 10)
        self.assertTrue(all(len(ticket) == 6 and len(set(ticket)) == 6 for ticket in tickets))
        actual = max(
            len(set(left).intersection(right))
            for i, left in enumerate(tickets)
            for right in tickets[i + 1:]
        )
        self.assertLessEqual(actual, effective)
        self.assertGreaterEqual(effective, 4)
        self.assertEqual(relaxed, effective > 4)

    def test_impossible_zero_overlap_is_reported_as_relaxed(self):
        probabilities = np.full(45, 6.0 / 45.0)
        tickets, effective, relaxed = generate_constrained_tickets(
            probabilities, count=10, seed=123, pair_strength=0.0,
            max_overlap=0,
        )
        self.assertEqual(len(tickets), 10)
        self.assertGreater(effective, 0)
        self.assertTrue(relaxed)

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_constrained_tickets(
                np.full(45, 6.0 / 45.0), max_overlap=6,
            )


if __name__ == "__main__":
    unittest.main()
