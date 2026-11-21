# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v84_event_portfolio import (
    V81_REFERENCE_TICKETS,
    exact_random_distinct_portfolio_baseline,
    generate_v84_candidates,
    portfolio_structure,
)


class V84EventPortfolioTests(unittest.TestCase):
    def test_search_is_deterministic_and_returns_nineteen_graph_signatures(self):
        first = generate_v84_candidates(seed=8_452_026, target_unique_graphs=19)
        second = generate_v84_candidates(seed=8_452_026, target_unique_graphs=19)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 19)
        self.assertEqual(len({tuple(row["graph_fingerprint"]) for row in first}), 19)

    def test_every_candidate_has_full_coverage_and_maximum_overlap_one(self):
        candidates = generate_v84_candidates(seed=8_452_026, target_unique_graphs=19)
        for candidate in candidates:
            metrics = portfolio_structure(candidate["tickets"])
            self.assertEqual(metrics["ticket_count"], 10)
            self.assertEqual(metrics["number_coverage"], 45)
            self.assertEqual(metrics["max_pairwise_overlap"], 1)
            self.assertAlmostEqual(metrics["mean_pairwise_overlap"], 1.0 / 3.0)
            self.assertEqual(metrics["pairwise_overlap_histogram"]["1"], 15)
            self.assertEqual(metrics["pairwise_overlap_histogram"]["2"], 0)

    def test_v81_reference_diagnostics_match_frozen_comparison(self):
        metrics = portfolio_structure(V81_REFERENCE_TICKETS)
        self.assertEqual(metrics["number_coverage"], 45)
        self.assertEqual(metrics["max_pairwise_overlap"], 2)
        self.assertEqual(metrics["pairwise_overlap_histogram"]["2"], 2)

    def test_exact_random_distinct_baseline_is_ordered_by_threshold(self):
        baseline = exact_random_distinct_portfolio_baseline(10)
        values = [baseline[f"any_{k}_plus_probability"] for k in (3, 4, 5, 6)]
        self.assertTrue(all(values[i] >= values[i + 1] for i in range(3)))
        self.assertAlmostEqual(values[0], 0.21433688902346237, places=12)
        self.assertAlmostEqual(values[1], 0.013847777540374473, places=12)


if __name__ == "__main__":
    unittest.main()
