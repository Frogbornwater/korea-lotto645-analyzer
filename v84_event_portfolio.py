# -*- coding: utf-8 -*-
"""V8.4 exact batch-event portfolio design.

This is a structure-only research module. It does not read lottery history,
fit a predictive model, or change any production picker or forward lock.
"""
from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np

V84_VERSION = "8.4.0-exact-batch-event-overlap"
OUTCOME_COUNT = math.comb(45, 6)
TICKET_COUNT = 10
TICKET_SIZE = 6

# Frozen V8.1 draw-1252 batch used only as a structural comparison reference.
V81_REFERENCE_TICKETS: tuple[tuple[int, ...], ...] = (
    (3, 18, 23, 30, 35, 44),
    (14, 25, 29, 37, 39, 41),
    (4, 6, 9, 16, 36, 38),
    (5, 8, 11, 15, 30, 31),
    (1, 7, 12, 18, 20, 21),
    (4, 6, 19, 26, 29, 34),
    (3, 10, 17, 24, 32, 45),
    (1, 22, 28, 32, 33, 41),
    (9, 10, 12, 27, 42, 43),
    (2, 11, 13, 14, 15, 40),
)


def _random_simple_cubic_graph(
    rng: random.Random,
    *,
    vertex_count: int = 10,
    max_restarts: int = 10_000,
) -> tuple[tuple[int, int], ...]:
    """Generate a simple 3-regular graph by randomized stub pairing."""
    if vertex_count != 10:
        raise ValueError("V8.4 currently designs exactly ten tickets.")
    for _ in range(max_restarts):
        stubs = [vertex for vertex in range(vertex_count) for _ in range(3)]
        rng.shuffle(stubs)
        edges: list[tuple[int, int]] = []
        edge_set: set[tuple[int, int]] = set()
        valid = True
        while stubs:
            left = stubs.pop()
            candidates = [
                idx for idx, right in enumerate(stubs)
                if right != left and tuple(sorted((left, right))) not in edge_set
            ]
            if not candidates:
                valid = False
                break
            right_idx = rng.choice(candidates)
            right = stubs.pop(right_idx)
            edge = tuple(sorted((left, right)))
            edge_set.add(edge)
            edges.append(edge)
        if valid and len(edges) == 15:
            degrees = [0] * vertex_count
            for left, right in edges:
                degrees[left] += 1
                degrees[right] += 1
            if degrees == [3] * vertex_count:
                return tuple(sorted(edges))
    raise RuntimeError("Could not construct a simple 3-regular graph.")


def _graph_fingerprint(
    edges: tuple[tuple[int, int], ...], vertex_count: int = 10
) -> tuple[float | int, ...]:
    """Cycle counts plus adjacency spectrum; isomorphic graphs share a key."""
    adjacency = [set() for _ in range(vertex_count)]
    matrix = np.zeros((vertex_count, vertex_count), dtype=float)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
        matrix[left, right] = 1.0
        matrix[right, left] = 1.0

    cycle_counts: list[int] = []
    for cycle_length in range(3, vertex_count + 1):
        count = 0

        def walk(start: int, path: list[int]) -> None:
            nonlocal count
            current = path[-1]
            if len(path) == cycle_length:
                if start in adjacency[current] and path[1] < path[-1]:
                    count += 1
                return
            for nxt in adjacency[current]:
                if nxt > start and nxt not in path:
                    walk(start, path + [nxt])

        for start in range(vertex_count):
            walk(start, [start])
        cycle_counts.append(count)

    spectrum = tuple(float(x) for x in np.round(np.linalg.eigvalsh(matrix), 6))
    return tuple(cycle_counts) + spectrum


def _tickets_from_graph(
    edges: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, ...], ...]:
    """Each graph edge is one shared number; each vertex gets three private."""
    incident: list[list[int]] = [[] for _ in range(10)]
    for number, (left, right) in enumerate(edges, start=1):
        incident[left].append(number)
        incident[right].append(number)
    tickets = []
    for vertex in range(10):
        private = list(range(16 + 3 * vertex, 19 + 3 * vertex))
        ticket = tuple(sorted(incident[vertex] + private))
        if len(ticket) != 6 or len(set(ticket)) != 6:
            raise RuntimeError("Invalid graph-derived ticket.")
        tickets.append(ticket)
    return tuple(tickets)


def generate_v84_candidates(
    *,
    seed: int = 8_452_026,
    target_unique_graphs: int = 19,
    max_attempts: int = 10_000,
) -> list[dict[str, object]]:
    """Create deterministic portfolios with full coverage and overlap <= 1.

    Each portfolio corresponds to a distinct cycle/spectrum fingerprint of a
    simple 3-regular graph on ten ticket vertices. This is a structural search,
    not a probability-model fit.
    """
    target = int(target_unique_graphs)
    if not 1 <= target <= 19:
        raise ValueError("target_unique_graphs must be between 1 and 19.")
    rng = random.Random(int(seed))
    unique: dict[tuple[float | int, ...], tuple[tuple[int, int], ...]] = {}
    for _ in range(int(max_attempts)):
        edges = _random_simple_cubic_graph(rng)
        fingerprint = _graph_fingerprint(edges)
        unique.setdefault(fingerprint, edges)
        if len(unique) >= target:
            break
    if len(unique) != target:
        raise RuntimeError(
            f"Only {len(unique)} unique graph fingerprints found; expected {target}."
        )

    candidates = []
    for index, (fingerprint, edges) in enumerate(sorted(unique.items()), start=1):
        candidates.append({
            "candidate_id": f"graph_{index:02d}",
            "graph_edges": [list(edge) for edge in edges],
            "graph_fingerprint": list(fingerprint),
            "tickets": [list(ticket) for ticket in _tickets_from_graph(edges)],
        })
    return candidates


def portfolio_structure(tickets: Iterable[Iterable[int]]) -> dict[str, object]:
    rows = [tuple(sorted(int(n) for n in ticket)) for ticket in tickets]
    if len(rows) != TICKET_COUNT:
        raise ValueError("A portfolio must contain exactly ten tickets.")
    if any(len(row) != TICKET_SIZE or len(set(row)) != TICKET_SIZE for row in rows):
        raise ValueError("Every ticket must contain six distinct numbers.")
    if any(n < 1 or n > 45 for row in rows for n in row):
        raise ValueError("Ticket numbers must be between 1 and 45.")
    if len(set(rows)) != TICKET_COUNT:
        raise ValueError("All ten tickets must be distinct.")

    coverage = set(number for row in rows for number in row)
    overlaps = [
        len(set(rows[i]) & set(rows[j]))
        for i in range(TICKET_COUNT)
        for j in range(i + 1, TICKET_COUNT)
    ]
    histogram = {str(level): overlaps.count(level) for level in range(7)}
    return {
        "ticket_count": len(rows),
        "number_coverage": len(coverage),
        "uncovered_numbers": sorted(set(range(1, 46)) - coverage),
        "max_pairwise_overlap": max(overlaps),
        "mean_pairwise_overlap": float(np.mean(overlaps)),
        "pairwise_overlap_histogram": histogram,
    }


def exact_random_distinct_portfolio_baseline(ticket_count: int = 10) -> dict[str, float]:
    """Exact expected batch event probability for uniformly sampled distinct tickets."""
    m = int(ticket_count)
    total = OUTCOME_COUNT
    result: dict[str, float] = {}
    for threshold in (3, 4, 5, 6):
        favorable_single = sum(
            math.comb(6, hits) * math.comb(39, 6 - hits)
            for hits in range(threshold, 7)
        )
        # m unique tickets sampled without replacement from all C(45,6).
        p_any = 1.0 - math.comb(total - favorable_single, m) / math.comb(total, m)
        result[f"any_{threshold}_plus_probability"] = p_any
    return result
