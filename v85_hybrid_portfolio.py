# -*- coding: utf-8 -*-
"""V8.5: probability-weighted batch quotas under V8.4 low-overlap structure."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np

from v81_batch_sampler import dependent_round_appearances

V85_VERSION = "8.5.0-probability-weighted-low-overlap-portfolio"
TICKET_COUNT = 10
TICKET_SIZE = 6
NUMBER_COUNT = 45

# The best exact-uniform V8.4 graph (graph_17), with vertices representing tickets.
V84_GRAPH17_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 5), (1, 5), (1, 8),
    (2, 5), (2, 6), (3, 7), (3, 8), (3, 9),
    (4, 6), (4, 7), (4, 9), (6, 7), (8, 9),
)


def _validate_probabilities(probabilities: Iterable[float]) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if p.shape != (NUMBER_COUNT,):
        raise ValueError("V8.5需要45个号码的边际入选概率。")
    if not np.all(np.isfinite(p)) or np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("V8.5边际概率必须是(0,1)内的有限数值。")
    if not np.isclose(float(p.sum()), TICKET_SIZE, atol=1e-5):
        raise ValueError("45个边际入选概率之和必须约为6。")
    return p


def _portfolio_metrics(tickets: list[list[int]]) -> dict[str, object]:
    rows = [tuple(sorted(int(n) for n in ticket)) for ticket in tickets]
    if len(rows) != TICKET_COUNT or len(set(rows)) != TICKET_COUNT:
        raise ValueError("V8.5必须生成10注互不相同的候选。")
    if any(len(row) != TICKET_SIZE or len(set(row)) != TICKET_SIZE
           or min(row) < 1 or max(row) > NUMBER_COUNT for row in rows):
        raise ValueError("每注必须是6个互不重复的1—45号码。")
    overlaps = [
        len(set(rows[i]).intersection(rows[j]))
        for i in range(TICKET_COUNT) for j in range(i + 1, TICKET_COUNT)
    ]
    covered = sorted({number for row in rows for number in row})
    return {
        "ticket_count": len(rows),
        "unique_ticket_count": len(set(rows)),
        "number_coverage": len(covered),
        "uncovered_numbers": sorted(set(range(1, NUMBER_COUNT + 1)) - set(covered)),
        "max_pairwise_overlap": int(max(overlaps, default=0)),
        "mean_pairwise_overlap": float(np.mean(overlaps)),
        "pairwise_overlap_histogram": {
            str(level): int(overlaps.count(level)) for level in range(TICKET_SIZE + 1)
        },
    }


def _graph17_portfolio(counts: np.ndarray, probabilities: np.ndarray) -> list[list[int]]:
    """Map 15 double-appearance numbers to graph edges, 30 singles to vertices."""
    repeated = sorted(np.flatnonzero(counts == 2).tolist(),
                      key=lambda index: (-float(probabilities[index]), index))
    singles = sorted(np.flatnonzero(counts == 1).tolist(),
                     key=lambda index: (-float(probabilities[index]), index))
    if len(repeated) != 15 or len(singles) != 30 or np.any((counts != 1) & (counts != 2)):
        raise ValueError("V8.4 graph mapping requires exactly 15 double and 30 single appearances.")
    incident: list[list[int]] = [[] for _ in range(TICKET_COUNT)]
    for edge_index, (left, right) in enumerate(V84_GRAPH17_EDGES):
        number = repeated[edge_index] + 1
        incident[left].append(number)
        incident[right].append(number)
    tickets: list[list[int]] = []
    for vertex in range(TICKET_COUNT):
        private = [int(x) + 1 for x in singles[3 * vertex:3 * vertex + 3]]
        tickets.append(sorted(incident[vertex] + private))
    return tickets


def _generic_incidence_portfolio(
    counts: np.ndarray, probabilities: np.ndarray, seed: int,
    max_search_nodes: int = 300_000,
) -> list[list[int]]:
    """Backtracking allocator for rounded column degrees other than 1/2."""
    rng = np.random.default_rng(int(seed))
    row_capacity = np.full(TICKET_COUNT, TICKET_SIZE, dtype=np.int16)
    pair_used = np.zeros((TICKET_COUNT, TICKET_COUNT), dtype=bool)
    assigned: dict[int, tuple[int, ...]] = {}
    tie = rng.random(NUMBER_COUNT)
    repeated_columns = [i for i, degree in enumerate(counts.tolist()) if int(degree) >= 2]
    repeated_columns.sort(key=lambda i: (-int(counts[i]), -float(probabilities[i]), float(tie[i])))
    visited = 0

    def place(position: int) -> bool:
        nonlocal visited
        visited += 1
        if visited > int(max_search_nodes):
            return False
        if position == len(repeated_columns):
            single_columns = [i for i, degree in enumerate(counts.tolist()) if int(degree) == 1]
            if int(row_capacity.sum()) != len(single_columns):
                return False
            rng.shuffle(single_columns)
            cursor = 0
            for row_index in range(TICKET_COUNT):
                need = int(row_capacity[row_index])
                for column in single_columns[cursor:cursor + need]:
                    assigned[column] = (row_index,)
                cursor += need
            return cursor == len(single_columns)

        column = repeated_columns[position]
        degree = int(counts[column])
        choices = []
        for subset in combinations(range(TICKET_COUNT), degree):
            if any(row_capacity[row] <= 0 for row in subset):
                continue
            if any(pair_used[left, right] for left, right in combinations(subset, 2)):
                continue
            loads = TICKET_SIZE - row_capacity.astype(float)
            after = loads.copy()
            after[list(subset)] += 1.0
            balance = float(np.square(after - after.mean()).sum())
            choices.append((balance, float(rng.random()), subset))
        choices.sort(key=lambda item: (item[0], item[1]))

        for _, _, subset in choices:
            for row in subset:
                row_capacity[row] -= 1
            for left, right in combinations(subset, 2):
                pair_used[left, right] = pair_used[right, left] = True
            assigned[column] = tuple(subset)
            if place(position + 1):
                return True
            assigned.pop(column, None)
            for left, right in combinations(subset, 2):
                pair_used[left, right] = pair_used[right, left] = False
            for row in subset:
                row_capacity[row] += 1
            if visited > int(max_search_nodes):
                break
        return False

    if not place(0):
        raise RuntimeError(
            "本期概率配额无法在V8.4低重号约束下完成分配；"
            f"搜索节点={visited}，配额分布={np.bincount(counts, minlength=11).tolist()}。"
        )
    tickets = [[] for _ in range(TICKET_COUNT)]
    for number_index, rows in assigned.items():
        for row_index in rows:
            tickets[row_index].append(number_index + 1)
    return [sorted(row) for row in tickets]


def generate_probability_weighted_portfolio(
    probabilities: Iterable[float], *, seed: int = 8_552_026,
) -> tuple[list[list[int]], dict[str, object]]:
    """Generate ten quota-aligned tickets with pairwise overlap no greater than one."""
    p = _validate_probabilities(probabilities)
    counts = dependent_round_appearances(
        p, ticket_count=TICKET_COUNT, sample_size=TICKET_SIZE, seed=int(seed)
    )
    if int(counts.sum()) != TICKET_COUNT * TICKET_SIZE:
        raise RuntimeError("V8.1依赖舍入未保留60个批次号码位。")
    if np.all((counts == 1) | (counts == 2)):
        tickets = _graph17_portfolio(counts, p)
        allocation_method = "V84_graph17_with_V81_unbiased_marginal_rounding"
    else:
        tickets = _generic_incidence_portfolio(counts, p, int(seed) ^ 0x85A51)
        allocation_method = "backtracking_low_overlap_incidence_design"

    structure = _portfolio_metrics(tickets)
    realized = np.zeros(NUMBER_COUNT, dtype=np.int64)
    for ticket in tickets:
        realized[np.asarray(ticket, dtype=int) - 1] += 1
    if not np.array_equal(realized, counts):
        raise RuntimeError("V8.5最终组合与依赖舍入的号码配额不一致。")
    expected = TICKET_COUNT * p
    if not np.all((counts >= np.floor(expected) - 1e-8)
                  & (counts <= np.ceil(expected) + 1e-8)):
        raise RuntimeError("V8.5最终出现次数不满足floor/ceil配额约束。")
    if structure["max_pairwise_overlap"] > 1:
        raise RuntimeError("V8.5违反任意两注最多共享一个号码的结构约束。")

    report: dict[str, object] = {
        "sampler_version": V85_VERSION,
        "allocation_method": allocation_method,
        **structure,
        "seed": int(seed),
        "total_number_slots": int(TICKET_COUNT * TICKET_SIZE),
        "target_expected_appearances": expected.tolist(),
        "realized_appearances": counts.tolist(),
        "batch_counts_are_floor_or_ceil": True,
        "batch_expected_appearances_preserved_by_randomized_rounding": True,
        "per_ticket_marginals_identical": False,
        "production_model_changed": False,
        "prospective_lock_created": False,
        "interpretation": (
            "V8.5先按V5给出的45个边际概率进行V8.1无偏依赖舍入，再将整数配额"
            "分配到最多两两共享一个号码的十注组合。跨随机种子的期望批次出现次数"
            "为10*p_i；单批号码出现次数是floor/ceil配额。它优化的是批次结构，"
            "不是单注独立分布，也不证明未来开奖预测优势。"
        ),
    }
    return tickets, report
