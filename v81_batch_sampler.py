# -*- coding: utf-8 -*-
"""V8.1 batch-quota portfolio sampling for Lotto 6/45.

V8.0 preserves the requested first-order inclusion probabilities for each
independent ticket. This module addresses a different, batch-level question:
construct a portfolio of m tickets (each containing k numbers) whose aggregate
number counts are randomized floor/ceil roundings of m * p_i.

Pairwise dependent rounding preserves E[count_i] = m * p_i while preserving
the exact total m*k. A randomized balanced bipartite allocation then places
those appearances into m distinct rows of k unique numbers each.

This is NOT the same as giving every individual ticket the same marginal
distribution. It is a batch-level allocation strategy and must be evaluated
separately from V8.0 independent conditional-Bernoulli sampling.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

V81_PORTFOLIO_VERSION = "8.1.0-dependent-rounding-batch-quota"


def _validate_probabilities(
    probabilities: Iterable[float] | np.ndarray,
    sample_size: int,
    *,
    sum_tolerance: float = 1e-5,
) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float)
    if raw.ndim != 1:
        raise ValueError("probabilities必须是一维向量。")
    p = raw.reshape(-1)
    if p.size < 2:
        raise ValueError("probabilities必须至少包含2个号码。")
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities包含非有限值。")
    k = int(sample_size)
    if not 1 <= k < p.size:
        raise ValueError("sample_size必须位于1与号码总数之间。")
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("每个号码的边际入选概率必须严格位于0与1之间。")
    if abs(float(p.sum()) - k) > float(sum_tolerance):
        raise ValueError(
            f"边际入选概率之和必须为{k}；当前为{float(p.sum()):.10f}。"
        )
    p = p * (float(k) / float(p.sum()))
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("概率总和微调后出现边界值。")
    return p


def dependent_round_appearances(
    probabilities: Iterable[float] | np.ndarray,
    ticket_count: int = 10,
    sample_size: int = 6,
    *,
    seed: int = 2028,
) -> np.ndarray:
    """Round m*p to integer counts without bias or changing the total.

    Each output count is either floor(m*p_i) or ceil(m*p_i), the sum is m*k,
    and pairwise randomized rounding preserves its expectation at m*p_i.
    """
    p = _validate_probabilities(probabilities, sample_size)
    m = int(ticket_count)
    k = int(sample_size)
    if not 1 <= m <= 1000:
        raise ValueError("ticket_count必须位于1至1000之间。")

    expected = p * float(m)
    values = expected.copy()
    rng = np.random.default_rng(int(seed))
    tolerance = 1e-10

    # Every pairwise update preserves the pair's sum and conditional mean.
    # At least one of the two selected fractional coordinates becomes integer.
    for _ in range(values.size * 20):
        fractional = np.flatnonzero(np.abs(values - np.rint(values)) > tolerance)
        if fractional.size < 2:
            break
        i, j = (int(x) for x in rng.choice(fractional, size=2, replace=False))
        frac_i = values[i] - math.floor(values[i])
        frac_j = values[j] - math.floor(values[j])

        step_up = min(math.ceil(values[i]) - values[i], frac_j)
        step_down = min(frac_i, math.ceil(values[j]) - values[j])
        if step_up + step_down <= tolerance:
            continue

        # Choose +step_up with this probability, otherwise -step_down.
        # The expected change is exactly zero:
        # q*step_up - (1-q)*step_down = 0.
        probability_up = step_down / (step_up + step_down)
        if rng.random() < probability_up:
            values[i] += step_up
            values[j] -= step_up
        else:
            values[i] -= step_down
            values[j] += step_down

    rounded = np.rint(values).astype(np.int64)
    expected_total = m * k
    if (
        not np.all(np.isfinite(values))
        or np.any(np.abs(values - rounded) > 1e-7)
        or int(rounded.sum()) != expected_total
        or np.any(rounded < 0)
        or np.any(rounded > m)
        or np.any(rounded < np.floor(expected) - 1e-7)
        or np.any(rounded > np.ceil(expected) + 1e-7)
    ):
        raise RuntimeError("依赖舍入未能保持整数配额、总数或floor/ceil约束。")
    return rounded


def _allocate_rows(
    counts: np.ndarray,
    ticket_count: int,
    sample_size: int,
    rng: np.random.Generator,
    *,
    max_attempts: int = 256,
) -> tuple[list[list[int]], int]:
    """Allocate number appearances to a binary m-by-n matrix with row sum k.

    For each number, its appearances go to distinct currently least-filled
    rows. This keeps all row sizes within one of each other; since the final
    total is exactly m*k, every row ends with exactly k numbers.
    """
    m = int(ticket_count)
    k = int(sample_size)
    values = np.asarray(counts, dtype=np.int64).reshape(-1)
    if values.size < 2 or np.any(values < 0) or np.any(values > m):
        raise ValueError("counts必须是0至ticket_count之间的整数向量。")
    if int(values.sum()) != m * k:
        raise ValueError("号码出现次数总和必须等于ticket_count*sample_size。")

    support = int(np.count_nonzero(values))
    unique_capacity = math.comb(support, k) if support >= k else 0
    if unique_capacity < m:
        raise ValueError(
            "本批次的号码配额过度集中：可形成的不同组合最多"
            f"{unique_capacity}组，少于要求的{m}组；不能通过改变号码配额来掩盖此问题。"
        )

    for attempt in range(1, int(max_attempts) + 1):
        rows: list[set[int]] = [set() for _ in range(m)]
        number_ties = rng.random(values.size)
        number_order = sorted(
            range(values.size),
            key=lambda idx: (-int(values[idx]), float(number_ties[idx])),
        )

        for idx in number_order:
            appearances = int(values[idx])
            if appearances == 0:
                continue
            row_sizes = np.fromiter((len(row) for row in rows), dtype=np.int64, count=m)
            row_ties = rng.random(m)
            least_filled_rows = np.lexsort((row_ties, row_sizes))[:appearances]
            for row_idx in least_filled_rows:
                rows[int(row_idx)].add(idx + 1)

        tickets = [sorted(row) for row in rows]
        if all(len(ticket) == k for ticket in tickets):
            if len({tuple(ticket) for ticket in tickets}) == m:
                return tickets, attempt

    raise RuntimeError(
        f"在{max_attempts}次随机分配中仍无法得到{m}组互不相同的组合；"
        "请使用V8.0独立抽样，或检查输入概率是否过度集中。"
    )


def portfolio_diagnostics(
    tickets: Iterable[Iterable[int]],
    probabilities: Iterable[float] | np.ndarray,
) -> dict[str, object]:
    """Summarize batch frequencies and natural overlap without imposing caps."""
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    rows = [tuple(int(number) for number in ticket) for ticket in tickets]
    if not rows:
        raise ValueError("tickets不能为空。")
    k = len(rows[0])
    for ticket in rows:
        if len(ticket) != k or len(set(ticket)) != k:
            raise ValueError("每组号码必须等长且不重复。")
        if any(number < 1 or number > p.size for number in ticket):
            raise ValueError("号码超出概率向量的范围。")

    counts = np.zeros(p.size, dtype=np.int64)
    for ticket in rows:
        counts[np.asarray(ticket, dtype=int) - 1] += 1
    realized = counts.astype(float) / float(len(rows))
    delta = realized - p
    overlaps = [
        len(set(rows[i]).intersection(rows[j]))
        for i in range(len(rows))
        for j in range(i + 1, len(rows))
    ]
    return {
        "ticket_count": int(len(rows)),
        "sample_size": int(k),
        "unique_ticket_count": int(len(set(rows))),
        "number_coverage": int(np.count_nonzero(counts)),
        "max_pairwise_overlap": int(max(overlaps, default=0)),
        "mean_pairwise_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "expected_appearances": (float(len(rows)) * p).tolist(),
        "realized_appearances": counts.tolist(),
        "realized_batch_marginals": realized.tolist(),
        "mean_absolute_batch_marginal_error": float(np.mean(np.abs(delta))),
        "max_absolute_batch_marginal_error": float(np.max(np.abs(delta))),
        "batch_counts_are_floor_or_ceil": bool(
            np.all(
                (counts >= np.floor(len(rows) * p) - 1e-7)
                & (counts <= np.ceil(len(rows) * p) + 1e-7)
            )
        ),
        "interpretation": (
            "V8.1约束的是整批号码出现次数：每个号码的批次次数为目标期望次数的floor或ceil，"
            "并且跨随机种子的期望次数等于ticket_count乘以输入边际概率。"
            "单注边际分布不再相同；本批次的号码覆盖和重叠是组合结构指标，不能解释为单注中奖概率提升。"
        ),
    }


def generate_batch_quota_portfolio(
    probabilities: Iterable[float] | np.ndarray,
    ticket_count: int = 10,
    sample_size: int = 6,
    *,
    seed: int = 2028,
    max_allocation_attempts: int = 256,
) -> tuple[list[list[int]], dict[str, object]]:
    """Generate a unique fixed-size portfolio with unbiased batch-level quotas."""
    p = _validate_probabilities(probabilities, sample_size)
    m = int(ticket_count)
    k = int(sample_size)
    if not 1 <= m <= 1000:
        raise ValueError("ticket_count必须位于1至1000之间。")
    counts = dependent_round_appearances(
        p,
        ticket_count=m,
        sample_size=k,
        seed=int(seed),
    )
    rng = np.random.default_rng(int(seed) ^ 0x5A1781)
    tickets, allocation_attempts = _allocate_rows(
        counts,
        ticket_count=m,
        sample_size=k,
        rng=rng,
        max_attempts=int(max_allocation_attempts),
    )
    report = portfolio_diagnostics(tickets, p)
    report.update({
        "sampler_version": V81_PORTFOLIO_VERSION,
        "dependent_rounding_seed": int(seed),
        "allocation_seed": int(seed) ^ 0x5A1781,
        "allocation_attempts": int(allocation_attempts),
        "total_number_slots": m * k,
        "batch_expected_appearances_preserved": True,
        "per_ticket_marginals_identical": False,
        "production_model_changed": False,
        "prospective_lock_created": False,
    })
    return tickets, report
