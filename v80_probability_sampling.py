# -*- coding: utf-8 -*-
"""V8.0 probability-faithful fixed-size ticket sampling.

The V5 engine estimates first-order inclusion probabilities p_i for 45 numbers,
with sum(p_i) = 6. This module converts those marginals into a maximum-entropy
fixed-size distribution over six-number subsets (conditional Bernoulli /
rejective sampling). The resulting single-ticket distribution has exactly the
requested first-order inclusion probabilities, up to numerical tolerance.

This module intentionally does not apply pairwise-co-occurrence adjustments,
portfolio overlap caps, or diversity filters. Such post-selection constraints
change the joint distribution and can move the final batch away from the
requested marginals. Portfolio diversification is therefore a separately
reported decision, not part of this probability-faithful sampler.

No lottery data is downloaded or retrieved by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

V80_SAMPLER_VERSION = "8.0.0-conditional-bernoulli"


@dataclass(frozen=True)
class ConditionalBernoulliFit:
    """Calibrated odds for a fixed-size maximum-entropy subset distribution."""

    log_odds: np.ndarray
    target_marginals: np.ndarray
    fitted_marginals: np.ndarray
    sample_size: int
    iterations: int
    max_abs_error: float
    objective: float
    version: str = V80_SAMPLER_VERSION


def _validate_marginals(
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
        raise ValueError("probabilities必须是一维向量，且至少包含2个号码。")
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities包含非有限值。")
    k = int(sample_size)
    if not 1 <= k < p.size:
        raise ValueError("sample_size必须位于1与号码总数之间。")
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("每个号码的边际入选概率必须严格位于0与1之间。")
    total_error = abs(float(p.sum()) - k)
    if total_error > float(sum_tolerance):
        raise ValueError(
            f"边际入选概率之和必须为{k}；当前为{float(p.sum()):.10f}。"
        )
    # Remove only floating-point drift. This is not statistical calibration.
    p = p * (float(k) / float(p.sum()))
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("概率总和微调后出现边界值，无法建立固定样本量分布。")
    return p


def _log_elementary_symmetric(
    log_weights: np.ndarray,
    degree: int,
    excluded: tuple[int, ...] = (),
) -> float:
    """Return log(e_degree(exp(log_weights))) with optional items excluded."""
    degree = int(degree)
    if degree == 0:
        return 0.0
    if degree < 0:
        return -math.inf
    skip = set(excluded)
    dp = np.full(degree + 1, -math.inf, dtype=float)
    dp[0] = 0.0
    seen = 0
    for idx, log_weight in enumerate(log_weights):
        if idx in skip:
            continue
        seen += 1
        for d in range(min(degree, seen), 0, -1):
            dp[d] = np.logaddexp(dp[d], float(log_weight) + dp[d - 1])
    return float(dp[degree])


def _marginals_and_covariance(
    log_odds: np.ndarray,
    sample_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute exact inclusion marginals and their covariance under the design."""
    n = int(log_odds.size)
    k = int(sample_size)
    log_partition = _log_elementary_symmetric(log_odds, k)
    fitted = np.empty(n, dtype=float)
    for i in range(n):
        log_remainder = _log_elementary_symmetric(log_odds, k - 1, (i,))
        fitted[i] = math.exp(float(log_odds[i]) + log_remainder - log_partition)

    joint = np.diag(fitted.copy())
    for i in range(n):
        for j in range(i + 1, n):
            log_remainder = _log_elementary_symmetric(log_odds, k - 2, (i, j))
            probability = math.exp(
                float(log_odds[i]) + float(log_odds[j])
                + log_remainder - log_partition
            )
            joint[i, j] = probability
            joint[j, i] = probability
    covariance = joint - np.outer(fitted, fitted)
    covariance = (covariance + covariance.T) * 0.5
    return fitted, covariance, log_partition


def fit_conditional_bernoulli(
    probabilities: Iterable[float] | np.ndarray,
    sample_size: int = 6,
    *,
    tolerance: float = 1e-9,
    max_iter: int = 80,
) -> ConditionalBernoulliFit:
    """Fit conditional-Bernoulli odds that reproduce the requested marginals.

    For each six-number set S, the probability is proportional to the product
    of exp(log_odds[i]) over i in S. The odds are found by minimizing the
    convex dual of the fixed-size exponential family. A reduced Newton system
    fixes the otherwise non-identifiable common offset in log-odds.
    """
    p = _validate_marginals(probabilities, sample_size)
    n = int(p.size)
    k = int(sample_size)
    if not 1e-12 <= float(tolerance) <= 1e-3:
        raise ValueError("tolerance必须位于1e-12至1e-3之间。")
    if not 1 <= int(max_iter) <= 500:
        raise ValueError("max_iter必须位于1至500之间。")

    # Independent Bernoulli logits provide a close, scale-invariant starting point.
    theta = np.log(p / (1.0 - p))
    theta -= theta[-1]
    converged = False
    objective = math.inf
    iterations = 0

    for iterations in range(1, int(max_iter) + 1):
        fitted, covariance, log_partition = _marginals_and_covariance(theta, k)
        gradient = fitted - p
        objective = float(log_partition - np.dot(p, theta))
        max_error = float(np.max(np.abs(gradient)))
        if max_error <= float(tolerance):
            converged = True
            break

        # The covariance has a null direction along the all-ones vector because
        # every sample contains exactly k items. Fix the last log-odds to zero.
        reduced_hessian = covariance[:-1, :-1]
        reduced_gradient = gradient[:-1]
        regularizer = 1e-11
        try:
            step_direction = np.linalg.solve(
                reduced_hessian + np.eye(n - 1) * regularizer,
                reduced_gradient,
            )
        except np.linalg.LinAlgError:
            step_direction = np.linalg.lstsq(
                reduced_hessian + np.eye(n - 1) * 1e-8,
                reduced_gradient,
                rcond=None,
            )[0]

        # Backtracking guarantees descent of the convex dual objective.
        step_size = 1.0
        accepted = False
        for _ in range(40):
            candidate = theta.copy()
            candidate[:-1] -= step_size * step_direction
            candidate -= candidate[-1]
            candidate_objective = (
                _log_elementary_symmetric(candidate, k) - float(np.dot(p, candidate))
            )
            if np.isfinite(candidate_objective) and candidate_objective < objective - 1e-14:
                theta = candidate
                accepted = True
                break
            step_size *= 0.5

        if not accepted:
            # A conservative gradient step is a safe fallback for near-boundary
            # marginal vectors where the reduced Hessian can be ill-conditioned.
            step_direction = reduced_gradient
            step_size = 0.05
            for _ in range(40):
                candidate = theta.copy()
                candidate[:-1] -= step_size * step_direction
                candidate -= candidate[-1]
                candidate_objective = (
                    _log_elementary_symmetric(candidate, k) - float(np.dot(p, candidate))
                )
                if np.isfinite(candidate_objective) and candidate_objective < objective - 1e-14:
                    theta = candidate
                    accepted = True
                    break
                step_size *= 0.5
        if not accepted:
            break

    fitted, _, log_partition = _marginals_and_covariance(theta, k)
    final_error = float(np.max(np.abs(fitted - p)))
    if not converged and final_error > max(float(tolerance) * 10.0, 1e-7):
        raise RuntimeError(
            "固定样本量概率校准未收敛："
            f"最大边际误差={final_error:.3g}，迭代次数={iterations}。"
        )
    objective = float(log_partition - np.dot(p, theta))
    return ConditionalBernoulliFit(
        log_odds=theta.copy(),
        target_marginals=p.copy(),
        fitted_marginals=fitted.copy(),
        sample_size=k,
        iterations=iterations,
        max_abs_error=final_error,
        objective=objective,
    )


def _suffix_log_partitions(log_odds: np.ndarray, sample_size: int) -> np.ndarray:
    """suffix[i, r] is log e_r of items i..n-1."""
    n = int(log_odds.size)
    k = int(sample_size)
    suffix = np.full((n + 1, k + 1), -math.inf, dtype=float)
    suffix[:, 0] = 0.0
    for i in range(n - 1, -1, -1):
        for r in range(1, k + 1):
            suffix[i, r] = np.logaddexp(
                suffix[i + 1, r], float(log_odds[i]) + suffix[i + 1, r - 1]
            )
    return suffix


def _sample_one(
    log_odds: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    n = int(log_odds.size)
    k = int(sample_size)
    suffix = _suffix_log_partitions(log_odds, k)
    selected: list[int] = []
    remaining = k
    for i in range(n):
        if remaining == 0:
            break
        if n - i == remaining:
            selected.extend(range(i, n))
            remaining = 0
            break
        log_probability = (
            float(log_odds[i]) + suffix[i + 1, remaining - 1] - suffix[i, remaining]
        )
        include_probability = math.exp(min(0.0, float(log_probability)))
        if rng.random() < include_probability:
            selected.append(i)
            remaining -= 1
    if remaining != 0 or len(selected) != k:
        raise RuntimeError("条件伯努利抽样未能生成指定数量的号码。")
    return tuple(index + 1 for index in selected)


def sample_conditional_bernoulli(
    fitted: ConditionalBernoulliFit,
    count: int = 1,
    *,
    seed: int = 2028,
) -> list[list[int]]:
    """Draw independent fixed-size tickets from a fitted marginal distribution.

    Sampling is with replacement across tickets. Duplicate tickets are possible
    (though usually rare); avoiding duplicates or applying cross-ticket overlap
    constraints is a separate portfolio-selection operation that changes the
    batch distribution and must be audited separately.
    """
    if not isinstance(fitted, ConditionalBernoulliFit):
        raise TypeError("fitted必须由fit_conditional_bernoulli返回。")
    n = int(fitted.log_odds.size)
    k = int(fitted.sample_size)
    if not 1 <= int(count) <= 100_000:
        raise ValueError("count必须位于1至100000之间。")
    if not np.all(np.isfinite(fitted.log_odds)):
        raise ValueError("校准后的log_odds包含非有限值。")
    rng = np.random.default_rng(int(seed))
    return [list(_sample_one(fitted.log_odds, k, rng)) for _ in range(int(count))]


def summarize_ticket_batch(
    tickets: Iterable[Iterable[int]],
    target_marginals: Iterable[float] | np.ndarray,
) -> dict[str, object]:
    """Report realized batch inclusion rates and their deviation from targets.

    This report is descriptive. A small batch such as ten tickets has substantial
    sampling noise; it must not be used as evidence that the underlying forecast
    is miscalibrated or superior.
    """
    target = np.asarray(target_marginals, dtype=float).reshape(-1)
    if target.ndim != 1 or target.size < 2 or not np.all(np.isfinite(target)):
        raise ValueError("target_marginals必须是一维有限概率向量。")
    if np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("target_marginals必须位于0至1之间。")
    rows = [tuple(int(number) for number in ticket) for ticket in tickets]
    if not rows:
        raise ValueError("tickets不能为空。")
    k = len(rows[0])
    if k <= 0:
        raise ValueError("每组号码不能为空。")
    for ticket in rows:
        if len(ticket) != k or len(set(ticket)) != k:
            raise ValueError("每组号码必须等长且不重复。")
        if any(number < 1 or number > target.size for number in ticket):
            raise ValueError("号码超出target_marginals对应的范围。")
    counts = np.zeros(target.size, dtype=float)
    for ticket in rows:
        counts[np.asarray(ticket, dtype=int) - 1] += 1.0
    realized = counts / float(len(rows))
    difference = realized - target
    return {
        "ticket_count": int(len(rows)),
        "sample_size": int(k),
        "probability_sum": float(target.sum()),
        "realized_frequency_sum": float(realized.sum()),
        "mean_absolute_marginal_deviation": float(np.mean(np.abs(difference))),
        "root_mean_square_marginal_deviation": float(np.sqrt(np.mean(difference ** 2))),
        "max_absolute_marginal_deviation": float(np.max(np.abs(difference))),
        "target_marginals": target.tolist(),
        "realized_marginals": realized.tolist(),
        "marginal_differences": difference.tolist(),
        "interpretation": (
            "小批量组合的实际入选频率会有抽样波动；该指标用于描述组合批次，"
            "不能替代大样本抽样校准或前瞻预测检验。"
        ),
    }


def generate_probability_faithful_tickets(
    probabilities: Iterable[float] | np.ndarray,
    count: int = 10,
    *,
    seed: int = 2028,
    sample_size: int = 6,
) -> tuple[list[list[int]], ConditionalBernoulliFit, dict[str, object]]:
    """Fit and sample tickets while preserving the requested per-number marginals."""
    fitted = fit_conditional_bernoulli(probabilities, sample_size=sample_size)
    tickets = sample_conditional_bernoulli(fitted, count=count, seed=seed)
    batch_report = summarize_ticket_batch(tickets, fitted.target_marginals)
    return tickets, fitted, batch_report
