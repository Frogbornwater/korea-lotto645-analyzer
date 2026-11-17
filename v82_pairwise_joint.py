# -*- coding: utf-8 -*-
"""V8.2 pairwise joint-inclusion research for Lotto 6/45.

This is a research-only second-order probability diagnostic. It asks whether
historical pair co-occurrence contains out-of-sample information after the
first-order number marginals have been accounted for.

For a 6-of-45 draw, each number i has pair-degree 5*p_i, because whenever i
appears it must appear with exactly five other numbers. The baseline pair
matrix is the symmetric maximum-product matrix q_ij = r_i*r_j (i != j), where
r is solved so each row sums exactly to 5*p_i. A shrunk historical residual
pair effect tilts this baseline in log-odds space; symmetric node offsets are
then calibrated so all 45 row sums remain exactly 5*p_i.

The result is a coherent second-order pair-inclusion forecast (symmetric,
bounded, and with the correct first-order degree constraints). It is NOT yet
a complete joint distribution over all 6-number subsets and is not used to
generate production tickets. The rolling research compares pair Brier and
LogLoss against the no-interaction baseline and a label-permutation placebo.

All histories must be supplied locally by the caller. This module performs no
network requests and does not access lottery websites or APIs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

from lotto_core import NUMBER_COLS, validate_frame
from v50_predictive_engine import (
    BASE_RATE,
    _fit_from_feature_cache,
    build_number_features,
)

V82_VERSION = "8.2.0-pairwise-joint-inclusion"
PAIR_COUNT = math.comb(45, 2)
PAIR_PRIOR_DRAWS = 1000.0
PAIR_STRENGTH = 0.5
PAIR_RESIDUAL_CAP = 0.75
PAIR_FEATURE_EXCLUSIONS = ("pair_mean_log_lift", "pair_sd_log_lift")


@dataclass(frozen=True)
class PairResidualFit:
    """Shrunk historical pair effects after conditioning on number marginals."""

    residual_log_odds: np.ndarray
    historical_marginals: np.ndarray
    historical_pair_baseline: np.ndarray
    posterior_pair_probabilities: np.ndarray
    pair_counts: np.ndarray
    training_draws: int
    prior_draws: float
    version: str = V82_VERSION


def _validate_marginals(probabilities: np.ndarray, sample_size: int = 6) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or p.size != 45:
        raise ValueError("45号码边际概率必须是一维向量。")
    if not np.all(np.isfinite(p)) or np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("45号码边际概率必须是0与1之间的有限值。")
    if abs(float(p.sum()) - float(sample_size)) > 1e-5:
        raise ValueError(f"45个边际概率之和必须为{sample_size}。")
    return p * (float(sample_size) / float(p.sum()))


def _logit(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return np.log(np.clip(x, 1e-12, 1.0 - 1e-12) / np.clip(1.0 - x, 1e-12, 1.0))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def pair_baseline_from_marginals(
    probabilities: np.ndarray,
    sample_size: int = 6,
) -> np.ndarray:
    """Return symmetric pair probabilities with row sums (k-1)*p_i.

    q_ij = r_i*r_j is the maximum-product pair matrix under the exact row
    constraints, with q_ii=0. It captures only first-order number information;
    no historical pair-specific effect is used.
    """
    p = _validate_marginals(probabilities, sample_size)
    target = float(sample_size - 1) * p
    max_target = float(target.max())
    lower_sum = 2.0 * math.sqrt(max_target) * (1.0 + 1e-12)

    def r_for_total(total: float) -> np.ndarray:
        discriminant = np.maximum(total * total - 4.0 * target, 0.0)
        # Stable quadratic root for r_i*(total-r_i)=target_i.
        return 2.0 * target / (total + np.sqrt(discriminant))

    def residual(total: float) -> float:
        return float(r_for_total(total).sum() - total)

    low = lower_sum
    if residual(low) <= 0.0:
        raise ValueError("输入边际概率过度集中，无法建立基准两两概率矩阵。")
    high = max(low * 2.0, 1.0)
    while residual(high) > 0.0:
        high *= 2.0
        if high > 1e8:
            raise RuntimeError("基准两两概率矩阵求解失败。")
    for _ in range(100):
        middle = (low + high) / 2.0
        if residual(middle) > 0.0:
            low = middle
        else:
            high = middle
    r = r_for_total((low + high) / 2.0)
    pair = np.outer(r, r)
    np.fill_diagonal(pair, 0.0)

    if (
        not np.all(np.isfinite(pair))
        or np.any(pair < 0.0)
        or np.any(pair >= 1.0)
        or not np.allclose(pair, pair.T, atol=1e-12)
        or float(np.max(np.abs(pair.sum(axis=1) - target))) > 1e-8
    ):
        raise RuntimeError("基准两两概率矩阵未满足对称性或一阶约束。")
    if np.any(pair - np.minimum.outer(p, p) > 1e-8):
        raise RuntimeError("基准两两概率违反单号概率上界。")
    return pair


def _clean_history(history: pd.DataFrame | np.ndarray | list[list[int]]) -> np.ndarray:
    if isinstance(history, pd.DataFrame):
        clean = validate_frame(history)
        if clean.empty:
            raise ValueError("没有有效历史数据。")
        matrix = clean[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)
    else:
        matrix = np.asarray(history, dtype=np.int16)
    if matrix.ndim != 2 or matrix.shape[1] != 6 or len(matrix) < 30:
        raise ValueError("历史矩阵必须至少包含30期、每期6个号码。")
    if np.any(matrix < 1) or np.any(matrix > 45):
        raise ValueError("历史号码必须位于1—45。")
    if any(len(set(int(x) for x in row)) != 6 for row in matrix):
        raise ValueError("每期开奖必须包含6个互不相同的号码。")
    return np.sort(matrix, axis=1)


def _pair_counts(matrix: np.ndarray) -> np.ndarray:
    counts = np.zeros((45, 45), dtype=float)
    for row in matrix:
        indices = row.astype(int) - 1
        counts[np.ix_(indices, indices)] += 1.0
    np.fill_diagonal(counts, 0.0)
    return counts


def fit_pair_residuals(
    history: pd.DataFrame | np.ndarray | list[list[int]],
    *,
    prior_draws: float = PAIR_PRIOR_DRAWS,
    residual_cap: float = PAIR_RESIDUAL_CAP,
) -> PairResidualFit:
    """Estimate pair-specific residual log-odds beyond empirical single rates.

    The empirical number marginals define a no-interaction pair baseline. The
    observed pair-count matrix and that baseline both have row sums 5*p_i.
    A Beta-style shrinkage prior pulls sparse pair estimates toward the
    no-interaction baseline before converting them to residual log-odds.
    """
    matrix = _clean_history(history)
    n_draws = int(len(matrix))
    prior = float(prior_draws)
    cap = float(residual_cap)
    if not np.isfinite(prior) or prior <= 0.0:
        raise ValueError("prior_draws必须是正的有限值。")
    if not np.isfinite(cap) or not 0.0 < cap <= 3.0:
        raise ValueError("residual_cap必须位于(0, 3]。")

    number_counts = np.bincount(matrix.reshape(-1), minlength=46)[1:46].astype(float)
    historical_marginals = number_counts / float(n_draws)
    if np.any(historical_marginals <= 0.0) or np.any(historical_marginals >= 1.0):
        raise ValueError("训练窗口必须让45个号码都至少出现一次。")

    baseline = pair_baseline_from_marginals(historical_marginals)
    observed_counts = _pair_counts(matrix)
    posterior = (observed_counts + prior * baseline) / (float(n_draws) + prior)
    residual = _logit(posterior) - _logit(baseline)
    residual = np.clip(residual, -cap, cap)
    np.fill_diagonal(residual, 0.0)

    expected_degree = 5.0 * historical_marginals
    if (
        not np.allclose(observed_counts, observed_counts.T, atol=0.0)
        or not np.allclose(posterior, posterior.T, atol=1e-12)
        or float(np.max(np.abs(observed_counts.sum(axis=1) / n_draws - expected_degree))) > 1e-10
        or float(np.max(np.abs(posterior.sum(axis=1) - expected_degree))) > 1e-8
        or not np.all(np.isfinite(residual))
    ):
        raise RuntimeError("历史两两残差估计未能保持一阶边际约束。")

    return PairResidualFit(
        residual_log_odds=residual,
        historical_marginals=historical_marginals,
        historical_pair_baseline=baseline,
        posterior_pair_probabilities=posterior,
        pair_counts=observed_counts,
        training_draws=n_draws,
        prior_draws=prior,
    )


def forecast_pair_probabilities(
    probabilities: np.ndarray,
    residual_log_odds: np.ndarray | PairResidualFit,
    *,
    strength: float = PAIR_STRENGTH,
    tolerance: float = 1e-8,
    max_sweeps: int = 160,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Tilt the pair baseline by historical residual effects and recalibrate.

    A symmetric node-offset model q_ij = sigmoid(logit(q0_ij)+s*delta_ij+a_i+a_j)
    is fit so each row sums exactly to 5*p_i. Bounds enforce q_ij <= min(p_i,p_j)
    and the Fréchet lower bound. Thus the pair forecast cannot improve one
    pair by silently changing a number's first-order inclusion probability.
    """
    p = _validate_marginals(probabilities)
    if isinstance(residual_log_odds, PairResidualFit):
        delta = np.asarray(residual_log_odds.residual_log_odds, dtype=float)
    else:
        delta = np.asarray(residual_log_odds, dtype=float)
    if delta.shape != (45, 45) or not np.all(np.isfinite(delta)):
        raise ValueError("pair residual必须是45×45有限矩阵。")
    if not np.allclose(delta, delta.T, atol=1e-10):
        raise ValueError("pair residual必须对称。")
    if not np.allclose(np.diag(delta), 0.0, atol=1e-10):
        raise ValueError("pair residual对角线必须为0。")
    strength = float(strength)
    if not 0.0 <= strength <= 2.0:
        raise ValueError("strength必须位于0—2。")
    if not 1e-12 <= float(tolerance) <= 1e-4:
        raise ValueError("tolerance必须位于1e-12至1e-4之间。")

    baseline = pair_baseline_from_marginals(p)
    if strength == 0.0:
        return baseline.copy(), baseline, {
            "max_row_sum_error": 0.0,
            "iterations": 0.0,
            "pair_probability_sum": float(baseline.sum() / 2.0),
        }

    n = 45
    target_rows = 5.0 * p
    upper = np.minimum.outer(p, p) - 1e-12
    lower = np.maximum(p[:, None] + p[None, :] - 1.0, 0.0) + 1e-12
    np.fill_diagonal(upper, 0.0)
    np.fill_diagonal(lower, 0.0)
    log_weights = _logit(np.maximum(baseline, 1e-12)) + strength * delta
    np.fill_diagonal(log_weights, 0.0)
    offsets = np.zeros(n, dtype=float)

    converged = False
    max_error = math.inf
    for sweep in range(1, int(max_sweeps) + 1):
        for i in range(n):
            mask = np.arange(n) != i
            z = log_weights[i, mask] + offsets[i] + offsets[mask]
            raw = _sigmoid(z)
            row = np.clip(raw, lower[i, mask], upper[i, mask])
            free = (raw > lower[i, mask] + 1e-14) & (raw < upper[i, mask] - 1e-14)
            derivative = float(np.sum((row * (1.0 - row))[free]))
            error = float(target_rows[i] - row.sum())
            if derivative > 1e-12:
                offsets[i] += float(np.clip(error / derivative, -2.0, 2.0))

        z = log_weights + offsets[:, None] + offsets[None, :]
        pair = np.clip(_sigmoid(z), lower, upper)
        np.fill_diagonal(pair, 0.0)
        max_error = float(np.max(np.abs(pair.sum(axis=1) - target_rows)))
        if max_error <= float(tolerance):
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"两两联合概率边际校准未收敛：最大行和误差={max_error:.3g}。"
        )
    if (
        not np.allclose(pair, pair.T, atol=1e-10)
        or np.any(pair < 0.0)
        or np.any(pair > upper + 1e-9)
        or float(abs(pair.sum() / 2.0 - 15.0)) > 1e-7
    ):
        raise RuntimeError("两两联合概率未满足对称性、概率边界或总对数约束。")

    return pair, baseline, {
        "max_row_sum_error": max_error,
        "iterations": float(sweep),
        "pair_probability_sum": float(pair.sum() / 2.0),
    }


def permute_pair_labels(
    residual_log_odds: np.ndarray | PairResidualFit,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a label-permutation placebo preserving the learned effect matrix."""
    delta = (
        residual_log_odds.residual_log_odds
        if isinstance(residual_log_odds, PairResidualFit)
        else np.asarray(residual_log_odds, dtype=float)
    )
    if delta.shape != (45, 45):
        raise ValueError("residual_log_odds必须是45×45矩阵。")
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(45)
    placebo = delta[np.ix_(permutation, permutation)].copy()
    np.fill_diagonal(placebo, 0.0)
    return placebo, permutation


def score_pair_probabilities(
    pair_probabilities: np.ndarray,
    actual_numbers: list[int] | tuple[int, ...] | set[int],
) -> dict[str, float]:
    """Score all 990 unordered pair-inclusion events for one target draw."""
    q = np.asarray(pair_probabilities, dtype=float)
    if q.shape != (45, 45) or not np.all(np.isfinite(q)):
        raise ValueError("pair_probabilities必须是45×45有限矩阵。")
    if not np.allclose(q, q.T, atol=1e-9) or not np.allclose(np.diag(q), 0.0, atol=1e-9):
        raise ValueError("两两概率矩阵必须对称且对角线为0。")
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("两两概率必须位于0—1。")
    actual = sorted(set(int(x) for x in actual_numbers))
    if len(actual) != 6 or any(x < 1 or x > 45 for x in actual):
        raise ValueError("actual_numbers必须包含6个不同的1—45号码。")
    i, j = np.triu_indices(45, 1)
    predicted = np.clip(q[i, j], 1e-12, 1.0 - 1e-12)
    selected = set(actual)
    observed = np.fromiter(
        (1.0 if (int(a) + 1 in selected and int(b) + 1 in selected) else 0.0 for a, b in zip(i, j)),
        dtype=float,
        count=PAIR_COUNT,
    )
    brier = float(np.mean((predicted - observed) ** 2))
    log_loss = float(np.mean(
        -(observed * np.log(predicted) + (1.0 - observed) * np.log(1.0 - predicted))
    ))
    positive_log_loss = float(-np.log(predicted[observed > 0.5]).mean())
    return {
        "pair_brier": brier,
        "pair_log_loss": log_loss,
        "positive_pair_log_loss": positive_log_loss,
        "observed_pairs": int(observed.sum()),
        "forecast_pair_sum": float(q.sum() / 2.0),
    }


def _moving_block_bootstrap_ci(
    values: np.ndarray,
    *,
    repetitions: int,
    block_length: int,
    seed: int,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float).reshape(-1)
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    block = max(1, min(int(block_length), n))
    reps = max(100, int(repetitions))
    rng = np.random.default_rng(int(seed))
    blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(reps, blocks))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    samples = x[indices.reshape(reps, -1)[:, :n]]
    means = samples.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def evaluate_v82_pairwise_research(
    frame: pd.DataFrame,
    *,
    test_draws: int = 500,
    training_window: int = 500,
    min_history: int = 180,
    ridge_alpha: float = 1.0,
    pair_prior_draws: float = PAIR_PRIOR_DRAWS,
    pair_strength: float = PAIR_STRENGTH,
    bootstrap_repetitions: int = 10000,
    block_length: int = 10,
    seed: int = 8202028,
) -> dict[str, Any]:
    """Run strict one-step-ahead pair-probability research using past-only data."""
    clean = validate_frame(frame).reset_index(drop=True)
    if clean.empty:
        raise ValueError("没有有效历史数据。")
    draw_numbers = clean["draw_no"].to_numpy(dtype=np.int64)
    if np.any(np.diff(draw_numbers) != 1):
        raise ValueError("V8.2滚动研究要求连续期号。")
    matrix = clean[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)
    n = len(matrix)
    test_draws = int(test_draws)
    training_window = int(training_window)
    min_history = int(min_history)
    if not 30 <= min_history < n or not 30 <= training_window <= 1000:
        raise ValueError("min_history或training_window超出允许范围。")
    if test_draws < 2:
        raise ValueError("test_draws至少为2。")
    if not np.isfinite(float(ridge_alpha)) or not 0.0 < float(ridge_alpha) <= 100.0:
        raise ValueError("ridge_alpha必须位于(0,100]。")
    if not 0.0 <= float(pair_strength) <= 2.0:
        raise ValueError("pair_strength必须位于0—2。")

    start_idx = max(min_history, n - test_draws)
    if n - start_idx < 2:
        raise ValueError("可用的样本外目标期不足。")

    # Pair summaries are excluded from V5 marginal features to isolate the
    # incremental value of the V8.2 joint-pair residual layer.
    feature_cache = {
        idx: build_number_features(matrix[:idx])
        for idx in range(min_history, n)
    }

    records: list[dict[str, Any]] = []
    for target_idx in range(start_idx, n):
        probabilities, _, diagnostics = _fit_from_feature_cache(
            matrix,
            feature_cache,
            target_idx,
            training_window,
            float(ridge_alpha),
            PAIR_FEATURE_EXCLUSIONS,
        )
        history_start = max(0, target_idx - training_window)
        past = matrix[history_start:target_idx]
        pair_fit = fit_pair_residuals(past, prior_draws=pair_prior_draws)
        adjusted, baseline, adjustment_diag = forecast_pair_probabilities(
            probabilities,
            pair_fit,
            strength=float(pair_strength),
        )

        target_draw = int(draw_numbers[target_idx])
        placebo_residual, permutation = permute_pair_labels(
            pair_fit,
            seed=int(np.random.SeedSequence([int(seed), target_draw]).generate_state(1)[0]),
        )
        placebo, _, placebo_diag = forecast_pair_probabilities(
            probabilities,
            placebo_residual,
            strength=float(pair_strength),
        )
        actual = matrix[target_idx].astype(int).tolist()

        baseline_score = score_pair_probabilities(baseline, actual)
        adjusted_score = score_pair_probabilities(adjusted, actual)
        placebo_score = score_pair_probabilities(placebo, actual)

        actual_vector = np.isin(np.arange(1, 46), actual).astype(float)
        marginal_brier = float(np.mean((probabilities - actual_vector) ** 2))
        clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
        marginal_log_loss = float(-np.mean(
            actual_vector * np.log(clipped) + (1.0 - actual_vector) * np.log(1.0 - clipped)
        ))

        records.append({
            "target_draw": target_draw,
            "training_end_draw": int(draw_numbers[target_idx - 1]),
            "training_draws": int(diagnostics["training_draws"]),
            "pair_training_draws": int(pair_fit.training_draws),
            "v5_marginal_brier": marginal_brier,
            "v5_marginal_log_loss": marginal_log_loss,
            "baseline_pair_brier": baseline_score["pair_brier"],
            "adjusted_pair_brier": adjusted_score["pair_brier"],
            "placebo_pair_brier": placebo_score["pair_brier"],
            "baseline_pair_log_loss": baseline_score["pair_log_loss"],
            "adjusted_pair_log_loss": adjusted_score["pair_log_loss"],
            "placebo_pair_log_loss": placebo_score["pair_log_loss"],
            "baseline_positive_pair_log_loss": baseline_score["positive_pair_log_loss"],
            "adjusted_positive_pair_log_loss": adjusted_score["positive_pair_log_loss"],
            "placebo_positive_pair_log_loss": placebo_score["positive_pair_log_loss"],
            "pair_brier_gain": baseline_score["pair_brier"] - adjusted_score["pair_brier"],
            "pair_log_loss_gain": baseline_score["pair_log_loss"] - adjusted_score["pair_log_loss"],
            "placebo_pair_brier_gain": baseline_score["pair_brier"] - placebo_score["pair_brier"],
            "placebo_pair_log_loss_gain": baseline_score["pair_log_loss"] - placebo_score["pair_log_loss"],
            "pair_adjustment_row_error": adjustment_diag["max_row_sum_error"],
            "placebo_adjustment_row_error": placebo_diag["max_row_sum_error"],
            "observed_pairs": adjusted_score["observed_pairs"],
            "history_fingerprint_sha256": hashlib.sha256(past.tobytes()).hexdigest(),
        })

    detail = pd.DataFrame(records)
    metric_pairs = [
        ("pair_brier_gain", "pair_brier"),
        ("pair_log_loss_gain", "pair_log_loss"),
        ("placebo_pair_brier_gain", "placebo_pair_brier"),
        ("placebo_pair_log_loss_gain", "placebo_pair_log_loss"),
    ]
    comparisons: list[dict[str, Any]] = []
    for offset, (gain_column, score_name) in enumerate(metric_pairs):
        values = detail[gain_column].to_numpy(dtype=float)
        low, high = _moving_block_bootstrap_ci(
            values,
            repetitions=bootstrap_repetitions,
            block_length=block_length,
            seed=int(seed) + 101 + offset,
        )
        comparisons.append({
            "comparison": gain_column,
            "metric": score_name,
            "mean_gain_positive_is_better": float(values.mean()),
            "moving_block_bootstrap_95ci_low": low,
            "moving_block_bootstrap_95ci_high": high,
            "scored_draws": int(len(values)),
        })

    block_reports = []
    block_size = 100
    for block_idx, start in enumerate(range(0, len(detail), block_size), start=1):
        subset = detail.iloc[start:start + block_size]
        block_reports.append({
            "block": block_idx,
            "target_range": [
                int(subset["target_draw"].min()),
                int(subset["target_draw"].max()),
            ],
            "pair_brier_gain": float(subset["pair_brier_gain"].mean()),
            "pair_log_loss_gain": float(subset["pair_log_loss_gain"].mean()),
            "placebo_pair_brier_gain": float(subset["placebo_pair_brier_gain"].mean()),
            "placebo_pair_log_loss_gain": float(subset["placebo_pair_log_loss_gain"].mean()),
        })

    comparison_lookup = {item["comparison"]: item for item in comparisons}
    brier_gate = comparison_lookup["pair_brier_gain"]["moving_block_bootstrap_95ci_low"] > 0.0
    log_gate = comparison_lookup["pair_log_loss_gain"]["moving_block_bootstrap_95ci_low"] > 0.0
    placebo_brier_gate = comparison_lookup["placebo_pair_brier_gain"]["mean_gain_positive_is_better"] <= 0.0
    placebo_log_gate = comparison_lookup["placebo_pair_log_loss_gain"]["mean_gain_positive_is_better"] <= 0.0
    enough_draws = len(detail) >= 200
    signal_supported = bool(
        enough_draws and brier_gate and log_gate
        and placebo_brier_gate and placebo_log_gate
    )

    base_pair_uniform = 1.0 / 66.0
    all_pairs = PAIR_COUNT
    positive_fraction = 15.0 / float(all_pairs)
    uniform_pair_brier = (
        positive_fraction * (1.0 - base_pair_uniform) ** 2
        + (1.0 - positive_fraction) * base_pair_uniform ** 2
    )
    uniform_pair_log_loss = -(
        positive_fraction * math.log(base_pair_uniform)
        + (1.0 - positive_fraction) * math.log(1.0 - base_pair_uniform)
    )

    return {
        "research_id": "lotto645-v82-pairwise-joint-20261118",
        "engine_version": V82_VERSION,
        "status": "historical_pairwise_screen_not_independent_or_prospective_evidence",
        "research_only": True,
        "test_draws": int(len(detail)),
        "target_start_draw": int(detail["target_draw"].min()),
        "target_end_draw": int(detail["target_draw"].max()),
        "training_window": training_window,
        "min_history": min_history,
        "ridge_alpha": float(ridge_alpha),
        "v5_pair_features_excluded": list(PAIR_FEATURE_EXCLUSIONS),
        "pair_prior_draws": float(pair_prior_draws),
        "pair_strength": float(pair_strength),
        "pair_residual_cap": PAIR_RESIDUAL_CAP,
        "pair_count_per_draw": 15,
        "pair_event_count_per_draw": PAIR_COUNT,
        "uniform_pair_baseline": {
            "pair_probability": base_pair_uniform,
            "pair_brier": uniform_pair_brier,
            "pair_log_loss": uniform_pair_log_loss,
        },
        "mean_scores": {
            "v5_marginal_brier": float(detail["v5_marginal_brier"].mean()),
            "v5_marginal_log_loss": float(detail["v5_marginal_log_loss"].mean()),
            "baseline_pair_brier": float(detail["baseline_pair_brier"].mean()),
            "adjusted_pair_brier": float(detail["adjusted_pair_brier"].mean()),
            "placebo_pair_brier": float(detail["placebo_pair_brier"].mean()),
            "baseline_pair_log_loss": float(detail["baseline_pair_log_loss"].mean()),
            "adjusted_pair_log_loss": float(detail["adjusted_pair_log_loss"].mean()),
            "placebo_pair_log_loss": float(detail["placebo_pair_log_loss"].mean()),
            "baseline_positive_pair_log_loss": float(detail["baseline_positive_pair_log_loss"].mean()),
            "adjusted_positive_pair_log_loss": float(detail["adjusted_positive_pair_log_loss"].mean()),
            "placebo_positive_pair_log_loss": float(detail["placebo_positive_pair_log_loss"].mean()),
        },
        "comparisons": comparisons,
        "block_reports": block_reports,
        "evidence_gate": {
            "minimum_draws_required": 200,
            "enough_draws": bool(enough_draws),
            "pair_brier_ci_excludes_zero_in_favorable_direction": bool(brier_gate),
            "pair_log_loss_ci_excludes_zero_in_favorable_direction": bool(log_gate),
            "placebo_pair_brier_mean_gain_nonpositive": bool(placebo_brier_gate),
            "placebo_pair_log_loss_mean_gain_nonpositive": bool(placebo_log_gate),
            "pairwise_incremental_signal_supported": signal_supported,
            "second_independent_replication_allowed": signal_supported,
            "production_model_change_allowed": False,
            "prediction_superiority_proven": False,
        },
        "data_fingerprint_sha256": hashlib.sha256(
            matrix[max(0, n - training_window):].tobytes()
        ).hexdigest(),
        "limitations": [
            "所有目标期的45维一阶概率和历史两两残差均只使用目标期之前的数据。",
            "V5的一阶预测特征明确排除pair_mean_log_lift和pair_sd_log_lift，以隔离第二阶增量信息。",
            "两两概率矩阵严格保持对称、0对角线、单号上界和每行5*p_i约束；但它不是完整的六号码联合分布。",
            "当前阶段只评分990个号码对，不把两两预测直接转换成正式投注号码。",
            "先验强度、交互强度和残差上限在研究前固定；本报告不进行测试区间内参数寻优。",
            "随机置换号码标签用于负对照；它不能代替新的独立前瞻验证。",
            "历史研究结果不能证明未来开奖具有可预测性，也不允许自动修改正式选号。",
            "所有数据必须来自用户手动维护的CSV/Excel；本研究不访问任何外部开奖网站或API。",
        ],
    }
