# -*- coding: utf-8 -*-
"""V5.0 regularized walk-forward probability model for Lotto 6/45.

This module is deliberately offline. It accepts only a history frame supplied
by the caller; it never retrieves lottery results or accesses external services.

The model estimates marginal number-inclusion probabilities with a ridge
logistic model trained on strictly pre-draw features. Pairwise co-occurrence
terms are strongly shrunk and used only as a modest sequential ticket-sampling
adjustment. Neither output is a guarantee of future advantage.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lotto_core import NUMBER_COLS, validate_frame

V50_ENGINE_VERSION = "5.0.0-predictive-research"
MODEL_NAME = "V5联合概率模型"
BASE_RATE = 6.0 / 45.0
PAIR_BASE_RATE = 1.0 / 66.0
FEATURE_NAMES = (
    "recent_30_z",
    "recent_60_z",
    "recent_120_z",
    "recent_240_z",
    "long_frequency_z",
    "trend_30_minus_120",
    "ewma_35_z",
    "gap_log_scaled",
    "pair_mean_log_lift",
    "pair_sd_log_lift",
)
_PAIR_I, _PAIR_J = np.triu_indices(6, 1)


def _clean_matrix(history: pd.DataFrame | np.ndarray | list[list[int]]) -> np.ndarray:
    if isinstance(history, pd.DataFrame):
        clean = validate_frame(history)
        if clean.empty:
            raise ValueError("没有可用历史数据。请手动导入 CSV 或 Excel。")
        matrix = clean[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)
    else:
        matrix = np.asarray(history, dtype=np.int16)
    if matrix.ndim != 2 or matrix.shape[1] != 6:
        raise ValueError("历史号码矩阵必须是 N×6。")
    if len(matrix) == 0:
        raise ValueError("没有可用历史数据。")
    if np.any(matrix < 1) or np.any(matrix > 45):
        raise ValueError("历史主号码必须全部位于1—45。")
    if any(len(set(int(x) for x in row)) != 6 for row in matrix):
        raise ValueError("每期开奖必须包含6个互不相同的主号码。")
    return np.sort(matrix, axis=1)


def _frequency_z(matrix: np.ndarray, window: int, prior_draws: float = 20.0) -> np.ndarray:
    sample = matrix[-min(int(window), len(matrix)):]
    counts = np.bincount(sample.reshape(-1), minlength=46)[1:46].astype(float)
    effective = float(len(sample)) + float(prior_draws)
    rate = (counts + float(prior_draws) * BASE_RATE) / max(effective, 1.0)
    standard_error = np.sqrt(BASE_RATE * (1.0 - BASE_RATE) / max(effective, 1.0))
    return np.clip((rate - BASE_RATE) / max(standard_error, 1e-9), -5.0, 5.0)


def _pair_log_lift(matrix: np.ndarray, prior_draws: float = 250.0) -> np.ndarray:
    """Posterior-shrunk pair log-lift relative to uniform 6-of-45 sampling."""
    ids = (matrix[:, _PAIR_I] - 1) * 45 + (matrix[:, _PAIR_J] - 1)
    upper = np.bincount(ids.reshape(-1), minlength=45 * 45).reshape(45, 45).astype(float)
    counts = upper + upper.T
    posterior_rate = (counts + float(prior_draws) * PAIR_BASE_RATE) / (len(matrix) + float(prior_draws))
    result = np.log(np.maximum(posterior_rate, 1e-12) / PAIR_BASE_RATE)
    np.fill_diagonal(result, 0.0)
    return np.clip(result, -0.8, 0.8)


def build_number_features(
    history: pd.DataFrame | np.ndarray | list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build one row per number using only the supplied pre-target history.

    Returns (45 x feature_count matrix, 45 x 45 shrunk pair interaction matrix).
    """
    matrix = _clean_matrix(history)
    n = len(matrix)
    z30 = _frequency_z(matrix, 30, 20.0)
    z60 = _frequency_z(matrix, 60, 25.0)
    z120 = _frequency_z(matrix, 120, 35.0)
    z240 = _frequency_z(matrix, 240, 50.0)
    zlong = _frequency_z(matrix, n, 80.0)

    ages = np.arange(n - 1, -1, -1, dtype=float)
    decay = np.exp(-np.log(2.0) * ages / 35.0)
    flat_numbers = matrix.reshape(-1)
    weighted_counts = np.bincount(
        flat_numbers,
        weights=np.repeat(decay, 6),
        minlength=46,
    )[1:46]
    weight_sum = float(decay.sum())
    effective_n = weight_sum * weight_sum / max(float(np.square(decay).sum()), 1e-12)
    ewma_rate = (weighted_counts + 20.0 * BASE_RATE) / (weight_sum + 20.0)
    ewma_se = np.sqrt(BASE_RATE * (1.0 - BASE_RATE) / max(effective_n + 20.0, 1.0))
    ewma_z = np.clip((ewma_rate - BASE_RATE) / max(ewma_se, 1e-9), -5.0, 5.0)

    last_seen = np.full(45, -1, dtype=np.int32)
    for idx, row in enumerate(matrix):
        last_seen[row - 1] = idx
    gaps = np.where(last_seen < 0, n, n - 1 - last_seen).astype(float)
    gap_scaled = np.log1p(gaps) / max(np.log1p(float(n)), 1e-9)

    pair_scores = _pair_log_lift(matrix)
    pair_mean = pair_scores.sum(axis=1) / 44.0
    pair_sd = pair_scores.std(axis=1)

    features = np.column_stack([
        z30, z60, z120, z240, zlong, z30 - z120,
        ewma_z, gap_scaled, pair_mean, pair_sd,
    ]).astype(float)
    if features.shape != (45, len(FEATURE_NAMES)) or not np.all(np.isfinite(features)):
        raise RuntimeError("V5特征生成出现维度错误或非有限值。")
    return features, pair_scores


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_ridge_logistic(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    max_iter: int = 35,
) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(y) < 45:
        raise ValueError("V5训练样本维度不足或不匹配。")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("V5训练数据包含非有限值。")
    if not 0.0 < float(alpha) <= 100.0:
        raise ValueError("ridge_alpha必须位于(0, 100]。")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = np.clip((x - mean) / scale, -8.0, 8.0)
    design = np.column_stack([np.ones(len(x), dtype=float), standardized])
    beta = np.zeros(design.shape[1], dtype=float)
    beta[0] = np.log(BASE_RATE / (1.0 - BASE_RATE))
    penalty = np.eye(design.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    n_rows = float(len(y))
    iterations = 0

    for iterations in range(1, int(max_iter) + 1):
        pred = _sigmoid(design @ beta)
        weights = np.maximum(pred * (1.0 - pred), 1e-6)
        gradient = (design.T @ (pred - y)) / n_rows + (penalty @ beta) / n_rows
        hessian = (design.T * weights) @ design / n_rows + penalty / n_rows
        hessian.flat[:: hessian.shape[0] + 1] += 1e-9
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta -= step
        if float(np.linalg.norm(step)) < 1e-7:
            break

    return {
        "beta": beta,
        "mean": mean,
        "scale": scale,
        "iterations": iterations,
        "training_rows": int(len(y)),
        "training_positive_rate": float(y.mean()),
        "ridge_alpha": float(alpha),
    }


def _calibrate_to_six(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.shape != (45,) or not np.all(np.isfinite(scores)):
        raise ValueError("预测分数必须包含45个有限值。")
    low, high = -30.0, 30.0
    for _ in range(90):
        mid = (low + high) / 2.0
        total = float(_sigmoid(scores + mid).sum())
        if total > 6.0:
            high = mid
        else:
            low = mid
    probs = _sigmoid(scores + (low + high) / 2.0)
    # Remove insignificant floating point drift while preserving positivity.
    probs *= 6.0 / float(probs.sum())
    return np.clip(probs, 1e-9, 1.0 - 1e-9)


def _fit_from_feature_cache(
    matrix: np.ndarray,
    feature_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    target_idx: int,
    training_window: int,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    first_train = max(min(feature_cache), int(target_idx) - int(training_window))
    train_indices = list(range(first_train, int(target_idx)))
    if len(train_indices) < 30:
        raise ValueError("V5可用的时间顺序训练样本不足。")
    x_train = np.vstack([feature_cache[i][0] for i in train_indices])
    y_train = np.concatenate([
        np.isin(np.arange(1, 46), matrix[i]).astype(float)
        for i in train_indices
    ])
    fitted = _fit_ridge_logistic(x_train, y_train, alpha=ridge_alpha)
    target_features, pair_scores = feature_cache[int(target_idx)]
    z = np.clip((target_features - fitted["mean"]) / fitted["scale"], -8.0, 8.0)
    raw_scores = fitted["beta"][0] + z @ fitted["beta"][1:]
    probabilities = _calibrate_to_six(raw_scores)
    diagnostics = {
        "training_draws": len(train_indices),
        "training_rows": fitted["training_rows"],
        "training_positive_rate": fitted["training_positive_rate"],
        "ridge_alpha": fitted["ridge_alpha"],
        "fit_iterations": fitted["iterations"],
        "feature_coefficients": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, fitted["beta"][1:])
        },
    }
    return probabilities, pair_scores, diagnostics


def fit_predict_next(
    history: pd.DataFrame,
    training_window: int = 500,
    min_history: int = 180,
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    """Fit the V5 model on prequential historical feature rows and predict next draw."""
    clean = validate_frame(history)
    if clean.empty:
        raise ValueError("没有可用历史数据。")
    draw_numbers = clean["draw_no"].to_numpy(dtype=np.int64)
    if len(draw_numbers) < int(min_history) + 30:
        raise ValueError(f"V5至少需要{int(min_history) + 30}期连续历史记录。")
    if np.any(np.diff(draw_numbers) != 1):
        raise ValueError("历史期号存在缺失，不能运行V5时间顺序预测。")
    training_window = int(training_window)
    if not 30 <= training_window <= 1000:
        raise ValueError("training_window必须位于30—1000。")
    matrix = clean[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)
    n = len(matrix)
    cache_start = max(int(min_history), n - training_window)
    cache = {
        idx: build_number_features(matrix[:idx])
        for idx in range(cache_start, n + 1)
    }
    probabilities, pair_scores, diagnostics = _fit_from_feature_cache(
        matrix, cache, n, training_window, ridge_alpha
    )
    probability_frame = pd.DataFrame({
        "号码": np.arange(1, 46, dtype=int),
        "预测入选概率": probabilities,
        "均匀随机基准概率": np.full(45, BASE_RATE, dtype=float),
        "相对基准差": probabilities - BASE_RATE,
    }).sort_values(["预测入选概率", "号码"], ascending=[False, True]).reset_index(drop=True)
    return {
        "model_name": MODEL_NAME,
        "engine_version": V50_ENGINE_VERSION,
        "target_draw": int(draw_numbers[-1]) + 1,
        "history_first_draw": int(draw_numbers[0]),
        "history_last_draw": int(draw_numbers[-1]),
        "history_rows": int(len(clean)),
        "training_window": training_window,
        "min_history": int(min_history),
        "ridge_alpha": float(ridge_alpha),
        "number_probabilities": probability_frame,
        "probability_sum": float(probabilities.sum()),
        "pair_scores": pair_scores,
        "diagnostics": diagnostics,
        "interpretation": (
            "45个主号码的边际入选概率估计，约束总和为6；不是官方概率，也不代表已经证明优于随机。"
        ),
    }


def generate_tickets(
    probabilities: np.ndarray,
    pair_scores: np.ndarray | None = None,
    count: int = 10,
    seed: int = 2026,
    pair_strength: float = 0.5,
) -> list[list[int]]:
    """Generate reproducible six-number tickets by sequential weighted sampling."""
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if probs.shape != (45,) or np.any(~np.isfinite(probs)) or np.any(probs <= 0):
        raise ValueError("probabilities必须是45个正的有限数值。")
    if not np.isclose(float(probs.sum()), 6.0, atol=1e-5):
        raise ValueError("45个号码的边际入选概率之和必须约为6。")
    if not 1 <= int(count) <= 20:
        raise ValueError("候选注数必须位于1—20。")
    if not 0.0 <= float(pair_strength) <= 2.0:
        raise ValueError("pair_strength必须位于0—2。")
    if pair_scores is None:
        pair = np.zeros((45, 45), dtype=float)
    else:
        pair = np.asarray(pair_scores, dtype=float)
        if pair.shape != (45, 45) or not np.all(np.isfinite(pair)):
            raise ValueError("pair_scores必须是45×45有限矩阵。")
        if not np.allclose(pair, pair.T, atol=1e-8):
            raise ValueError("pair_scores必须是对称矩阵。")
    rng = np.random.default_rng(int(seed))
    base_logits = np.log(np.clip(probs, 1e-9, 1 - 1e-9) / np.clip(1.0 - probs, 1e-9, 1.0))
    tickets: list[list[int]] = []
    attempts = 0
    max_attempts = max(1000, int(count) * 200)
    while len(tickets) < int(count) and attempts < max_attempts:
        attempts += 1
        selected: list[int] = []
        selected_indices: list[int] = []
        for _ in range(6):
            scores = base_logits.copy()
            if selected_indices:
                scores += float(pair_strength) * pair[:, selected_indices].sum(axis=1)
            scores[selected_indices] = -np.inf
            finite = scores[np.isfinite(scores)]
            shifted = scores - float(np.max(finite))
            weights = np.zeros(45, dtype=float)
            available = np.isfinite(scores)
            weights[available] = np.exp(np.clip(shifted[available], -40.0, 0.0))
            weights /= weights.sum()
            chosen_index = int(rng.choice(np.arange(45), p=weights))
            selected_indices.append(chosen_index)
            selected.append(chosen_index + 1)
        ticket = sorted(selected)
        if ticket not in tickets:
            tickets.append(ticket)
    if len(tickets) != int(count):
        raise RuntimeError("V5未能生成足够的不同号码组合。")
    return tickets


def build_v50_forward_lock(
    history: pd.DataFrame,
    prediction: dict[str, Any],
    tickets_per_group: int = 10,
    seed: int = 5050,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a timestamped, paired V5-vs-uniform pre-draw lock."""
    clean = validate_frame(history).reset_index(drop=True)
    if clean.empty:
        raise ValueError("没有可用于前瞻锁定的历史数据。")
    draws = clean["draw_no"].to_numpy(dtype=np.int64)
    if np.any(np.diff(draws) != 1):
        raise ValueError("历史期号存在缺失，不能建立V5前瞻锁定。")
    if int(prediction.get("history_last_draw", -1)) != int(draws[-1]):
        raise ValueError("预测所用历史末期与当前数据不一致，拒绝锁定。")
    if prediction.get("engine_version") != V50_ENGINE_VERSION:
        raise ValueError("预测引擎版本不匹配，拒绝锁定。")
    if not 1 <= int(tickets_per_group) <= 20:
        raise ValueError("每个实验组的注数必须为1—20。")

    probs_frame = prediction["number_probabilities"].sort_values("号码")
    probs = probs_frame["预测入选概率"].to_numpy(dtype=float)
    pair = np.asarray(prediction["pair_scores"], dtype=float)
    target_draw = int(draws[-1]) + 1
    model_seed = int(np.random.SeedSequence([int(seed), target_draw, 5001]).generate_state(1)[0])
    random_seed = int(np.random.SeedSequence([int(seed), target_draw, 7001]).generate_state(1)[0])
    model_tickets = generate_tickets(probs, pair, int(tickets_per_group), model_seed, 0.5)
    random_tickets = generate_tickets(np.full(45, BASE_RATE), None, int(tickets_per_group), random_seed, 0.0)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    history_hash = hashlib.sha256(clean.to_csv(index=False).encode("utf-8")).hexdigest()
    probability_lookup = {int(row["号码"]): float(row["预测入选概率"]) for _, row in probs_frame.iterrows()}
    records: list[dict[str, Any]] = []
    for group, tickets in ((MODEL_NAME, model_tickets), ("均匀随机对照", random_tickets)):
        for ticket_no, ticket in enumerate(tickets, 1):
            record = {
                "目标期号": target_draw,
                "锁定时间UTC": created,
                "策略": group,
                "注序": ticket_no,
                "号码": "-".join(f"{x:02d}" for x in ticket),
                "V5边际概率之和": float(sum(probability_lookup[x] for x in ticket)) if group == MODEL_NAME else None,
            }
            record.update({f"n{i}": number for i, number in enumerate(ticket, 1)})
            records.append(record)
    detail = pd.DataFrame(records)
    metadata = {
        "engine_version": V50_ENGINE_VERSION,
        "lock_type": "v50_pre_draw_paired_lock",
        "created_at_utc": created,
        "target_draw": target_draw,
        "history_first_draw": int(draws[0]),
        "history_last_draw": int(draws[-1]),
        "history_rows": int(len(clean)),
        "history_fingerprint_sha256": history_hash,
        "training_window": int(prediction["training_window"]),
        "ridge_alpha": float(prediction["ridge_alpha"]),
        "min_history": int(prediction["min_history"]),
        "tickets_per_group": int(tickets_per_group),
        "model_seed": model_seed,
        "random_seed": random_seed,
        "paired_control": "均匀随机对照",
        "policy": "预测在目标期开奖结果导入前生成并保存；开奖后不得重写",
        "warning": "该锁定只证明预测时间和数据版本；是否优于随机需在后续开奖后按预先规定的指标检验。",
    }
    return detail, metadata
