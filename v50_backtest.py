# -*- coding: utf-8 -*-
"""V5.0 prequential walk-forward evaluation; no network or external data access."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from lotto_core import NUMBER_COLS, validate_frame
from v50_predictive_engine import (
    MODEL_NAME, V50_ENGINE_VERSION, _fit_from_feature_cache,
    build_number_features, generate_tickets, _calibrate_to_six,
)


def _score_ticket_batch(tickets: list[list[int]], actual: set[int]) -> dict[str, float]:
    hits = np.asarray([len(set(ticket) & actual) for ticket in tickets], dtype=float)
    return {
        "mean_hits": float(hits.mean()),
        "ticket_3plus": float(np.mean(hits >= 3)),
        "ticket_4plus": float(np.mean(hits >= 4)),
        "batch_3plus": float(np.any(hits >= 3)),
        "batch_4plus": float(np.any(hits >= 4)),
        "best_hits": float(hits.max()),
        "total_hits": float(hits.sum()),
    }


def _v4_long_frequency_probabilities(past: np.ndarray, long_prior: float = 174.941214) -> np.ndarray:
    counts = np.bincount(np.asarray(past, dtype=np.int16).reshape(-1), minlength=46)[1:46].astype(float)
    raw = (counts + float(long_prior) * (6.0 / 45.0)) / (len(past) + float(long_prior))
    # Calibrate the legacy relative-frequency scores to a six-inclusion probability vector.
    return _calibrate_to_six(np.log(np.clip(raw, 1e-9, 1.0 - 1e-9) / np.clip(1.0 - raw, 1e-9, 1.0)))


def _probability_scores(probabilities: np.ndarray, actual: set[int]) -> tuple[float, float]:
    y = np.zeros(45, dtype=float)
    y[np.asarray(sorted(actual), dtype=int) - 1] = 1.0
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    return brier, log_loss


def _bootstrap_mean_ci(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    samples = rng.integers(0, len(x), size=(int(repetitions), len(x)))
    means = x[samples].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_v50_walk_forward(
    frame: pd.DataFrame,
    test_draws: int = 100,
    training_window: int = 500,
    min_history: int = 180,
    ridge_alpha: float = 1.0,
    tickets_per_draw: int = 10,
    seed: int = 5026,
    bootstrap_reps: int = 2000,
    progress_callback: Callable[[int, int], None] | None = None,
    feature_exclusions: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Evaluate one-step-ahead predictions, training each target from past only."""
    started = time.perf_counter()
    clean = validate_frame(frame).reset_index(drop=True)
    if clean.empty:
        raise ValueError("没有可用于V5回测的历史数据。")
    draws = clean["draw_no"].to_numpy(dtype=np.int64)
    if np.any(np.diff(draws) != 1):
        raise ValueError("V5回测要求所提供历史范围内期号连续。")
    matrix = clean[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)
    n = len(matrix)
    test_draws = int(test_draws)
    min_history = int(min_history)
    training_window = int(training_window)
    tickets_per_draw = int(tickets_per_draw)
    if test_draws < 1 or tickets_per_draw < 1 or tickets_per_draw > 20:
        raise ValueError("测试期数必须为正数，每期每模型注数必须为1—20。")
    if not 30 <= min_history < n or not 30 <= training_window <= 1000:
        raise ValueError("min_history或training_window超出允许范围。")
    start_idx = max(min_history, n - min(test_draws, n - min_history))
    if start_idx >= n:
        raise ValueError("可用样本外测试期数不足。")

    # Feature row i is constructed from draws strictly before i.
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for idx in range(min_history, n):
        cache[idx] = build_number_features(matrix[:idx])
        if progress_callback and (idx - min_history + 1) % 25 == 0:
            progress_callback(idx - min_history + 1, n - min_history)

    rows: list[dict[str, Any]] = []
    for target_idx in range(start_idx, n):
        probabilities, pair_scores, diagnostics = _fit_from_feature_cache(
            matrix, cache, target_idx, training_window, ridge_alpha, feature_exclusions
        )
        target_draw = int(draws[target_idx])
        actual = set(int(x) for x in matrix[target_idx])
        target_seed = int(np.random.SeedSequence([int(seed), target_draw]).generate_state(1)[0])
        pair_ablation = {"pair_mean_log_lift", "pair_sd_log_lift"}.intersection(feature_exclusions)
        model_tickets = generate_tickets(
            probabilities, pair_scores, tickets_per_draw, target_seed,
            pair_strength=0.0 if pair_ablation else 0.5
        )
        v4_probabilities = _v4_long_frequency_probabilities(matrix[:target_idx], 174.941214)
        v4_tickets = generate_tickets(
            v4_probabilities, None, tickets_per_draw, target_seed ^ 0x4F4F, pair_strength=0.0
        )
        random_probabilities = np.full(45, 6.0 / 45.0)
        random_tickets = generate_tickets(
            random_probabilities, None, tickets_per_draw,
            target_seed ^ 0x5A17, pair_strength=0.0
        )
        model_metrics = _score_ticket_batch(model_tickets, actual)
        v4_metrics = _score_ticket_batch(v4_tickets, actual)
        random_metrics = _score_ticket_batch(random_tickets, actual)
        brier, log_loss = _probability_scores(probabilities, actual)
        v4_brier, v4_log_loss = _probability_scores(v4_probabilities, actual)
        random_brier, random_log_loss = _probability_scores(random_probabilities, actual)
        row: dict[str, Any] = {
            "目标期号": target_draw,
            "训练结束期": int(draws[target_idx - 1]),
            "训练期数": int(diagnostics["training_draws"]),
            "V5_Brier": brier,
            "V5_LogLoss": log_loss,
            "V4_Brier": v4_brier,
            "V4_LogLoss": v4_log_loss,
            "随机_Brier": random_brier,
            "随机_LogLoss": random_log_loss,
            "V5_概率和": float(probabilities.sum()),
            "V4_概率和": float(v4_probabilities.sum()),
            "V4_号码列表": "|".join(str(x) for x in v4_tickets),
            "V5_号码列表": "|".join(str(x) for x in model_tickets),
            "随机_号码列表": "|".join(str(x) for x in random_tickets),
        }
        for key, value in model_metrics.items():
            row[f"V5_{key}"] = value
        for key, value in v4_metrics.items():
            row[f"V4_{key}"] = value
        for key, value in random_metrics.items():
            row[f"随机_{key}"] = value
        for key in ("mean_hits", "ticket_3plus", "ticket_4plus", "batch_3plus", "batch_4plus", "best_hits"):
            row[f"差值_{key}"] = model_metrics[key] - random_metrics[key]
            row[f"差值_V5_vs_V4_{key}"] = model_metrics[key] - v4_metrics[key]
        rows.append(row)
        if progress_callback:
            progress_callback(target_idx - start_idx + 1, n - start_idx)

    detail = pd.DataFrame(rows)
    metric_pairs = [
        ("mean_hits", "平均命中数"),
        ("ticket_3plus", "单注3+率"),
        ("ticket_4plus", "单注4+率"),
        ("batch_3plus", "整批至少一注3+率"),
        ("batch_4plus", "整批至少一注4+率"),
        ("best_hits", "整批最佳命中数"),
    ]
    summary_rows = []
    for key, label in metric_pairs:
        diffs = detail[f"差值_{key}"].to_numpy(dtype=float)
        low, high = _bootstrap_mean_ci(diffs, bootstrap_reps, seed + len(summary_rows) * 31)
        v4_diffs = detail[f"差值_V5_vs_V4_{key}"].to_numpy(dtype=float)
        v4_low, v4_high = _bootstrap_mean_ci(v4_diffs, bootstrap_reps, seed + 1701 + len(summary_rows) * 31)
        summary_rows.append({
            "指标": label,
            "V5平均": float(detail[f"V5_{key}"].mean()),
            "V4冻结模型平均": float(detail[f"V4_{key}"].mean()),
            "随机基准平均": float(detail[f"随机_{key}"].mean()),
            "差值": float(diffs.mean()),
            "V5_vs_V4差值": float(v4_diffs.mean()),
            "V5_vs_V4差值95%CI下限": v4_low,
            "V5_vs_V4差值95%CI上限": v4_high,
            "差值95%CI下限": low,
            "差值95%CI上限": high,
            "V5领先期数": int(np.sum(diffs > 0)),
            "随机领先期数": int(np.sum(diffs < 0)),
            "平局期数": int(np.sum(diffs == 0)),
        })
    for metric, label in (("Brier", "概率Brier分数"), ("LogLoss", "概率Log Loss")):
        v5_values = detail[f"V5_{metric}"].to_numpy(dtype=float)
        v4_values = detail[f"V4_{metric}"].to_numpy(dtype=float)
        random_values = detail[f"随机_{metric}"].to_numpy(dtype=float)
        # Proper scores are lower-is-better; positive deltas mean V5 is better.
        improvement = random_values - v5_values
        v4_improvement = v4_values - v5_values
        low, high = _bootstrap_mean_ci(improvement, bootstrap_reps, seed + 911 + len(summary_rows))
        v4_low, v4_high = _bootstrap_mean_ci(v4_improvement, bootstrap_reps, seed + 1901 + len(summary_rows))
        summary_rows.append({
            "指标": label,
            "V5平均": float(v5_values.mean()),
            "V4冻结模型平均": float(v4_values.mean()),
            "随机基准平均": float(random_values.mean()),
            "差值": float(improvement.mean()),
            "V5_vs_V4差值": float(v4_improvement.mean()),
            "V5_vs_V4差值95%CI下限": v4_low,
            "V5_vs_V4差值95%CI上限": v4_high,
            "差值95%CI下限": low,
            "差值95%CI上限": high,
            "V5领先期数": int(np.sum(improvement > 0)),
            "随机领先期数": int(np.sum(improvement < 0)),
            "平局期数": int(np.sum(improvement == 0)),
        })
    summary = pd.DataFrame(summary_rows)
    fingerprint = hashlib.sha256(clean.to_csv(index=False).encode("utf-8")).hexdigest()
    return {
        "engine_version": V50_ENGINE_VERSION,
        "model_name": MODEL_NAME,
        "test_start_draw": int(draws[start_idx]),
        "test_end_draw": int(draws[-1]),
        "test_draws": int(len(detail)),
        "training_window": training_window,
        "min_history": min_history,
        "ridge_alpha": float(ridge_alpha),
        "feature_exclusions": list(feature_exclusions),
        "tickets_per_draw": tickets_per_draw,
        "seed": int(seed),
        "data_fingerprint_sha256": fingerprint,
        "detail": detail,
        "summary": summary,
        "elapsed_seconds": float(time.perf_counter() - started),
        "evidence_status": "retrospective_only_not_prospective_evidence",
        "interpretation": (
            "历史滚动回测只用于筛查和比较，不构成未来预测优势证据。"
            "V5必须在开奖前锁定预测，并在新开奖出现后逐期评分。"
        ),
    }
