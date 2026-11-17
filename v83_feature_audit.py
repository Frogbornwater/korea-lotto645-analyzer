# -*- coding: utf-8 -*-
"""V8.3 first-order V5 feature contribution audit.

This research isolates the incremental value of the eight non-pair V5 number
features. The full model and each leave-one-feature-out model use the same
pre-draw rolling targets, training window, Ridge alpha, and feature cache.
Pair-derived features are excluded from every model so second-order pair
effects cannot be counted twice.

A draw-label permutation placebo preserves each six-number label vector but
breaks its alignment with the historical feature rows. The audit reports
paired moving-block bootstrap intervals and Holm-adjusted one-sided p-values
across all feature/metric comparisons.

This module is offline and research-only. It does not modify production
predictions, generate tickets, or access external lottery services.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

from lotto_core import NUMBER_COLS, validate_frame
from v50_predictive_engine import (
    BASE_RATE,
    FEATURE_NAMES,
    _fit_from_feature_cache,
    build_number_features,
)

V83_VERSION = "8.3.0-first-order-feature-audit"
PAIR_FEATURE_EXCLUSIONS = ("pair_mean_log_lift", "pair_sd_log_lift")
FIRST_ORDER_FEATURES = tuple(name for name in FEATURE_NAMES if name not in PAIR_FEATURE_EXCLUSIONS)
FEATURE_METRICS = ("top6_hits", "top10_hits", "Brier", "LogLoss")
BOOTSTRAP_REPETITIONS = 10000
BLOCK_LENGTH = 10


def _score_probabilities(
    probabilities: np.ndarray,
    actual_numbers: np.ndarray | list[int],
) -> dict[str, float]:
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if p.shape != (45,) or not np.all(np.isfinite(p)):
        raise ValueError("V8.3预测必须包含45个有限概率。")
    if abs(float(p.sum()) - 6.0) > 1e-5 or np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("V8.3预测概率必须位于0—1且总和为6。")
    actual = np.asarray(actual_numbers, dtype=int).reshape(-1)
    if actual.size != 6 or len(set(int(x) for x in actual)) != 6:
        raise ValueError("实际开奖必须包含6个互不相同的号码。")
    if np.any(actual < 1) or np.any(actual > 45):
        raise ValueError("实际号码必须位于1—45。")
    y = np.zeros(45, dtype=float)
    y[actual - 1] = 1.0

    ranking = np.argsort(-p, kind="stable")
    top6 = float(y[ranking[:6]].sum())
    top10 = float(y[ranking[:10]].sum())
    top15 = float(y[ranking[:15]].sum())
    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))
    return {
        "top6_hits": top6,
        "top10_hits": top10,
        "top15_hits": top15,
        "Brier": brier,
        "LogLoss": log_loss,
    }


def _bootstrap_index_matrix(
    n: int,
    *,
    repetitions: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    if n < 2:
        raise ValueError("区块Bootstrap至少需要2期。")
    block = max(1, min(int(block_length), n))
    reps = max(100, int(repetitions))
    blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(int(seed))
    starts = rng.integers(0, n, size=(reps, blocks))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    return indices.reshape(reps, -1)[:, :n]


def _bootstrap_summary(
    values: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    x = np.asarray(values, dtype=float).reshape(-1)
    if indices.ndim != 2 or indices.shape[1] != len(x):
        raise ValueError("Bootstrap索引维度与样本长度不一致。")
    observed = float(x.mean())
    boot = x[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])

    # One-sided H1: mean gain > 0. Centering the paired series constructs the
    # block-bootstrap null distribution while preserving serial dependence.
    centered = x - observed
    null_boot = centered[indices].mean(axis=1)
    p_value = (1.0 + float(np.sum(null_boot >= observed))) / (len(null_boot) + 1.0)
    return {
        "mean_gain_positive_is_better": observed,
        "moving_block_bootstrap_95ci_low": float(low),
        "moving_block_bootstrap_95ci_high": float(high),
        "one_sided_bootstrap_p": float(p_value),
    }


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down family-wise error correction."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Holm校正输入必须是0—1之间的有限p值。")
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, float(m - rank) * float(p[idx]))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def evaluate_v83_feature_audit(
    frame: pd.DataFrame,
    *,
    test_draws: int = 500,
    training_window: int = 500,
    min_history: int = 180,
    ridge_alpha: float = 1.0,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
    block_length: int = BLOCK_LENGTH,
    seed: int = 8302028,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run 500-draw prequential feature ablation and placebo audit."""
    clean = validate_frame(frame).reset_index(drop=True)
    if clean.empty:
        raise ValueError("没有可用于V8.3研究的手动历史数据。")
    draws = clean["draw_no"].to_numpy(dtype=np.int64)
    if np.any(np.diff(draws) != 1):
        raise ValueError("V8.3滚动研究要求期号连续。")
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

    start_idx = max(min_history, n - test_draws)
    if n - start_idx < 2:
        raise ValueError("可用样本外测试期不足。")

    # The same target features are reused across all ablation fits.
    feature_cache = {
        idx: build_number_features(matrix[:idx])
        for idx in range(min_history, n)
    }

    records: list[dict[str, Any]] = []
    for target_idx in range(start_idx, n):
        target_draw = int(draws[target_idx])
        actual = matrix[target_idx].astype(int).tolist()
        full_probabilities, _, diagnostics = _fit_from_feature_cache(
            matrix,
            feature_cache,
            target_idx,
            training_window,
            float(ridge_alpha),
            PAIR_FEATURE_EXCLUSIONS,
        )
        full_score = _score_probabilities(full_probabilities, actual)

        placebo_probabilities, _, _ = _fit_from_feature_cache(
            matrix,
            feature_cache,
            target_idx,
            training_window,
            float(ridge_alpha),
            PAIR_FEATURE_EXCLUSIONS,
            placebo_seed=int(seed) + target_draw,
        )
        placebo_score = _score_probabilities(placebo_probabilities, actual)

        row: dict[str, Any] = {
            "target_draw": target_draw,
            "training_end_draw": int(draws[target_idx - 1]),
            "training_draws": int(diagnostics["training_draws"]),
            **{f"full_{key}": float(value) for key, value in full_score.items()},
            **{f"placebo_{key}": float(value) for key, value in placebo_score.items()},
            "history_fingerprint_sha256": hashlib.sha256(
                matrix[max(0, target_idx - training_window):target_idx].tobytes()
            ).hexdigest(),
        }

        for feature in FIRST_ORDER_FEATURES:
            excluded = tuple(PAIR_FEATURE_EXCLUSIONS) + (feature,)
            probabilities, _, _ = _fit_from_feature_cache(
                matrix,
                feature_cache,
                target_idx,
                training_window,
                float(ridge_alpha),
                excluded,
            )
            score = _score_probabilities(probabilities, actual)
            for key, value in score.items():
                row[f"without_{feature}_{key}"] = float(value)
        records.append(row)

    detail = pd.DataFrame(records)
    bootstrap_indices = _bootstrap_index_matrix(
        len(detail),
        repetitions=bootstrap_repetitions,
        block_length=block_length,
        seed=int(seed) + 701,
    )

    comparison_rows: list[dict[str, Any]] = []
    for feature in FIRST_ORDER_FEATURES:
        for metric in FEATURE_METRICS:
            full = detail[f"full_{metric}"].to_numpy(dtype=float)
            without = detail[f"without_{feature}_{metric}"].to_numpy(dtype=float)
            # All gains are oriented so a positive value means keeping the
            # feature in the full model is better than removing it.
            if metric in ("top6_hits", "top10_hits"):
                gains = full - without
            else:
                gains = without - full
            stats = _bootstrap_summary(gains, bootstrap_indices)
            comparison_rows.append({
                "feature": feature,
                "metric": metric,
                **stats,
            })

    holm_values = _holm_adjust([row["one_sided_bootstrap_p"] for row in comparison_rows])
    for row, adjusted in zip(comparison_rows, holm_values):
        row["holm_adjusted_p"] = float(adjusted)
        row["holm_significant_05"] = bool(adjusted < 0.05)

    feature_summaries: list[dict[str, Any]] = []
    for feature in FIRST_ORDER_FEATURES:
        group = [row for row in comparison_rows if row["feature"] == feature]
        lookup = {row["metric"]: row for row in group}
        consistent_direction = all(
            lookup[metric]["mean_gain_positive_is_better"] > 0.0
            for metric in FEATURE_METRICS
        )
        probability_metrics_supported = (
            lookup["Brier"]["holm_adjusted_p"] < 0.05
            and lookup["LogLoss"]["holm_adjusted_p"] < 0.05
        )
        feature_supported = bool(consistent_direction and probability_metrics_supported)
        feature_summaries.append({
            "feature": feature,
            "consistent_positive_gain_across_all_metrics": bool(consistent_direction),
            "Brier_gain": lookup["Brier"]["mean_gain_positive_is_better"],
            "Brier_holm_p": lookup["Brier"]["holm_adjusted_p"],
            "LogLoss_gain": lookup["LogLoss"]["mean_gain_positive_is_better"],
            "LogLoss_holm_p": lookup["LogLoss"]["holm_adjusted_p"],
            "Top6_gain": lookup["top6_hits"]["mean_gain_positive_is_better"],
            "Top10_gain": lookup["top10_hits"]["mean_gain_positive_is_better"],
            "incremental_feature_supported": feature_supported,
        })

    p0 = BASE_RATE
    uniform = {
        "top6_hits": 0.8,
        "top10_hits": 10.0 * 6.0 / 45.0,
        "top15_hits": 2.0,
        "Brier": p0 * (1.0 - p0),
        "LogLoss": -(p0 * math.log(p0) + (1.0 - p0) * math.log(1.0 - p0)),
    }

    baseline_vs_uniform: list[dict[str, Any]] = []
    for offset, metric in enumerate(FEATURE_METRICS + ("top15_hits",)):
        values = detail[f"full_{metric}"].to_numpy(dtype=float)
        # Higher hits are better; lower proper scores are better.
        gain = values - uniform[metric] if metric in ("top6_hits", "top10_hits", "top15_hits") else uniform[metric] - values
        stats = _bootstrap_summary(
            gain,
            _bootstrap_index_matrix(
                len(detail),
                repetitions=bootstrap_repetitions,
                block_length=block_length,
                seed=int(seed) + 901 + offset,
            ),
        )
        baseline_vs_uniform.append({"metric": metric, **stats})

    placebo_comparisons: list[dict[str, Any]] = []
    for offset, metric in enumerate(FEATURE_METRICS):
        full = detail[f"full_{metric}"].to_numpy(dtype=float)
        placebo = detail[f"placebo_{metric}"].to_numpy(dtype=float)
        gain = full - placebo if metric in ("top6_hits", "top10_hits") else placebo - full
        stats = _bootstrap_summary(
            gain,
            _bootstrap_index_matrix(
                len(detail),
                repetitions=bootstrap_repetitions,
                block_length=block_length,
                seed=int(seed) + 1101 + offset,
            ),
        )
        placebo_comparisons.append({"metric": metric, **stats})

    block_reports = []
    for block_idx, start in enumerate(range(0, len(detail), 100), start=1):
        subset = detail.iloc[start:start + 100]
        block_reports.append({
            "block": block_idx,
            "target_range": [
                int(subset["target_draw"].min()),
                int(subset["target_draw"].max()),
            ],
            "full_top6_hits": float(subset["full_top6_hits"].mean()),
            "full_top10_hits": float(subset["full_top10_hits"].mean()),
            "full_Brier": float(subset["full_Brier"].mean()),
            "full_LogLoss": float(subset["full_LogLoss"].mean()),
            "placebo_top10_hits": float(subset["placebo_top10_hits"].mean()),
            "placebo_Brier": float(subset["placebo_Brier"].mean()),
            "placebo_LogLoss": float(subset["placebo_LogLoss"].mean()),
        })

    supported_features = [
        row["feature"] for row in feature_summaries
        if row["incremental_feature_supported"]
    ]
    report = {
        "research_id": "lotto645-v83-first-order-feature-audit-20261118",
        "engine_version": V83_VERSION,
        "status": "historical_feature_ablation_not_independent_or_prospective_evidence",
        "research_only": True,
        "test_draws": int(len(detail)),
        "target_start_draw": int(detail["target_draw"].min()),
        "target_end_draw": int(detail["target_draw"].max()),
        "training_window": training_window,
        "min_history": min_history,
        "ridge_alpha": float(ridge_alpha),
        "first_order_features": list(FIRST_ORDER_FEATURES),
        "pair_features_excluded": list(PAIR_FEATURE_EXCLUSIONS),
        "candidate_models": 1 + len(FIRST_ORDER_FEATURES) + 1,
        "metrics": list(FEATURE_METRICS) + ["top15_hits"],
        "bootstrap": {
            "method": "circular moving-block bootstrap",
            "repetitions": int(bootstrap_repetitions),
            "block_length": int(block_length),
            "feature_tests": int(len(comparison_rows)),
            "multiple_testing": "Holm step-down across all 8 features x 4 endpoints (32 tests)",
        },
        "full_model_mean_scores": {
            metric: float(detail[f"full_{metric}"].mean())
            for metric in ("top6_hits", "top10_hits", "top15_hits", "Brier", "LogLoss")
        },
        "placebo_model_mean_scores": {
            metric: float(detail[f"placebo_{metric}"].mean())
            for metric in ("top6_hits", "top10_hits", "top15_hits", "Brier", "LogLoss")
        },
        "uniform_exact_baseline": uniform,
        "full_model_vs_uniform": baseline_vs_uniform,
        "full_model_vs_label_placebo": placebo_comparisons,
        "feature_comparisons": comparison_rows,
        "feature_summaries": feature_summaries,
        "incremental_features_supported_after_holm": supported_features,
        "evidence_gate": {
            "minimum_draws_required": 200,
            "enough_draws": bool(len(detail) >= 200),
            "at_least_one_feature_supported": bool(supported_features),
            "independent_replication_allowed": bool(len(detail) >= 200 and bool(supported_features)),
            "production_model_change_allowed": False,
            "prediction_superiority_proven": False,
        },
        "block_reports": block_reports,
        "data_fingerprint_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "limitations": [
            "每个目标期的特征和标签训练严格使用此前开奖；所有配置共用相同目标期和训练窗口。",
            "本阶段不搜索训练窗口、Ridge参数或特征组合，只做预先指定的逐特征删除实验。",
            "共现特征在所有配置中统一排除，避免与V8.2二阶研究重复计数。",
            "32项特征增量检验使用Holm方法控制家族错误率。",
            "标签置乱负对照破坏历史特征与开奖标签的时间对应关系，但保留每期开奖6个主号码的向量结构。",
            "即使某个特征通过本轮筛查，也只允许进入下一轮独立复验，不可直接变更正式选号。",
            "历史研究不能证明未来开奖具有可预测性。",
            "所有历史数据仅来自用户手动维护的CSV/Excel；不访问任何开奖网站或API。",
        ],
    }
    return report, detail
