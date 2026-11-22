# -*- coding: utf-8 -*-
"""V8.6 prequential audit of V5 probability calibration and ranking quality."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lotto_core import NUMBER_COLS, validate_frame
from v50_predictive_engine import (
    BASE_RATE, V50_ENGINE_VERSION, _calibrate_to_six,
    _fit_from_feature_cache, build_number_features,
)

TEMPERATURE_GRID = (0.50, 0.67, 0.80, 1.00, 1.25, 1.50, 2.00, 3.00)
TOP_K_VALUES = (6, 10, 15)


def _labels(numbers: np.ndarray) -> np.ndarray:
    return np.isin(np.arange(1, 46), np.asarray(numbers, dtype=int)).astype(float)


def _v4_probabilities(past: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.asarray(past, dtype=np.int16).reshape(-1), minlength=46)[1:46].astype(float)
    raw = (counts + 174.941214 * BASE_RATE) / (len(past) + 174.941214)
    logits = np.log(np.clip(raw, 1e-9, 1.0 - 1e-9) / np.clip(1.0 - raw, 1e-9, 1.0))
    return _calibrate_to_six(logits)


def _temperature_transform(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Monotone temperature scaling, with a per-draw intercept restoring sum(p)=6."""
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if p.shape != (45,) or not np.all(np.isfinite(p)) or np.any((p <= 0) | (p >= 1)):
        raise ValueError("温度校准需要45个(0,1)内的有限概率。")
    if not 0.25 <= float(temperature) <= 4.0:
        raise ValueError("temperature超出允许范围。")
    logits = np.log(p / (1.0 - p)) / float(temperature)
    low, high = -30.0, 30.0
    for _ in range(80):
        mid = (low + high) / 2.0
        total = float((1.0 / (1.0 + np.exp(-np.clip(logits + mid, -40.0, 40.0)))).sum())
        if total > 6.0:
            high = mid
        else:
            low = mid
    calibrated = 1.0 / (1.0 + np.exp(-np.clip(logits + (low + high) / 2.0, -40.0, 40.0)))
    calibrated *= 6.0 / float(calibrated.sum())
    return calibrated


def _log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
    y = np.asarray(labels, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _select_temperature(
    past_probabilities: list[np.ndarray],
    past_labels: list[np.ndarray],
) -> tuple[float, dict[str, float]]:
    if len(past_probabilities) != len(past_labels) or not past_probabilities:
        raise ValueError("温度选择需要成对的历史预测和实际标签。")
    scores: dict[str, float] = {}
    for temperature in TEMPERATURE_GRID:
        losses = [
            _log_loss(_temperature_transform(p, temperature), y)
            for p, y in zip(past_probabilities, past_labels)
        ]
        scores[str(temperature)] = float(np.mean(losses))
    # Deterministic tie-break: prefer the value closest to no calibration (T=1).
    chosen = min(TEMPERATURE_GRID, key=lambda t: (scores[str(t)], abs(np.log(t))))
    return float(chosen), scores


def _rank_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    order = np.argsort(-p, kind="stable")
    actual = set(np.flatnonzero(y > 0.5).tolist())
    ranks = np.empty(45, dtype=int)
    ranks[order] = np.arange(1, 46)
    result: dict[str, float] = {
        "mean_actual_rank": float(np.mean([ranks[i] for i in actual])),
        "actual_mean_percentile": float(np.mean([(45 - ranks[i]) / 44.0 for i in actual])),
    }
    for k in TOP_K_VALUES:
        result[f"top{k}_hits"] = float(sum(i in actual for i in order[:k]))
    return result


def _score(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    result = {
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": _log_loss(p, y),
    }
    result.update(_rank_metrics(p, y))
    return result


def _moving_block_bootstrap(
    values: np.ndarray, *, repetitions: int, block_length: int, seed: int,
) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float).reshape(-1)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan"), 1.0
    block_count = int(np.ceil(n / int(block_length)))
    rng = np.random.default_rng(int(seed))
    starts = rng.integers(0, n, size=(int(repetitions), block_count))
    offsets = np.arange(int(block_length))
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    means = x[indices.reshape(int(repetitions), -1)[:, :n]].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    p_one_sided = (1.0 + float(np.sum(means <= 0.0))) / (float(repetitions) + 1.0)
    return float(low), float(high), p_one_sided


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.ones(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * float(p_values[int(index)])))
        adjusted[int(index)] = running
    return adjusted.tolist()


def _reliability_bins(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10,
) -> tuple[list[dict[str, float]], float]:
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=float).reshape(-1)
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, int(bins))
    rows: list[dict[str, float]] = []
    ece = 0.0
    for index, group in enumerate(groups, 1):
        if not len(group):
            continue
        predicted = float(p[group].mean())
        observed = float(y[group].mean())
        weight = len(group) / len(p)
        ece += weight * abs(predicted - observed)
        rows.append({
            "bin": float(index),
            "count": float(len(group)),
            "mean_predicted": predicted,
            "observed_frequency": observed,
            "absolute_gap": abs(predicted - observed),
        })
    return rows, float(ece)


def _rank_stability(probabilities: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 45 or len(matrix) < 2:
        return {"adjacent_spearman_mean": float("nan"), "top6_overlap_mean": float("nan")}
    correlations = []
    overlaps = []
    for previous, current in zip(matrix[:-1], matrix[1:]):
        prev_rank = np.argsort(np.argsort(-previous, kind="stable"))
        curr_rank = np.argsort(np.argsort(-current, kind="stable"))
        correlations.append(float(np.corrcoef(prev_rank, curr_rank)[0, 1]))
        overlaps.append(float(len(set(np.argsort(-previous)[:6]) & set(np.argsort(-current)[:6]))))
    return {
        "adjacent_spearman_mean": float(np.mean(correlations)),
        "top6_overlap_mean": float(np.mean(overlaps)),
    }


def run_v86_probability_audit(
    frame: pd.DataFrame, *, test_draws: int = 500, warmup_draws: int = 200,
    calibration_window: int = 200, training_window: int = 500,
    min_history: int = 180, ridge_alpha: float = 1.0,
    bootstrap_repetitions: int = 10_000, block_length: int = 10,
    seed: int = 86_2026,
) -> dict[str, Any]:
    """Audit 500 targets; each temperature is selected only from prior forecasts."""
    started = time.perf_counter()
    clean = validate_frame(frame).reset_index(drop=True)
    if clean.empty:
        raise ValueError("没有可用于V8.6的手动历史数据。")
    draws = clean["draw_no"].to_numpy(dtype=np.int64)
    if draws[0] != 1 or np.any(np.diff(draws) != 1):
        raise ValueError("V8.6要求从第1期开始且期号连续的手动历史数据。")
    n = len(clean)
    test_draws, warmup_draws, calibration_window = int(test_draws), int(warmup_draws), int(calibration_window)
    if test_draws < 2 or warmup_draws < calibration_window or calibration_window < 10:
        raise ValueError("V8.6要求测试期数至少2，且预热期数不少于校准窗口（至少10）。")
    if test_draws + warmup_draws > n - int(min_history):
        raise ValueError("V8.6可用的预热加测试期数不足。")
    if not 30 <= int(training_window) <= 1000 or not 30 <= int(min_history) < n:
        raise ValueError("V8.6训练窗口或最少历史期数无效。")
    if int(bootstrap_repetitions) < 100 or int(block_length) < 1:
        raise ValueError("Bootstrap重复次数至少100，区块长度必须为正数。")

    first_test_idx = n - test_draws
    first_forecast_idx = first_test_idx - warmup_draws
    # Feature rows are all strictly pre-draw and are cached once for both warm-up
    # and evaluation. No feature row contains its target draw.
    feature_cache = {
        idx: build_number_features(clean.loc[:idx - 1, NUMBER_COLS].to_numpy(dtype=np.int16))
        for idx in range(int(min_history), n)
    }
    matrix = clean[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)
    forecast_probabilities: list[np.ndarray] = []
    forecast_labels: list[np.ndarray] = []
    forecast_draws: list[int] = []
    detail_rows: list[dict[str, Any]] = []

    for target_idx in range(first_forecast_idx, n):
        probabilities, _, diagnostics = _fit_from_feature_cache(
            matrix, feature_cache, target_idx, int(training_window), float(ridge_alpha)
        )
        labels = _labels(matrix[target_idx])
        target_draw = int(draws[target_idx])
        if len(forecast_probabilities) < calibration_window:
            temperature = 1.0
            calibration_loss = None
            calibration_scores: dict[str, float] = {}
        else:
            temperature, calibration_scores = _select_temperature(
                forecast_probabilities[-calibration_window:],
                forecast_labels[-calibration_window:],
            )
            calibration_loss = calibration_scores[str(temperature)]
        calibrated = _temperature_transform(probabilities, temperature)
        v4 = _v4_probabilities(matrix[:target_idx])

        forecast_probabilities.append(probabilities.copy())
        forecast_labels.append(labels.copy())
        forecast_draws.append(target_draw)

        if target_idx < first_test_idx:
            continue

        raw_metrics = _score(probabilities, labels)
        calibrated_metrics = _score(calibrated, labels)
        v4_metrics = _score(v4, labels)
        uniform = np.full(45, BASE_RATE, dtype=float)
        uniform_metrics = _score(uniform, labels)
        row: dict[str, Any] = {
            "target_draw": target_draw,
            "training_end_draw": int(draws[target_idx - 1]),
            "training_draws": int(diagnostics["training_draws"]),
            "v5_probability_sum": float(probabilities.sum()),
            "v5_probability_min": float(probabilities.min()),
            "v5_probability_max": float(probabilities.max()),
            "temperature_selected_from_prior_forecasts": float(temperature),
            "calibration_window": calibration_window,
            "calibration_training_logloss": calibration_loss,
            "actual_main_numbers": json.dumps(sorted(int(x) for x in matrix[target_idx])),
            "v5_probabilities": json.dumps([float(x) for x in probabilities], separators=(",", ":")),
            "calibrated_probabilities": json.dumps([float(x) for x in calibrated], separators=(",", ":")),
        }
        for prefix, metrics in (
            ("v5", raw_metrics), ("calibrated", calibrated_metrics),
            ("v4", v4_metrics), ("uniform", uniform_metrics),
        ):
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        detail_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    if len(detail) != test_draws:
        raise RuntimeError("V8.6最终评分期数与预设测试期数不一致。")

    # Primary comparisons are predeclared and directionally coded so positive
    # values always mean improvement of the tested V5 method.
    comparisons: list[dict[str, Any]] = []
    p_values: list[float] = []
    comparison_specs = (
        ("V5 Brier improvement vs uniform", detail["uniform_brier"] - detail["v5_brier"]),
        ("V5 LogLoss improvement vs uniform", detail["uniform_logloss"] - detail["v5_logloss"]),
        ("V5 top-6 hit improvement vs uniform expectation", detail["v5_top6_hits"] - 0.8),
        ("Rolling temperature Brier improvement vs raw V5", detail["v5_brier"] - detail["calibrated_brier"]),
        ("Rolling temperature LogLoss improvement vs raw V5", detail["v5_logloss"] - detail["calibrated_logloss"]),
    )
    for index, (name, values) in enumerate(comparison_specs):
        low, high, p_value = _moving_block_bootstrap(
            values.to_numpy(dtype=float), repetitions=bootstrap_repetitions,
            block_length=block_length, seed=int(seed) + 101 * index,
        )
        p_values.append(p_value)
        comparisons.append({
            "comparison": name,
            "mean_improvement": float(values.mean()),
            "moving_block_bootstrap_95ci": [low, high],
            "one_sided_bootstrap_p": p_value,
        })
    for item, adjusted in zip(comparisons, _holm_adjust(p_values)):
        item["holm_adjusted_p"] = float(adjusted)

    raw_prob_matrix = np.asarray([json.loads(x) for x in detail["v5_probabilities"]], dtype=float)
    cal_prob_matrix = np.asarray([json.loads(x) for x in detail["calibrated_probabilities"]], dtype=float)
    actual_matrix = np.vstack([_labels(row) for row in matrix[first_test_idx:]])
    raw_reliability, raw_ece = _reliability_bins(raw_prob_matrix, actual_matrix)
    cal_reliability, cal_ece = _reliability_bins(cal_prob_matrix, actual_matrix)
    temperature_counts = {
        str(float(t)): int(np.sum(np.isclose(detail["temperature_selected_from_prior_forecasts"], t)))
        for t in TEMPERATURE_GRID
    }
    metric_summary = {}
    for prefix in ("v5", "calibrated", "v4", "uniform"):
        metric_summary[prefix] = {
            key: float(detail[f"{prefix}_{key}"].mean())
            for key in ("brier", "logloss", "top6_hits", "top10_hits", "top15_hits", "mean_actual_rank", "actual_mean_percentile")
        }
    report = {
        "research_version": "8.6.0-prequential-probability-calibration-audit",
        "status": "historical_prequential_audit_not_independent_or_prospective_evidence",
        "target_start_draw": int(detail["target_draw"].iloc[0]),
        "target_end_draw": int(detail["target_draw"].iloc[-1]),
        "test_draws": int(len(detail)),
        "warmup_forecasts": warmup_draws,
        "calibration_window": calibration_window,
        "training_window": int(training_window),
        "min_history": int(min_history),
        "ridge_alpha": float(ridge_alpha),
        "temperature_grid": list(TEMPERATURE_GRID),
        "temperature_selection": "minimum mean logloss over previous 200 prequential forecast vectors only; no target leakage",
        "v5_engine_version": V50_ENGINE_VERSION,
        "data_fingerprint_sha256": hashlib.sha256(clean.to_csv(index=False).encode("utf-8")).hexdigest(),
        "methods": {
            "v5": "unchanged V5 raw 45-number marginal probabilities",
            "calibrated": "rolling scalar temperature scaling selected from prior forecast outcomes; rank-preserving and sum-to-six",
            "v4": "frozen V4 long-frequency probability baseline",
            "uniform": "constant 6/45 per-number inclusion probability",
        },
        "uniform_expected_topk_hits": {str(k): 6.0 * k / 45.0 for k in TOP_K_VALUES},
        "primary_comparisons": comparisons,
        "summary_by_method": metric_summary,
        "reliability": {
            "raw_v5_equal_count_bins": raw_reliability,
            "calibrated_equal_count_bins": cal_reliability,
            "raw_v5_expected_calibration_error": raw_ece,
            "calibrated_expected_calibration_error": cal_ece,
        },
        "ranking_stability": _rank_stability(raw_prob_matrix),
        "selected_temperature_counts": temperature_counts,
        "evidence_gate": {
            "probability_quality_audited": True,
            "ranking_quality_audited": True,
            "calibration_is_strictly_prequential": True,
            "predictive_superiority_proven": False,
            "production_model_change_allowed": False,
            "forward_lock_created": False,
            "prospective_validation_required": True,
        },
        "limitations": [
            "The 752-1251 target window has already been examined in prior V7/V8 research; it is not an untouched holdout.",
            "The 200 warm-up forecasts are used only to initialize the rolling calibrator and are not included in scored metrics.",
            "Temperature scaling preserves each target's exact V5 number ranking; it cannot improve top-k hit counts.",
            "Reliability bins are descriptive post-hoc diagnostics; inferential uncertainty is computed at draw level with moving blocks.",
            "Lottery draws are designed to be independent; historical patterns do not guarantee future advantage.",
        ],
        "production_model_changed": False,
        "forward_lock_created": False,
        "manual_history_only": True,
        "official_lottery_site_or_api_accessed": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return {
        "report": report,
        "detail": detail.drop(columns=["v5_probabilities", "calibrated_probabilities"]),
        "reliability": pd.DataFrame([
            {"method": "V5 raw", **row} for row in raw_reliability
        ] + [
            {"method": "V5 rolling temperature", **row} for row in cal_reliability
        ]),
    }


def write_v86_artifacts(result: dict[str, Any], output_dir: str | None = None) -> tuple[str, str, str]:
    root = Path(output_dir) if output_dir else Path("artifacts") / "v86"
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "v86_probability_audit_report.json"
    detail_path = root / "v86_probability_audit_detail.csv"
    reliability_path = root / "v86_reliability_bins.csv"
    report_path.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2), encoding="utf-8")
    result["detail"].to_csv(detail_path, index=False, encoding="utf-8-sig")
    result["reliability"].to_csv(reliability_path, index=False, encoding="utf-8-sig")
    return str(report_path), str(detail_path), str(reliability_path)
