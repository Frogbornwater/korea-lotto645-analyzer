# -*- coding: utf-8 -*-
"""V8.8: 1,000 independent random ten-ticket portfolios per target draw."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lotto_core import NUMBER_COLS, validate_frame

METHODS = ("V5", "V8.1", "V8.5")
METRICS = ("batch_3plus", "batch_4plus", "mean_hits", "ticket_3plus", "ticket_4plus", "best_hits")
PRIMARY_METRIC = "batch_3plus"
V88_VERSION = "8.8.0-multiple-random-control"


def _score_random_controls(actual: set[int], *, repetitions: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Vectorized uniform sampling: independent portfolios, 10 distinct tickets per portfolio."""
    keys = rng.random((repetitions, 10, 45))
    picks = np.argpartition(keys, 6, axis=2)[:, :, :6] + 1
    # A collision means two identical tickets inside one portfolio. Regenerate only collisions.
    for batch in range(repetitions):
        while len({tuple(sorted(row.tolist())) for row in picks[batch]}) != 10:
            keys[batch] = rng.random((10, 45))
            picks[batch] = np.argpartition(keys[batch], 6, axis=1)[:, :6] + 1
    hits = np.isin(picks, np.asarray(sorted(actual), dtype=np.int16)).sum(axis=2)
    return {
        "batch_3plus": np.any(hits >= 3, axis=1).astype(float),
        "batch_4plus": np.any(hits >= 4, axis=1).astype(float),
        "mean_hits": hits.mean(axis=1),
        "ticket_3plus": (hits >= 3).mean(axis=1),
        "ticket_4plus": (hits >= 4).mean(axis=1),
        "best_hits": hits.max(axis=1).astype(float),
    }


def _mbb(values: np.ndarray, *, repetitions: int, block_length: int, seed: int) -> dict[str, float]:
    x = np.asarray(values, dtype=float).reshape(-1)
    n = len(x)
    if n < 2:
        return {"mean": float(x.mean()), "ci_low": float("nan"), "ci_high": float("nan"), "p_one_sided": 1.0}
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(repetitions, blocks))
    idx = (starts[:, :, None] + np.arange(block_length)[None, None, :]) % n
    means = x[idx.reshape(repetitions, -1)[:, :n]].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_one_sided": float((1 + np.sum(means <= 0)) / (repetitions + 1)),
    }


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.ones(len(p_values), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * float(p_values[int(idx)])))
        adjusted[int(idx)] = running
    return adjusted.tolist()


def run_v88_random_control(
    frame: pd.DataFrame,
    detail: pd.DataFrame,
    *,
    random_portfolios_per_draw: int = 1000,
    seed: int = 88_2026,
    bootstrap_repetitions: int = 10_000,
    block_length: int = 10,
) -> dict[str, Any]:
    """Compare precomputed causal strategies with a large per-draw random null."""
    started = time.perf_counter()
    clean = validate_frame(frame).reset_index(drop=True)
    if clean.empty or clean["draw_no"].iloc[0] != 1 or np.any(np.diff(clean["draw_no"]) != 1):
        raise ValueError("V8.8 requires complete, consecutive manually imported draw history from draw 1.")
    if random_portfolios_per_draw < 100:
        raise ValueError("At least 100 random portfolios per target draw are required.")
    if bootstrap_repetitions < 100 or block_length < 1:
        raise ValueError("Bootstrap repetitions must be >=100 and block length must be positive.")
    required = {"target_draw", *[f"{m}_{metric}" for m in METHODS for metric in METRICS]}
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"V8.8 detail is missing required columns: {sorted(missing)}")
    indexed = clean.set_index("draw_no")
    rows: list[dict[str, Any]] = []
    metric_differences: dict[str, dict[str, list[float]]] = {
        method: {metric: [] for metric in METRICS} for method in METHODS
    }
    percentile_counts = {method: 0 for method in METHODS}
    total_draws = len(detail)

    for row_index, row in detail.reset_index(drop=True).iterrows():
        draw = int(row["target_draw"])
        if draw not in indexed.index:
            raise ValueError(f"Target draw {draw} is absent from manually imported history.")
        actual = set(int(indexed.loc[draw, col]) for col in NUMBER_COLS)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), draw]))
        null = _score_random_controls(actual, repetitions=int(random_portfolios_per_draw), rng=rng)
        summary: dict[str, Any] = {"target_draw": draw, "random_portfolios": int(random_portfolios_per_draw)}
        for metric in METRICS:
            values = null[metric]
            summary[f"random_{metric}_mean"] = float(values.mean())
            summary[f"random_{metric}_sd"] = float(values.std(ddof=1))
            summary[f"random_{metric}_p025"] = float(np.quantile(values, 0.025))
            summary[f"random_{metric}_p975"] = float(np.quantile(values, 0.975))
            summary[f"random_{metric}_p95"] = float(np.quantile(values, 0.95))
            for method in METHODS:
                observed = float(row[f"{method}_{metric}"])
                metric_differences[method][metric].append(observed - float(values.mean()))
                if metric == PRIMARY_METRIC and observed > float(values.mean()):
                    percentile_counts[method] += 1
                if metric == PRIMARY_METRIC:
                    summary[f"{method}_batch3plus"] = observed
                    summary[f"{method}_batch3plus_empirical_percentile"] = float(
                        (np.sum(values < observed) + 0.5 * np.sum(values == observed)) / len(values)
                    )
                    summary[f"{method}_batch3plus_random_exceedance_rate"] = float(
                        (1 + np.sum(values >= observed)) / (len(values) + 1)
                    )
        rows.append(summary)

    per_draw = pd.DataFrame(rows)
    comparisons: list[dict[str, Any]] = []
    primary_p: list[float] = []
    for index, method in enumerate(METHODS):
        diff = np.asarray(metric_differences[method][PRIMARY_METRIC], dtype=float)
        result = _mbb(diff, repetitions=int(bootstrap_repetitions), block_length=int(block_length),
                      seed=int(seed) + 1009 * (index + 1))
        primary_p.append(result["p_one_sided"])
        comparisons.append({
            "method": method,
            "metric": PRIMARY_METRIC,
            "mean_difference_vs_per_draw_random_expectation": result["mean"],
            "moving_block_bootstrap_95ci": [result["ci_low"], result["ci_high"]],
            "one_sided_p_value": result["p_one_sided"],
            "draws_above_random_mean": int(np.sum(diff > 0)),
            "draws_below_random_mean": int(np.sum(diff < 0)),
            "draws_tied_random_mean": int(np.sum(diff == 0)),
        })
    adjusted = _holm(primary_p)
    for item, p_adjusted in zip(comparisons, adjusted):
        item["holm_adjusted_p"] = float(p_adjusted)

    secondary: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        secondary[method] = {}
        for metric in METRICS:
            values = np.asarray(metric_differences[method][metric], dtype=float)
            secondary[method][metric] = {
                "mean_difference_vs_random_expectation": float(values.mean()),
                "draws_above_random_expectation": int(np.sum(values > 0)),
            }

    report = {
        "research_version": V88_VERSION,
        "status": "historical_prequential_replay_with_monte_carlo_random_null_not_prospective_evidence",
        "target_start_draw": int(per_draw["target_draw"].min()),
        "target_end_draw": int(per_draw["target_draw"].max()),
        "test_draws": int(total_draws),
        "random_portfolios_per_draw": int(random_portfolios_per_draw),
        "random_ticket_count_per_portfolio": 10,
        "ticket_size": 6,
        "random_sampling": "independent uniform six-of-45 tickets, without replacement within tickets; duplicate tickets within each ten-ticket portfolio rejected",
        "master_seed": int(seed),
        "primary_endpoint": PRIMARY_METRIC,
        "primary_comparisons": comparisons,
        "secondary_descriptive_comparisons": secondary,
        "fraction_of_draws_above_random_mean": {
            method: float(percentile_counts[method] / total_draws) for method in METHODS
        },
        "inference_design": {
            "statistical_unit": "target draw, not Monte Carlo portfolio",
            "random_controls_are_monte_carlo_estimates_not_additional_draws": True,
            "paired_by_target_draw": True,
            "moving_block_bootstrap_repetitions": int(bootstrap_repetitions),
            "moving_block_length_draws": int(block_length),
            "holm_family": "three primary method-vs-random comparisons",
            "monte_carlo_replicates_per_draw": int(random_portfolios_per_draw),
        },
        "evidence_boundary": {
            "historical_targets_are_not_untouched_holdout": True,
            "predictive_superiority_proven": False,
            "production_model_changed": False,
            "forward_lock_created": False,
            "manual_history_only": True,
            "official_lottery_site_or_api_accessed": False,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "interpretation": "每期1,000组随机组合用于估计该期开奖下的随机组合结果分布；统计推断仍以开奖期为单位。历史逐期回放不能替代未来前瞻验证。",
    }
    return {"report": report, "per_draw": per_draw}


def write_v88_artifacts(result: dict[str, Any], output_dir: str | Path = "artifacts/v88") -> tuple[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "v88_random_control_report.json"
    detail_path = root / "v88_random_control_per_draw.csv"
    if report_path.exists() or detail_path.exists():
        raise FileExistsError("V8.8 report output already exists; refusing to overwrite.")
    report_path.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2), encoding="utf-8")
    result["per_draw"].to_csv(detail_path, index=False, encoding="utf-8-sig")
    return str(report_path), str(detail_path)
