# -*- coding: utf-8 -*-
"""V8.5 strictly prequential comparison of V5, V8.1, V8.5 and random."""
from __future__ import annotations

import hashlib
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lotto_core import NUMBER_COLS, validate_frame
from v50_predictive_engine import (
    V50_ENGINE_VERSION,
    _fit_from_feature_cache,
    build_number_features,
    generate_tickets,
)
from v81_batch_sampler import generate_batch_quota_portfolio
from v85_hybrid_portfolio import V85_VERSION, generate_probability_weighted_portfolio

METHODS = ("V5", "V8.1", "V8.5", "随机")
METRICS = (
    "mean_hits", "ticket_3plus", "ticket_4plus",
    "batch_3plus", "batch_4plus", "best_hits", "total_hits",
)
PRIMARY_METRIC = "batch_3plus"


def _score_portfolio(tickets: list[list[int]], actual: set[int]) -> dict[str, float]:
    hits = np.asarray(
        [len(set(int(n) for n in ticket).intersection(actual)) for ticket in tickets],
        dtype=float,
    )
    return {
        "mean_hits": float(hits.mean()),
        "ticket_3plus": float(np.mean(hits >= 3)),
        "ticket_4plus": float(np.mean(hits >= 4)),
        "batch_3plus": float(np.any(hits >= 3)),
        "batch_4plus": float(np.any(hits >= 4)),
        "best_hits": float(hits.max()),
        "total_hits": float(hits.sum()),
    }


def _structure(tickets: list[list[int]]) -> dict[str, float]:
    overlaps = [
        len(set(tickets[i]).intersection(tickets[j]))
        for i, j in combinations(range(len(tickets)), 2)
    ]
    return {
        "coverage": float(len({n for ticket in tickets for n in ticket})),
        "max_pairwise_overlap": float(max(overlaps, default=0)),
        "mean_pairwise_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
    }


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
    sampled = x[indices.reshape(int(repetitions), -1)[:, :n]]
    means = sampled.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    p_one_sided = (1.0 + float(np.sum(means <= 0.0))) / (float(repetitions) + 1.0)
    return float(low), float(high), float(p_one_sided)


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.ones(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * float(p_values[int(index)]))
        running = max(running, value)
        adjusted[int(index)] = running
    return adjusted.tolist()


def run_v85_walk_forward(
    frame: pd.DataFrame, *, test_draws: int = 500, training_window: int = 500,
    min_history: int = 180, ridge_alpha: float = 1.0, tickets_per_draw: int = 10,
    seed: int = 85_2026, bootstrap_repetitions: int = 10_000, block_length: int = 10,
) -> dict[str, Any]:
    """Generate every strategy before seeing that target's outcome, then score."""
    started = time.perf_counter()
    clean = validate_frame(frame).reset_index(drop=True)
    if clean.empty:
        raise ValueError("没有可用历史数据，请先手动导入CSV或Excel。")
    draws = clean["draw_no"].to_numpy(dtype=np.int64)
    if draws[0] != 1 or np.any(np.diff(draws) != 1):
        raise ValueError("V8.5要求从第1期开始且期号连续的手动历史数据。")
    matrix = clean[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)
    n = len(matrix)
    test_draws = int(test_draws)
    training_window = int(training_window)
    min_history = int(min_history)
    tickets_per_draw = int(tickets_per_draw)
    if not 1 <= test_draws <= n - min_history:
        raise ValueError("V8.5测试期数超出可用历史范围。")
    if not 30 <= training_window <= 1000 or not 30 <= min_history < n:
        raise ValueError("V8.5训练窗口或最少历史期数无效。")
    if tickets_per_draw != 10:
        raise ValueError("V8.5当前固定比较每期十注组合。")
    if int(bootstrap_repetitions) < 100 or int(block_length) < 1:
        raise ValueError("Bootstrap重复次数至少100，区块长度必须为正数。")

    first_target_idx = max(min_history, n - test_draws)
    cache_start = max(min_history, first_target_idx - training_window)
    feature_cache = {
        idx: build_number_features(matrix[:idx])
        for idx in range(cache_start, n)
    }
    rows: list[dict[str, Any]] = []

    for target_idx in range(first_target_idx, n):
        target_draw = int(draws[target_idx])
        actual = set(int(x) for x in matrix[target_idx])
        probabilities, pair_scores, diagnostics = _fit_from_feature_cache(
            matrix, feature_cache, target_idx, training_window, float(ridge_alpha)
        )
        if not np.isclose(float(probabilities.sum()), 6.0, atol=1e-5):
            raise RuntimeError("V5输出的45号码边际概率总和不为6。")

        target_seed = int(
            np.random.SeedSequence([int(seed), target_draw]).generate_state(1)[0]
        )
        quota_seed = target_seed ^ 0x81A55
        v5_tickets = generate_tickets(
            probabilities, pair_scores, tickets_per_draw, target_seed, pair_strength=0.5
        )
        v81_tickets, v81_report = generate_batch_quota_portfolio(
            probabilities, ticket_count=tickets_per_draw, sample_size=6, seed=quota_seed
        )
        v85_tickets, v85_report = generate_probability_weighted_portfolio(
            probabilities, seed=quota_seed
        )
        random_tickets = generate_tickets(
            np.full(45, 6.0 / 45.0), None, tickets_per_draw,
            target_seed ^ 0x5A17, pair_strength=0.0,
        )

        # Same per-target rounded counts: only the row allocation constraint differs.
        if v81_report["realized_appearances"] != v85_report["realized_appearances"]:
            raise RuntimeError("V8.1与V8.5未使用相同的批次号码配额。")

        portfolios = {
            "V5": v5_tickets, "V8.1": v81_tickets,
            "V8.5": v85_tickets, "随机": random_tickets,
        }
        row: dict[str, Any] = {
            "target_draw": target_draw,
            "training_end_draw": int(draws[target_idx - 1]),
            "training_draws": int(diagnostics["training_draws"]),
            "v5_probability_sum": float(probabilities.sum()),
            "v5_probability_min": float(probabilities.min()),
            "v5_probability_max": float(probabilities.max()),
            "v5_data_prefix_sha256": hashlib.sha256(matrix[:target_idx].tobytes()).hexdigest(),
            "v81_v85_quotas_identical": True,
            "v85_allocation_method": str(v85_report["allocation_method"]),
        }
        for method, tickets in portfolios.items():
            row[f"{method}_tickets"] = json.dumps(tickets, separators=(",", ":"))
            for metric, value in _score_portfolio(tickets, actual).items():
                row[f"{method}_{metric}"] = value
            for metric, value in _structure(tickets).items():
                row[f"{method}_{metric}"] = value
        row["v85_batch_counts_floor_ceil"] = bool(v85_report["batch_counts_are_floor_or_ceil"])
        row["v85_number_coverage"] = int(v85_report["number_coverage"])
        row["v85_max_pairwise_overlap"] = int(v85_report["max_pairwise_overlap"])
        rows.append(row)

    detail = pd.DataFrame(rows)
    comparisons = []
    primary_p = []
    for offset, benchmark in enumerate(("V5", "V8.1", "随机")):
        difference = (
            detail[f"V8.5_{PRIMARY_METRIC}"].to_numpy(dtype=float)
            - detail[f"{benchmark}_{PRIMARY_METRIC}"].to_numpy(dtype=float)
        )
        low, high, p_value = _moving_block_bootstrap(
            difference, repetitions=bootstrap_repetitions, block_length=block_length,
            seed=int(seed) + 7001 + offset * 101,
        )
        primary_p.append(p_value)
        comparisons.append({
            "primary_metric": PRIMARY_METRIC,
            "V8.5_vs": benchmark,
            "difference_mean": float(difference.mean()),
            "moving_block_bootstrap_95ci": [low, high],
            "one_sided_bootstrap_p": p_value,
            "draws_won": int(np.sum(difference > 0)),
            "draws_lost": int(np.sum(difference < 0)),
            "draws_tied": int(np.sum(difference == 0)),
        })
    adjusted = _holm_adjust(primary_p)
    for item, p_adjusted in zip(comparisons, adjusted):
        item["holm_adjusted_p"] = float(p_adjusted)

    metric_summary: dict[str, dict[str, float]] = {}
    for method in METHODS:
        metric_summary[method] = {
            metric: float(detail[f"{method}_{metric}"].mean())
            for metric in (*METRICS, "coverage", "max_pairwise_overlap", "mean_pairwise_overlap")
        }

    structure_compliance = {
        "all_v85_batches_have_ten_unique_tickets": bool(
            (detail["V8.5_tickets"].map(
                lambda value: len({tuple(x) for x in json.loads(value)})
            ) == 10).all()
        ),
        "all_v85_batches_have_pairwise_overlap_at_most_one": bool(
            (detail["v85_max_pairwise_overlap"] <= 1).all()
        ),
        "all_v85_batches_respect_floor_ceil_quotas": bool(
            detail["v85_batch_counts_floor_ceil"].all()
        ),
        "all_v85_batches_match_v81_rounded_quotas": bool(
            detail["v81_v85_quotas_identical"].all()
        ),
    }
    report = {
        "research_version": V85_VERSION,
        "status": "historical_prequential_comparison_not_independent_or_prospective_evidence",
        "target_start_draw": int(detail["target_draw"].iloc[0]),
        "target_end_draw": int(detail["target_draw"].iloc[-1]),
        "test_draws": int(len(detail)),
        "training_window": training_window,
        "min_history": min_history,
        "ridge_alpha": float(ridge_alpha),
        "tickets_per_strategy_per_draw": tickets_per_draw,
        "master_seed": int(seed),
        "v5_engine_version": V50_ENGINE_VERSION,
        "data_fingerprint_sha256": hashlib.sha256(clean.to_csv(index=False).encode("utf-8")).hexdigest(),
        "methods": {
            "V5": "V5 sequential weighted ticket sampling with pair-strength 0.5",
            "V8.1": "dependent-rounded batch quotas with V8.1 row allocation",
            "V8.5": "same V8.1 batch quotas allocated under V8.4 pairwise-overlap-at-most-one structure",
            "随机": "uniform random distinct-ticket control",
        },
        "primary_endpoint": "整批至少一注命中3个或以上号码（batch_3plus）",
        "primary_comparisons": comparisons,
        "summary_by_method": metric_summary,
        "structure_compliance": structure_compliance,
        "evidence_gate": {
            "v85_is_structurally_valid": all(structure_compliance.values()),
            "historical_batch_3plus_comparison_completed": True,
            "predictive_superiority_proven": False,
            "production_model_change_allowed": False,
            "new_forward_lock_created": False,
            "prospective_validation_required": True,
        },
        "evidence_limitations": [
            "The 752-1251 target window is retrospective and has previously been examined in V7/V8 research.",
            "Each target prediction uses only earlier rows, but this aggregate window is not an untouched holdout.",
            "V5 marginal probabilities have not been proven to contain future predictive signal.",
            "V8.5 changes only batch allocation; it does not alter any individual fixed ticket probability.",
            "A better realized batch metric on this retrospective window is not prospective predictive superiority.",
        ],
        "production_model_changed": False,
        "forward_lock_created": False,
        "manual_history_only": True,
        "official_lottery_site_or_api_accessed": False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "interpretation": (
            "This is a strictly prequential historical comparison of five ten-ticket "
            "portfolio methods. The V8.5-vs-V8.1 comparison uses identical randomized "
            "integer batch quotas and changes only the ticket-row allocation constraint. "
            "All results remain retrospective and do not authorize production promotion."
        ),
    }
    return {"report": report, "detail": detail}


def write_v85_artifacts(
    result: dict[str, Any], output_dir: str | None = None,
) -> tuple[str, str]:
    root = Path(output_dir) if output_dir else Path("artifacts") / "v85"
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "v85_hybrid_portfolio_report.json"
    detail_path = root / "v85_hybrid_portfolio_detail.csv"
    report_path.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2), encoding="utf-8")
    result["detail"].to_csv(detail_path, index=False, encoding="utf-8-sig")
    return str(report_path), str(detail_path)
