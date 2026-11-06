# -*- coding: utf-8 -*-
"""Joint AutoML optimizer for Lotto 6/45 V3.7.

Offline-only. Searches 500 unique candidate configurations across the seven
parameterized strategies while evaluating uniform random as a paired control.
Uses nested rolling validation and one frozen final configuration on a reserved
holdout. This is a statistical research tool, not a lottery prediction system.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from backtest_engine import (
    _bootstrap_ci,
    _holm_adjust,
    _make_weights,
    _paired_signflip_p,
    _structure_bounds,
)
from lotto_core import NUMBER_COLS, validate_frame
from model_registry import ENGINE_VERSION, MODEL_NAMES, get_model_version

RANDOM_MODEL = "均匀随机基准"
SEARCH_MODELS = [name for name in MODEL_NAMES if name != RANDOM_MODEL]
OBJECTIVE_NAME = "balanced"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _config_key(config: dict[str, Any]) -> str:
    return json.dumps(_jsonable(config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _sample_config(model_name: str, rng: np.random.Generator) -> dict[str, Any]:
    """Sample one configuration from a predeclared search space."""
    if model_name == "近期频率模型":
        return {
            "recent_window": int(rng.choice([30, 50, 80, 100, 150, 200, 300, 500])),
            "recent_prior": round(_log_uniform(rng, 2.0, 40.0), 6),
        }
    if model_name == "长期频率模型":
        return {"long_prior": round(_log_uniform(rng, 5.0, 300.0), 6)}
    if model_name == "指数时间衰减模型":
        return {"half_life": round(_log_uniform(rng, 10.0, 300.0), 6)}
    if model_name == "贝叶斯平滑模型":
        return {"bayes_prior": round(_log_uniform(rng, 5.0, 400.0), 6)}
    if model_name == "综合评分模型":
        weights = rng.dirichlet(np.array([1.5, 1.5, 1.5], dtype=float))
        weights = [round(float(weights[0]), 8), round(float(weights[1]), 8)]
        weights.append(round(1.0 - weights[0] - weights[1], 8))
        short = int(rng.choice([30, 50, 80, 100]))
        medium = int(rng.choice([x for x in [100, 150, 200, 300, 400] if x > short]))
        return {
            "ensemble_weights": weights,
            "ensemble_short_window": short,
            "ensemble_mid_window": medium,
        }
    if model_name == "历史结构约束模型":
        lower = float(rng.uniform(0.005, 0.12))
        upper = float(rng.uniform(max(0.88, lower + 0.55), 0.995))
        return {
            "structure_lower_quantile": round(lower, 6),
            "structure_upper_quantile": round(upper, 6),
        }
    if model_name == "多模型集成模型":
        weights = rng.dirichlet(np.array([1.5, 1.5, 1.5], dtype=float))
        weights = [round(float(weights[0]), 8), round(float(weights[1]), 8)]
        weights.append(round(1.0 - weights[0] - weights[1], 8))
        return {
            "multi_model_weights": weights,
            "recent_window": int(rng.choice([30, 50, 80, 100, 150, 200, 300, 500])),
            "recent_prior": round(_log_uniform(rng, 2.0, 40.0), 6),
            "bayes_prior": round(_log_uniform(rng, 5.0, 400.0), 6),
            "half_life": round(_log_uniform(rng, 10.0, 300.0), 6),
        }
    raise ValueError(f"未支持联合参数搜索的模型：{model_name}")


def generate_joint_configs(
    budget: int = 500, seed: int = 2026
) -> list[dict[str, Any]]:
    """Create exactly `budget` unique configs, balanced across all 7 tunable models.

    Uniform random is deliberately not parameter-searched; it is evaluated as a
    common paired benchmark in every inner fold, outer fold and final holdout.
    """
    budget = int(budget)
    if budget < len(SEARCH_MODELS):
        raise ValueError(f"参数预算至少需要 {len(SEARCH_MODELS)}，才能覆盖全部可调模型。")
    rng = np.random.default_rng(int(seed))
    base, remainder = divmod(budget, len(SEARCH_MODELS))
    allocations = {name: base + (idx < remainder) for idx, name in enumerate(SEARCH_MODELS)}
    configs: list[dict[str, Any]] = []
    next_id = 1
    for model_name in SEARCH_MODELS:
        target = int(allocations[model_name])
        seen: set[str] = set()
        attempts = 0
        while len(seen) < target:
            attempts += 1
            if attempts > target * 100:
                raise RuntimeError(f"无法为{model_name}生成足够的唯一参数配置。")
            params = _sample_config(model_name, rng)
            key = _config_key(params)
            if key in seen:
                continue
            seen.add(key)
            configs.append({
                "config_id": next_id,
                "model_name": model_name,
                "model_version": get_model_version(model_name),
                "params": params,
            })
            next_id += 1
    return configs


def _draw_tickets_fast(
    rng: np.random.Generator,
    weights: np.ndarray,
    count: int,
    structure_bounds: tuple[float, float] | None = None,
) -> list[tuple[int, ...]]:
    """Vectorized Plackett-Luce weighted sampling without replacement.

    Exponential-race keys, -log(U)/weight, implement a Plackett-Luce
    weighted-without-replacement draw while generating many tickets in one
    NumPy operation. This is distributionally defined, not bit-for-bit
    equivalent to NumPy's separate choice implementation.
    """
    probs = np.asarray(weights, dtype=float).copy()
    if probs.shape != (45,) or not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("模型权重必须是45个有限且非负的数值。")
    if probs.sum() <= 0:
        raise ValueError("模型权重总和必须大于0。")
    probs = np.maximum(probs / probs.sum(), 1e-15)
    chosen: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    max_candidates = max(20_000, int(count) * 2_000)
    while len(chosen) < count and attempts < max_candidates:
        remaining = count - len(chosen)
        batch_size = max(32, remaining * (5 if structure_bounds is not None else 3))
        uniforms = np.maximum(rng.random((batch_size, 45)), 1e-15)
        race_keys = -np.log(uniforms) / probs[None, :]
        picks = np.argpartition(race_keys, kth=5, axis=1)[:, :6] + 1
        picks.sort(axis=1)
        attempts += batch_size
        for row in picks:
            ticket = tuple(int(x) for x in row)
            if ticket in seen:
                continue
            if structure_bounds is not None:
                total = sum(ticket)
                odd = sum(n % 2 for n in ticket)
                low = sum(n <= 15 for n in ticket)
                middle = sum(16 <= n <= 30 for n in ticket)
                high = sum(n >= 31 for n in ticket)
                if not (structure_bounds[0] <= total <= structure_bounds[1]
                        and 1 <= odd <= 5 and low >= 1 and middle >= 1 and high >= 1):
                    continue
            seen.add(ticket)
            chosen.append(ticket)
            if len(chosen) >= count:
                break
    if len(chosen) != count:
        raise RuntimeError(
            f"向量化抽样未能在{max_candidates:,}个候选内凑齐{count}注；"
            "请检查结构约束或增加训练数据。"
        )
    return chosen


def _metrics(tickets: list[tuple[int, ...]], actual: set[int]) -> dict[str, float]:
    ticket_array = np.asarray(tickets, dtype=np.int16)
    actual_array = np.asarray(sorted(actual), dtype=np.int16)
    hits = np.isin(ticket_array, actual_array).sum(axis=1)
    bitmasks = [sum(1 << int(number) for number in ticket) for ticket in tickets]
    overlaps = [
        (bitmasks[i] & bitmasks[j]).bit_count()
        for i in range(len(bitmasks)) for j in range(i + 1, len(bitmasks))
    ]
    return {
        "mean_hits": float(np.mean(hits)),
        "single_3plus": float(np.mean(hits >= 3)),
        "single_4plus": float(np.mean(hits >= 4)),
        "batch_3plus": float(np.any(hits >= 3)),
        "batch_max_hits": float(np.max(hits)) if len(hits) else 0.0,
        "coverage": float(np.unique(ticket_array).size),
        "mean_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
    }


def _weights_for_config(
    past: np.ndarray,
    config: dict[str, Any],
    minimum_train: int,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    model_name = str(config["model_name"])
    params = config["params"]
    weights = _make_weights(
        past,
        model_name,
        int(params.get("recent_window", minimum_train)),
        float(params.get("half_life", 40.0)),
        float(params.get("bayes_prior", 80.0)),
        tuple(params.get("ensemble_weights", (0.40, 0.35, 0.25))),
        tuple(params.get("multi_model_weights", (0.40, 0.35, 0.25))),
        float(params.get("recent_prior", 8.0)),
        float(params.get("long_prior", 40.0)),
        int(params.get("ensemble_short_window", 50)),
        int(params.get("ensemble_mid_window", 200)),
    )
    bounds = None
    if model_name == "历史结构约束模型":
        q_low = float(params.get("structure_lower_quantile", 0.02))
        q_high = float(params.get("structure_upper_quantile", 0.98))
        bounds = _structure_bounds(past[-min(len(past), minimum_train):], q_low, q_high)
    return weights, bounds


def _random_batch(
    matrix: np.ndarray,
    target_idx: int,
    seed_idx: int,
    tickets_per_draw: int,
    seed: int,
    cache: dict[tuple[int, int], tuple[list[tuple[int, ...]], dict[str, float]]],
) -> tuple[list[tuple[int, ...]], dict[str, float]]:
    key = (int(target_idx), int(seed_idx))
    if key not in cache:
        # Common random numbers: every candidate sees the same random benchmark
        # tickets for the same target draw and seed group.
        target_draw = target_idx  # draw index, mixed with a fixed random stream id below
        stream = np.random.SeedSequence([int(seed), int(target_draw), int(seed_idx + 1), 777_777])
        rng = np.random.default_rng(stream)
        tickets = _draw_tickets_fast(rng, np.ones(45, dtype=float), tickets_per_draw)
        cache[key] = (tickets, {})
    return cache[key]


def _make_inner_folds(
    prefix_length: int,
    min_train: int,
    inner_folds: int,
    inner_validation_draws: int,
) -> list[tuple[int, list[int]]]:
    needed = int(inner_folds) * int(inner_validation_draws)
    start = int(prefix_length) - needed
    if start < int(min_train):
        raise ValueError(
            "历史数据不足以构建嵌套内层验证：每个外层训练前缀至少需要 "
            f"最低训练 {min_train} 期 + {inner_folds}×{inner_validation_draws} 期内层验证。"
        )
    return [
        (fold_idx + 1, list(range(start + fold_idx * inner_validation_draws,
                                  start + (fold_idx + 1) * inner_validation_draws)))
        for fold_idx in range(inner_folds)
    ]


def _evaluate_config_set(
    matrix: np.ndarray,
    draw_numbers: np.ndarray,
    folds: list[tuple[int, list[int]]],
    configs: list[dict[str, Any]],
    minimum_train: int,
    tickets_per_draw: int,
    n_seeds: int,
    seed: int,
    random_cache: dict,
    stage_name: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
    progress_start: int = 0,
    progress_total: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Evaluate all unique configs over nested inner folds; one period is unit."""
    period_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    total_units = max(1, len(configs) * len(folds))
    completed_units = 0

    for config in configs:
        config_id = int(config["config_id"])
        model_name = str(config["model_name"])
        params_json = json.dumps(_jsonable(config["params"]), ensure_ascii=False, sort_keys=True)
        strategy_index = MODEL_NAMES.index(model_name) + 1
        for inner_fold_id, target_indices in folds:
            model_period_metrics: list[dict[str, float]] = []
            random_period_metrics: list[dict[str, float]] = []
            for target_idx in target_indices:
                past = matrix[:target_idx]
                actual = set(int(x) for x in matrix[target_idx])
                weights, bounds = _weights_for_config(past, config, minimum_train)
                model_seed_metrics: list[dict[str, float]] = []
                random_seed_metrics: list[dict[str, float]] = []
                for seed_idx in range(n_seeds):
                    model_stream = np.random.SeedSequence([
                        int(seed), int(draw_numbers[target_idx]), int(seed_idx + 1),
                        int(strategy_index), int(config_id), int(inner_fold_id),
                    ])
                    model_rng = np.random.default_rng(model_stream)
                    model_tickets = _draw_tickets_fast(model_rng, weights, tickets_per_draw, bounds)
                    model_seed_metrics.append(_metrics(model_tickets, actual))

                    random_tickets, _ = _random_batch(
                        matrix, target_idx, seed_idx, tickets_per_draw, seed, random_cache
                    )
                    random_seed_metrics.append(_metrics(random_tickets, actual))

                model_avg = {k: float(np.mean([x[k] for x in model_seed_metrics])) for k in model_seed_metrics[0]}
                random_avg = {k: float(np.mean([x[k] for x in random_seed_metrics])) for k in random_seed_metrics[0]}
                row = {
                    "配置编号": config_id,
                    "模型": model_name,
                    "模型版本": config["model_version"],
                    "参数JSON": params_json,
                    "内层折": int(inner_fold_id),
                    "期号": int(draw_numbers[target_idx]),
                    **{f"模型_{k}": v for k, v in model_avg.items()},
                    **{f"随机_{k}": v for k, v in random_avg.items()},
                }
                row["平均命中差_vs随机"] = model_avg["mean_hits"] - random_avg["mean_hits"]
                row["单注3+率差_vs随机"] = model_avg["single_3plus"] - random_avg["single_3plus"]
                row["整批3+差_vs随机"] = model_avg["batch_3plus"] - random_avg["batch_3plus"]
                row["号码覆盖差_vs随机"] = model_avg["coverage"] - random_avg["coverage"]
                row["组间重号减少_vs随机"] = random_avg["mean_overlap"] - model_avg["mean_overlap"]
                period_rows.append(row)
                model_period_metrics.append(model_avg)
                random_period_metrics.append(random_avg)

            # Equal weighting of periods after averaging seeds within each period.
            model_fold = {k: float(np.mean([x[k] for x in model_period_metrics])) for k in model_period_metrics[0]}
            random_fold = {k: float(np.mean([x[k] for x in random_period_metrics])) for k in random_period_metrics[0]}
            fold_rows.append({
                "配置编号": config_id,
                "模型": model_name,
                "模型版本": config["model_version"],
                "参数JSON": params_json,
                "内层折": int(inner_fold_id),
                "验证期数": int(len(target_indices)),
                "验证开始期号": int(draw_numbers[target_indices[0]]),
                "验证结束期号": int(draw_numbers[target_indices[-1]]),
                **{f"模型_{k}": v for k, v in model_fold.items()},
                **{f"随机_{k}": v for k, v in random_fold.items()},
                "平均命中差_vs随机": model_fold["mean_hits"] - random_fold["mean_hits"],
                "单注3+率差_vs随机": model_fold["single_3plus"] - random_fold["single_3plus"],
                "整批3+差_vs随机": model_fold["batch_3plus"] - random_fold["batch_3plus"],
                "号码覆盖差_vs随机": model_fold["coverage"] - random_fold["coverage"],
                "组间重号减少_vs随机": random_fold["mean_overlap"] - model_fold["mean_overlap"],
            })
        completed_units += 1
        if progress_callback is not None:
            progress_callback(
                progress_start + completed_units,
                progress_total,
                stage_name,
            )

    period_detail = pd.DataFrame(period_rows)
    fold_detail = pd.DataFrame(fold_rows)
    return period_detail, fold_detail, {}


def _pareto_mask(values: np.ndarray) -> np.ndarray:
    """Return non-dominated rows when every objective is maximized."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    result = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.all(np.isfinite(values[i])):
            result[i] = False
            continue
        for j in range(n):
            if i == j or not np.all(np.isfinite(values[j])):
                continue
            if np.all(values[j] >= values[i]) and np.any(values[j] > values[i]):
                result[i] = False
                break
    return result


OBJECTIVE_COLUMNS = [
    "平均命中差_vs随机",
    "单注3+率差_vs随机",
    "整批3+差_vs随机",
    "号码覆盖差_vs随机",
    "组间重号减少_vs随机",
    "内层折间稳定性",
]


def _summarize_configs(
    fold_detail: pd.DataFrame,
    configs: list[dict[str, Any]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for config in configs:
        cid = int(config["config_id"])
        subset = fold_detail[fold_detail["配置编号"] == cid]
        if subset.empty:
            continue
        record: dict[str, Any] = {
            "配置编号": cid,
            "模型": config["model_name"],
            "模型版本": config["model_version"],
            "参数JSON": json.dumps(_jsonable(config["params"]), ensure_ascii=False, sort_keys=True),
            "内层验证折数": int(subset["内层折"].nunique()),
            "内层验证期数": int(subset["验证期数"].sum()),
        }
        metric_map = {
            "平均命中数": "模型_mean_hits",
            "单注3+命中率": "模型_single_3plus",
            "单注4+命中率": "模型_single_4plus",
            "整批至少一注3+比例": "模型_batch_3plus",
            "平均每批最高命中数": "模型_batch_max_hits",
            "平均每批号码覆盖数": "模型_coverage",
            "平均组间重号数": "模型_mean_overlap",
            "随机平均命中数": "随机_mean_hits",
        }
        for output, source in metric_map.items():
            record[output] = float(np.average(subset[source], weights=subset["验证期数"]))
        for col in [
            "平均命中差_vs随机", "单注3+率差_vs随机", "整批3+差_vs随机",
            "号码覆盖差_vs随机", "组间重号减少_vs随机",
        ]:
            record[col] = float(np.average(subset[col], weights=subset["验证期数"]))
        record["内层折间稳定性"] = -float(subset["平均命中差_vs随机"].std(ddof=0))
        records.append(record)
    summary = pd.DataFrame(records)
    if summary.empty:
        return summary

    # Balanced objective: equally weighted percentile ranks across six separate
    # dimensions; this is a selection heuristic, not a statistical significance score.
    rank_cols: list[str] = []
    for idx, col in enumerate(OBJECTIVE_COLUMNS):
        rank_col = f"目标百分位_{idx + 1}"
        summary[rank_col] = summary[col].rank(pct=True, method="average")
        rank_cols.append(rank_col)
    summary["balanced_score"] = summary[rank_cols].mean(axis=1)

    # Append the paired uniform baseline as a visible Pareto reference point.
    # It is not a tunable candidate and therefore receives no balanced rank.
    first_cfg = int(configs[0]["config_id"])
    baseline_rows = fold_detail[fold_detail["配置编号"] == first_cfg]
    baseline_record: dict[str, Any] = {
        "配置编号": 0, "模型": RANDOM_MODEL, "模型版本": get_model_version(RANDOM_MODEL),
        "参数JSON": json.dumps({"role": "paired_uniform_random_control"}),
        "内层验证折数": int(baseline_rows["内层折"].nunique()),
        "内层验证期数": int(baseline_rows["验证期数"].sum()),
        "平均命中数": float(np.average(baseline_rows["随机_mean_hits"], weights=baseline_rows["验证期数"])),
        "单注3+命中率": float(np.average(baseline_rows["随机_single_3plus"], weights=baseline_rows["验证期数"])),
        "单注4+命中率": float(np.average(baseline_rows["随机_single_4plus"], weights=baseline_rows["验证期数"])),
        "整批至少一注3+比例": float(np.average(baseline_rows["随机_batch_3plus"], weights=baseline_rows["验证期数"])),
        "平均每批最高命中数": float(np.average(baseline_rows["随机_batch_max_hits"], weights=baseline_rows["验证期数"])),
        "平均每批号码覆盖数": float(np.average(baseline_rows["随机_coverage"], weights=baseline_rows["验证期数"])),
        "平均组间重号数": float(np.average(baseline_rows["随机_mean_overlap"], weights=baseline_rows["验证期数"])),
        "随机平均命中数": float(np.average(baseline_rows["随机_mean_hits"], weights=baseline_rows["验证期数"])),
        "平均命中差_vs随机": 0.0, "单注3+率差_vs随机": 0.0,
        "整批3+差_vs随机": 0.0, "号码覆盖差_vs随机": 0.0,
        "组间重号减少_vs随机": 0.0, "内层折间稳定性": 0.0,
        "balanced_score": np.nan, "是否Pareto前沿": False, "是否基准": True,
    }
    for rank_col in rank_cols:
        baseline_record[rank_col] = np.nan
    baseline_frame = pd.DataFrame([baseline_record])
    candidates = summary.copy()
    candidates["是否基准"] = False
    combined = pd.concat([candidates, baseline_frame], ignore_index=True, sort=False)
    combined["是否Pareto前沿"] = _pareto_mask(combined[OBJECTIVE_COLUMNS].to_numpy(dtype=float))
    combined = combined.sort_values(
        ["是否基准", "是否Pareto前沿", "balanced_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    combined.insert(0, "联合排名", [i + 1 if not is_base else 0 for i, is_base in enumerate(combined["是否基准"])])
    return combined


def _select_balanced(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        raise RuntimeError("联合优化没有产生有效参数结果。")
    candidates = summary[summary["配置编号"] != 0]
    front = candidates[candidates["是否Pareto前沿"]]
    pool = front if not front.empty else candidates
    chosen = pool.sort_values("balanced_score", ascending=False).iloc[0]
    return {
        "config_id": int(chosen["配置编号"]),
        "model_name": str(chosen["模型"]),
        "model_version": str(chosen["模型版本"]),
        "params": json.loads(chosen["参数JSON"]),
        "balanced_score": float(chosen["balanced_score"]),
        "pareto_front": bool(chosen["是否Pareto前沿"]),
        "selection_rule": "仅在内层验证 Pareto 前沿中选取 balanced_score 最高的单一配置；不读取外层测试块和最终保留区间。",
    }


def _evaluate_frozen_config(
    matrix: np.ndarray,
    draw_numbers: np.ndarray,
    target_indices: list[int],
    config: dict[str, Any],
    minimum_train: int,
    tickets_per_draw: int,
    n_seeds: int,
    seed: int,
    random_cache: dict,
    fold_label: str,
    include_tickets: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_obj = {
        "config_id": int(config["config_id"]),
        "model_name": str(config["model_name"]),
        "model_version": str(config["model_version"]),
        "params": dict(config["params"]),
    }
    model_name = config_obj["model_name"]
    model_idx = MODEL_NAMES.index(model_name) + 1
    period_rows: list[dict[str, Any]] = []
    ticket_rows: list[dict[str, Any]] = []
    for target_idx in target_indices:
        past = matrix[:target_idx]
        actual = set(int(x) for x in matrix[target_idx])
        target_draw = int(draw_numbers[target_idx])
        weights, bounds = _weights_for_config(past, config_obj, minimum_train)
        model_metrics_seed: list[dict[str, float]] = []
        random_metrics_seed: list[dict[str, float]] = []
        for seed_idx in range(n_seeds):
            model_stream = np.random.SeedSequence([
                int(seed), target_draw, seed_idx + 1, model_idx,
                int(config_obj["config_id"]), 98_765,
            ])
            model_rng = np.random.default_rng(model_stream)
            model_tickets = _draw_tickets_fast(model_rng, weights, tickets_per_draw, bounds)
            random_tickets, _ = _random_batch(
                matrix, target_idx, seed_idx, tickets_per_draw, seed, random_cache
            )
            model_metrics_seed.append(_metrics(model_tickets, actual))
            random_metrics_seed.append(_metrics(random_tickets, actual))
            if include_tickets:
                for strategy, tickets in [(model_name, model_tickets), (RANDOM_MODEL, random_tickets)]:
                    for ticket_no, ticket in enumerate(tickets, start=1):
                        ticket_hits = len(set(ticket).intersection(actual))
                        row = {
                            "验证阶段": fold_label,
                            "期号": target_draw,
                            "随机种子组": seed_idx + 1,
                            "策略": strategy,
                            "配置编号": config_obj["config_id"] if strategy != RANDOM_MODEL else 0,
                            "参数JSON": json.dumps(_jsonable(config_obj["params"]), ensure_ascii=False, sort_keys=True) if strategy != RANDOM_MODEL else "{}",
                            "注序": ticket_no,
                            "号码": "-".join(f"{n:02d}" for n in ticket),
                            "号码JSON": json.dumps(list(ticket), ensure_ascii=False),
                            "命中数": ticket_hits,
                            "命中3个以上": int(ticket_hits >= 3),
                            "命中4个以上": int(ticket_hits >= 4),
                        }
                        row.update({f"n{i}": ticket[i - 1] for i in range(1, 7)})
                        ticket_rows.append(row)
        model_avg = {k: float(np.mean([x[k] for x in model_metrics_seed])) for k in model_metrics_seed[0]}
        random_avg = {k: float(np.mean([x[k] for x in random_metrics_seed])) for k in random_metrics_seed[0]}
        row = {
            "验证阶段": fold_label,
            "期号": target_draw,
            "模型": model_name,
            **{f"模型_{k}": v for k, v in model_avg.items()},
            **{f"随机_{k}": v for k, v in random_avg.items()},
        }
        row["平均命中差_vs随机"] = model_avg["mean_hits"] - random_avg["mean_hits"]
        row["单注3+率差_vs随机"] = model_avg["single_3plus"] - random_avg["single_3plus"]
        row["整批3+差_vs随机"] = model_avg["batch_3plus"] - random_avg["batch_3plus"]
        row["号码覆盖差_vs随机"] = model_avg["coverage"] - random_avg["coverage"]
        row["组间重号减少_vs随机"] = random_avg["mean_overlap"] - model_avg["mean_overlap"]
        period_rows.append(row)
    return pd.DataFrame(period_rows), pd.DataFrame(ticket_rows)


def _summarize_periods(period_detail: pd.DataFrame, seed: int) -> pd.DataFrame:
    if period_detail.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(int(seed) + 600_013)
    deltas = {
        "平均命中差_vs随机": period_detail["平均命中差_vs随机"].to_numpy(dtype=float),
        "单注3+率差_vs随机": period_detail["单注3+率差_vs随机"].to_numpy(dtype=float),
        "整批3+差_vs随机": period_detail["整批3+差_vs随机"].to_numpy(dtype=float),
    }
    rows = []
    for col, values in deltas.items():
        lo, hi = _bootstrap_ci(values, rng, 5000)
        p = _paired_signflip_p(values, rng, 10000)
        rows.append({"指标": col, "平均差": float(np.mean(values)), "95%CI下限": lo, "95%CI上限": hi, "配对检验p值": p})
    p_adjusted = _holm_adjust([x["配对检验p值"] for x in rows])
    for row, adjusted in zip(rows, p_adjusted):
        row["Holm校正p值"] = adjusted
    return pd.DataFrame(rows)


def run_joint_optimization(
    frame: pd.DataFrame,
    budget: int = 500,
    tickets_per_draw: int = 10,
    n_seeds: int = 3,
    minimum_train: int = 300,
    holdout_draws: int = 100,
    outer_folds: int = 2,
    outer_validation_draws: int = 30,
    inner_folds: int = 2,
    inner_validation_draws: int = 5,
    seed: int = 2026,
    objective: str = OBJECTIVE_NAME,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run 500-config, all-model nested rolling search with one final holdout.

    The 7 parameterized strategies receive a balanced allocation of 500 unique
    configurations. Uniform random is generated once per draw/seed and is the
    paired control for all configurations. One balanced winner is frozen using
    development data, then evaluated exactly once on the reserved tail.
    """
    started = time.perf_counter()
    if objective != OBJECTIVE_NAME:
        raise ValueError("V3.7 当前正式支持 balanced 综合平衡目标。")
    clean = validate_frame(frame)
    if clean.empty:
        raise ValueError("没有可用历史数据，请手动导入 CSV 或 Excel。")
    budget, tickets_per_draw, n_seeds = int(budget), int(tickets_per_draw), int(n_seeds)
    minimum_train, holdout_draws = int(minimum_train), int(holdout_draws)
    outer_folds, outer_validation_draws = int(outer_folds), int(outer_validation_draws)
    inner_folds, inner_validation_draws = int(inner_folds), int(inner_validation_draws)
    if budget < len(SEARCH_MODELS) or budget > 2000:
        raise ValueError(f"参数预算必须介于{len(SEARCH_MODELS)}至2000组之间。")
    if tickets_per_draw < 1 or tickets_per_draw > 20:
        raise ValueError("每期每个策略的模拟注数必须介于1至20注。")
    if n_seeds < 1 or n_seeds > 10:
        raise ValueError("随机种子组数必须介于1至10组。")
    if min(minimum_train, holdout_draws, outer_folds, outer_validation_draws,
           inner_folds, inner_validation_draws) < 1:
        raise ValueError("训练期数、保留期数和内外层折数必须为正数。")
    if holdout_draws < 30:
        raise ValueError("独立保留测试集至少需要30期，建议100期。")

    draw_numbers = clean["draw_no"].to_numpy(dtype=np.int64)
    if np.any(np.diff(draw_numbers) != 1):
        raise ValueError("手动历史数据存在缺期，无法运行嵌套滚动验证。请先补齐期号。")
    matrix = clean[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)
    development_length = len(clean) - holdout_draws
    outer_total = outer_folds * outer_validation_draws
    earliest_outer_train = development_length - outer_total
    minimum_required = minimum_train + inner_folds * inner_validation_draws
    if earliest_outer_train < minimum_required:
        raise ValueError(
            "历史数据不足以完成指定嵌套滚动验证。当前需要至少 "
            f"{holdout_draws}期独立保留 + {outer_total}期外层测试 + "
            f"{minimum_required}期最早外层训练/内层验证数据；当前仅有{len(clean)}期。"
        )

    configs = generate_joint_configs(budget=budget, seed=seed)
    allocations: dict[str, int] = {}
    for item in configs:
        allocations[item["model_name"]] = allocations.get(item["model_name"], 0) + 1
    random_cache: dict[tuple[int, int], tuple[list[tuple[int, ...]], dict[str, float]]] = {}
    total_units = (outer_folds + 1) * budget + outer_folds + 1
    progress_done = 0

    def update_progress(label: str) -> None:
        nonlocal progress_done
        progress_done += 1
        if progress_callback is not None:
            progress_callback(progress_done, total_units, label)

    outer_period_tables: list[pd.DataFrame] = []
    outer_ticket_tables: list[pd.DataFrame] = []
    outer_selection_rows: list[dict[str, Any]] = []
    outer_start = development_length - outer_total

    # Nested outer folds: each fold searches only inside its own historical prefix.
    for outer_fold in range(outer_folds):
        outer_test_start = outer_start + outer_fold * outer_validation_draws
        outer_test_end = outer_test_start + outer_validation_draws
        outer_train_length = outer_test_start
        inner_splits = _make_inner_folds(
            outer_train_length, minimum_train, inner_folds, inner_validation_draws
        )
        period_search, fold_search, _ = _evaluate_config_set(
            matrix[:development_length], draw_numbers[:development_length],
            inner_splits, configs, minimum_train, tickets_per_draw, n_seeds,
            seed, random_cache, f"外层{outer_fold + 1}折内层搜索",
            progress_callback=progress_callback,
            progress_start=progress_done,
            progress_total=total_units,
        )
        progress_done += budget
        search_summary = _summarize_configs(fold_search, configs)
        chosen = _select_balanced(search_summary)
        chosen_config = next(x for x in configs if x["config_id"] == chosen["config_id"])
        outer_indices = list(range(outer_test_start, outer_test_end))
        outer_period, outer_tickets = _evaluate_frozen_config(
            matrix, draw_numbers, outer_indices, chosen_config, minimum_train,
            tickets_per_draw, n_seeds, seed, random_cache,
            f"外层{outer_fold + 1}折测试", include_tickets=False,
        )
        outer_period_tables.append(outer_period)
        # The model selection procedure is nested; store only the selected
        # candidate's aggregate period data, not all 500 outer-fold predictions.
        outer_selection_rows.append({
            "外层折": outer_fold + 1,
            "外层训练截止期号": int(draw_numbers[outer_test_start - 1]),
            "外层测试开始期号": int(draw_numbers[outer_test_start]),
            "外层测试结束期号": int(draw_numbers[outer_test_end - 1]),
            "选择模型": chosen["model_name"],
            "配置编号": chosen["config_id"],
            "参数JSON": json.dumps(chosen["params"], ensure_ascii=False, sort_keys=True),
            "内层balanced_score": chosen["balanced_score"],
            "内层Pareto前沿": chosen["pareto_front"],
        })
        update_progress(f"完成外层第{outer_fold + 1}/{outer_folds}折验证")

    outer_detail = pd.concat(outer_period_tables, ignore_index=True) if outer_period_tables else pd.DataFrame()
    outer_selections = pd.DataFrame(outer_selection_rows)
    outer_summary = _summarize_periods(outer_detail, seed + 71)

    # Final development search: all 500 configs, using only the development prefix.
    final_inner_splits = _make_inner_folds(
        development_length, minimum_train, inner_folds, inner_validation_draws
    )
    final_period_detail, final_fold_detail, _ = _evaluate_config_set(
        matrix[:development_length], draw_numbers[:development_length],
        final_inner_splits, configs, minimum_train, tickets_per_draw, n_seeds,
        seed, random_cache, "最终开发集参数选择",
        progress_callback=progress_callback,
        progress_start=progress_done,
        progress_total=total_units,
    )
    progress_done += budget
    final_summary = _summarize_configs(final_fold_detail, configs)
    final_selected = _select_balanced(final_summary)
    selected_config = next(x for x in configs if x["config_id"] == final_selected["config_id"])
    pareto_front = final_summary[final_summary["是否Pareto前沿"]].copy().reset_index(drop=True)
    update_progress("完成最终开发集选型")

    # One frozen selection is evaluated once on the untouched tail holdout.
    holdout_indices = list(range(development_length, len(clean)))
    holdout_detail, holdout_tickets = _evaluate_frozen_config(
        matrix, draw_numbers, holdout_indices, selected_config, minimum_train,
        tickets_per_draw, n_seeds, seed, random_cache,
        "独立保留测试集", include_tickets=True,
    )
    holdout_summary = _summarize_periods(holdout_detail, seed + 93)
    holdout_start_draw = int(draw_numbers[development_length])
    holdout_end_draw = int(draw_numbers[-1])
    update_progress("完成独立保留测试集")

    data_fingerprint = hashlib.sha256(clean.to_csv(index=False).encode("utf-8")).hexdigest()
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "engine_version": ENGINE_VERSION,
        "experiment_type": "v37_joint_nested_automl",
        "experiment_id": f"lotto645-v37-joint-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{seed}",
        "created_at_utc": created_at,
        "network_policy": "offline_only_manual_csv_excel_no_official_website_api_or_scraping",
        "objective": OBJECTIVE_NAME,
        "model_count_including_control": len(MODEL_NAMES),
        "parameterized_model_count": len(SEARCH_MODELS),
        "models_in_scope": MODEL_NAMES,
        "uniform_random_role": "每个目标期和随机种子组共用的配对随机基准；不消耗500个可调配置预算。",
        "unique_configuration_budget": budget,
        "configuration_allocation": allocations,
        "tickets_per_draw_per_model": tickets_per_draw,
        "seed_groups": n_seeds,
        "base_seed": int(seed),
        "minimum_training_draws": minimum_train,
        "outer_folds": outer_folds,
        "outer_validation_draws_per_fold": outer_validation_draws,
        "inner_folds": inner_folds,
        "inner_validation_draws_per_fold": inner_validation_draws,
        "holdout_draws": holdout_draws,
        "data_first_draw": int(draw_numbers[0]),
        "data_last_draw": int(draw_numbers[-1]),
        "data_rows": int(len(clean)),
        "data_fingerprint_sha256": data_fingerprint,
        "development_end_draw": int(draw_numbers[development_length - 1]),
        "holdout_start_draw": holdout_start_draw,
        "holdout_end_draw": holdout_end_draw,
        "holdout_evaluated_once_after_selection": True,
        "holdout_consumed": True,
        "nested_validation_rule": "每个外层折仅使用其测试块之前的数据进行内层500配置搜索；选定单一balanced Pareto配置后评估该折未来测试块。",
        "final_selection_rule": "在全部开发集内层滚动验证的Pareto前沿中，选择balanced_score最高的单一配置；随后仅对该配置评估一次最新独立保留区间。",
        "balanced_objectives": OBJECTIVE_COLUMNS,
        "balanced_score_definition": "六个目标的百分位排名等权平均，仅作为多目标选择启发式，不是胜率或显著性分数。",
        "pareto_rule": "六个目标均最大化；不存在同时在所有目标不差且至少一个目标更好的候选时，该候选位于Pareto前沿。",
        "selected_configuration": final_selected,
        "outer_fold_selections": outer_selection_rows,
        "outer_validation_summary": outer_summary.to_dict(orient="records"),
        "holdout_summary": holdout_summary.to_dict(orient="records"),
        "warning": "历史开奖公平且独立时，任意固定六号码组合的理论头奖概率相同。搜索结果可能受到有限样本、随机波动和选择偏差影响；独立保留集已使用，不得反复调参。",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    return {
        "search_summary": final_summary,
        "pareto_front": pareto_front,
        "inner_fold_detail": final_fold_detail,
        "inner_period_detail": final_period_detail,
        "outer_detail": outer_detail,
        "outer_selections": outer_selections,
        "outer_summary": outer_summary,
        "holdout_detail": holdout_detail,
        "holdout_ticket_detail": holdout_tickets,
        "holdout_summary": holdout_summary,
        "selected_config": final_selected,
        "metadata": metadata,
    }
