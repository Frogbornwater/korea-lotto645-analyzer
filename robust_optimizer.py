# -*- coding: utf-8 -*-
"""V3.8 robust rolling research engine for Lotto 6/45.

This module separates predictive-performance selection from ticket-portfolio
structure analysis. It uses only manually supplied historical data; it does
not retrieve or synchronize lottery results from any external source.

The default retrospective cutoff (draw 1088) is deliberate: the user's V3.7
outer-validation draws 1089-1148 and final holdout 1149-1248 were already
reviewed. V3.8 must not recycle those exposed outcomes as a fresh retrospective
test. These later draws may still be used as known history when creating future
predictions, but not as a new unseen test set.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import time
from typing import Any, Callable
import io
import zipfile

import numpy as np
import pandas as pd

from joint_optimizer import (
    RANDOM_MODEL,
    SEARCH_MODELS,
    _draw_tickets_fast,
    _evaluate_config_set,
    _evaluate_frozen_config,
    _jsonable,
    _make_inner_folds,
    _pareto_mask,
    _random_batch,
    _summarize_periods,
    _weights_for_config,
)
from lotto_core import NUMBER_COLS, validate_frame
from model_registry import ENGINE_VERSION, get_model_version

KNOWN_V37_REVIEWED_START_DRAW = 1089
KNOWN_V37_REVIEWED_END_DRAW = 1248
V38_RETROSPECTIVE_CUTOFF_DRAW = KNOWN_V37_REVIEWED_START_DRAW - 1
PREDICTIVE_OBJECTIVES = [
    "平均命中差_vs随机",
    "单注3+率差_vs随机",
    "整批3+差_vs随机",
    "内层折间稳定性",
]
STRUCTURE_OBJECTIVES = [
    "号码覆盖差_vs随机",
    "组间重号减少_vs随机",
]
PRIMARY_HIT_COLUMNS = [
    "平均命中差_vs随机",
    "单注3+率差_vs随机",
    "整批3+差_vs随机",
]


def _config_seeded_latin_hypercube(model_name: str, count: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Return a model-specific Latin-hypercube sample over declared ranges."""
    count = int(count)
    if count <= 0:
        return []

    def lhs(dimensions: int) -> np.ndarray:
        values = np.empty((count, dimensions), dtype=float)
        for col in range(dimensions):
            values[:, col] = (rng.permutation(count) + rng.random(count)) / count
        return values

    def log_value(u: float, low: float, high: float) -> float:
        return float(np.exp(np.log(low) + float(u) * (np.log(high) - np.log(low))))

    def pick(values: list[int], u: float) -> int:
        return int(values[min(len(values) - 1, int(float(u) * len(values)))])

    if model_name == "近期频率模型":
        x = lhs(2)
        return [
            {"recent_window": pick([30, 50, 80, 100, 150, 200, 300, 500], row[0]),
             "recent_prior": round(log_value(row[1], 2.0, 40.0), 6)}
            for row in x
        ]
    if model_name == "长期频率模型":
        x = lhs(1)
        return [{"long_prior": round(log_value(row[0], 5.0, 300.0), 6)} for row in x]
    if model_name == "指数时间衰减模型":
        x = lhs(1)
        return [{"half_life": round(log_value(row[0], 10.0, 300.0), 6)} for row in x]
    if model_name == "贝叶斯平滑模型":
        x = lhs(1)
        return [{"bayes_prior": round(log_value(row[0], 5.0, 400.0), 6)} for row in x]
    if model_name == "综合评分模型":
        x = lhs(4)
        rows = []
        for row in x:
            a, b = sorted((float(row[0]), float(row[1])))
            weights = [round(a, 8), round(b - a, 8)]
            weights.append(round(1.0 - sum(weights), 8))
            short = pick([30, 50, 80, 100], row[2])
            medium_choices = [v for v in [100, 150, 200, 300, 400] if v > short]
            rows.append({
                "ensemble_weights": weights,
                "ensemble_short_window": short,
                "ensemble_mid_window": pick(medium_choices, row[3]),
            })
        return rows
    if model_name == "历史结构约束模型":
        x = lhs(2)
        rows = []
        for row in x:
            lower = 0.005 + float(row[0]) * (0.12 - 0.005)
            upper_floor = max(0.88, lower + 0.55)
            upper = upper_floor + float(row[1]) * (0.995 - upper_floor)
            rows.append({
                "structure_lower_quantile": round(lower, 6),
                "structure_upper_quantile": round(upper, 6),
            })
        return rows
    if model_name == "多模型集成模型":
        x = lhs(6)
        rows = []
        for row in x:
            a, b = sorted((float(row[0]), float(row[1])))
            weights = [round(a, 8), round(b - a, 8)]
            weights.append(round(1.0 - sum(weights), 8))
            rows.append({
                "multi_model_weights": weights,
                "recent_window": pick([30, 50, 80, 100, 150, 200, 300, 500], row[2]),
                "recent_prior": round(log_value(row[3], 2.0, 40.0), 6),
                "bayes_prior": round(log_value(row[4], 5.0, 400.0), 6),
                "half_life": round(log_value(row[5], 10.0, 300.0), 6),
            })
        return rows
    raise ValueError(f"未支持 V3.8 参数空间设计的模型：{model_name}")


def generate_robust_configs(budget: int = 500, seed: int = 2028) -> list[dict[str, Any]]:
    """Generate balanced, unique candidates with per-model Latin hypercubes."""
    budget = int(budget)
    if budget < len(SEARCH_MODELS) or budget > 2000:
        raise ValueError(f"参数预算必须介于{len(SEARCH_MODELS)}至2000组之间。")
    rng = np.random.default_rng(int(seed))
    base, remainder = divmod(budget, len(SEARCH_MODELS))
    configs: list[dict[str, Any]] = []
    next_id = 1
    for model_idx, model_name in enumerate(SEARCH_MODELS):
        target = base + (model_idx < remainder)
        params_list = _config_seeded_latin_hypercube(model_name, target, rng)
        seen: set[str] = set()
        for params in params_list:
            key = json.dumps(_jsonable(params), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if key in seen:
                # Continuous dimensions are rounded to 6-8 decimals. In the
                # unlikely event of a collision, regenerate from a fresh LHS.
                raise RuntimeError(f"{model_name} 的空间均衡参数出现重复，请更换随机种子。")
            seen.add(key)
            configs.append({
                "config_id": next_id,
                "model_name": model_name,
                "model_version": get_model_version(model_name),
                "params": params,
            })
            next_id += 1
    return configs


def make_spread_inner_folds(
    prefix_length: int,
    min_train: int = 300,
    inner_folds: int = 3,
    inner_validation_draws: int = 6,
) -> list[tuple[int, list[int]]]:
    """Place non-overlapping validation blocks across the available past.

    Every target index is >= min_train, and downstream evaluation constructs
    features from matrix[:target_index] only. Fold windows are spread over the
    training prefix rather than clustered at its very end.
    """
    prefix_length = int(prefix_length)
    min_train = int(min_train)
    inner_folds = int(inner_folds)
    inner_validation_draws = int(inner_validation_draws)
    if min(prefix_length, min_train, inner_folds, inner_validation_draws) < 1:
        raise ValueError("滚动窗口配置必须为正整数。")
    if prefix_length < min_train + inner_folds * inner_validation_draws:
        raise ValueError(
            f"训练前缀至少需要 {min_train + inner_folds * inner_validation_draws} 期，"
            f"当前为{prefix_length}期。"
        )
    earliest = min_train
    latest = prefix_length - inner_validation_draws
    raw_starts = np.linspace(earliest, latest, num=inner_folds)
    starts = [int(round(x)) for x in raw_starts]
    # Rounding must not create overlapping adjacent windows.
    for idx in range(1, len(starts)):
        starts[idx] = max(starts[idx], starts[idx - 1] + inner_validation_draws)
    if starts[-1] + inner_validation_draws > prefix_length:
        overflow = starts[-1] + inner_validation_draws - prefix_length
        starts = [x - overflow for x in starts]
    if starts[0] < min_train or any(starts[i] + inner_validation_draws > starts[i + 1] for i in range(len(starts) - 1)):
        raise ValueError("无法构建互不重叠的分散滚动验证窗口。")
    return [
        (fold_id + 1, list(range(start, start + inner_validation_draws)))
        for fold_id, start in enumerate(starts)
    ]


def _summarize_robust_configs(
    fold_detail: pd.DataFrame,
    configs: list[dict[str, Any]],
) -> pd.DataFrame:
    """Rank prediction and portfolio-structure objectives independently."""
    if fold_detail.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for config in configs:
        cid = int(config["config_id"])
        subset = fold_detail[fold_detail["配置编号"] == cid].sort_values("内层折")
        if subset.empty:
            continue
        weights = subset["验证期数"].to_numpy(dtype=float)
        record: dict[str, Any] = {
            "配置编号": cid,
            "模型": config["model_name"],
            "模型版本": config["model_version"],
            "参数JSON": json.dumps(_jsonable(config["params"]), ensure_ascii=False, sort_keys=True),
            "内层验证窗口数": int(subset["内层折"].nunique()),
            "内层验证期数": int(subset["验证期数"].sum()),
        }
        metric_map = {
            "平均命中数": "模型_mean_hits",
            "随机平均命中数": "随机_mean_hits",
            "单注3+命中率": "模型_single_3plus",
            "随机单注3+率": "随机_single_3plus",
            "单注4+命中率": "模型_single_4plus",
            "随机单注4+率": "随机_single_4plus",
            "整批至少一注3+比例": "模型_batch_3plus",
            "随机整批3+比例": "随机_batch_3plus",
            "平均每批号码覆盖数": "模型_coverage",
            "随机每批号码覆盖数": "随机_coverage",
            "平均组间重号数": "模型_mean_overlap",
            "随机平均组间重号数": "随机_mean_overlap",
        }
        for output, source in metric_map.items():
            record[output] = float(np.average(subset[source], weights=weights))
        for column in PRIMARY_HIT_COLUMNS + STRUCTURE_OBJECTIVES:
            record[column] = float(np.average(subset[column], weights=weights))
        # Add rare 4+ and peak-batch deltas for reporting, not primary ranking.
        record["单注4+率差_vs随机"] = record["单注4+命中率"] - record["随机单注4+率"]
        record["内层折间稳定性"] = -float(subset["平均命中差_vs随机"].std(ddof=0))
        record["领先随机的主要指标数"] = int(sum(record[c] > 0 for c in PRIMARY_HIT_COLUMNS))
        record["是否三个主要点估计均高于随机"] = bool(record["领先随机的主要指标数"] == len(PRIMARY_HIT_COLUMNS))
        record["预测证据状态"] = (
            "三个主要指标点估计均高于随机（仍需前瞻验证）"
            if record["是否三个主要点估计均高于随机"]
            else "未同时高于随机；不能宣称预测优势"
        )
        records.append(record)

    summary = pd.DataFrame(records)
    if summary.empty:
        return summary
    pred_rank_cols: list[str] = []
    for idx, col in enumerate(PREDICTIVE_OBJECTIVES, start=1):
        rank_col = f"预测目标百分位_{idx}"
        summary[rank_col] = summary[col].rank(pct=True, method="average")
        pred_rank_cols.append(rank_col)
    summary["预测综合分"] = summary[pred_rank_cols].mean(axis=1)

    struct_rank_cols: list[str] = []
    for idx, col in enumerate(STRUCTURE_OBJECTIVES, start=1):
        rank_col = f"结构目标百分位_{idx}"
        summary[rank_col] = summary[col].rank(pct=True, method="average")
        struct_rank_cols.append(rank_col)
    summary["结构综合分"] = summary[struct_rank_cols].mean(axis=1)

    summary["预测Pareto前沿"] = _pareto_mask(summary[PREDICTIVE_OBJECTIVES].to_numpy(dtype=float))
    summary["结构Pareto前沿"] = _pareto_mask(summary[STRUCTURE_OBJECTIVES].to_numpy(dtype=float))
    summary = summary.sort_values(
        ["预测综合分", "平均命中差_vs随机", "单注3+率差_vs随机", "配置编号"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    summary.insert(0, "预测排名", np.arange(1, len(summary) + 1))
    return summary


def _select_finalists(
    summary: pd.DataFrame,
    configs: list[dict[str, Any]],
    finalists_per_model: int,
) -> list[dict[str, Any]]:
    selected_ids: set[int] = set()
    for model_name in SEARCH_MODELS:
        subset = summary[summary["模型"] == model_name].sort_values(
            ["预测综合分", "平均命中差_vs随机", "单注3+率差_vs随机", "配置编号"],
            ascending=[False, False, False, True],
        )
        selected_ids.update(int(x) for x in subset.head(finalists_per_model)["配置编号"].tolist())
    chosen = [x for x in configs if int(x["config_id"]) in selected_ids]
    return sorted(chosen, key=lambda x: (SEARCH_MODELS.index(x["model_name"]), x["config_id"]))


def _pareto_frames(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    predictive = summary[summary["预测Pareto前沿"]].copy().reset_index(drop=True)
    structural = summary[summary["结构Pareto前沿"]].copy().reset_index(drop=True)
    return predictive, structural


def _run_screen_and_refinement(
    matrix: np.ndarray,
    draw_numbers: np.ndarray,
    screen_folds: list[tuple[int, list[int]]],
    finalist_folds: list[tuple[int, list[int]]],
    configs: list[dict[str, Any]],
    minimum_train: int,
    tickets_per_draw: int,
    screening_seeds: int,
    finalist_seeds: int,
    finalists_per_model: int,
    seed: int,
    random_cache: dict,
    progress_callback: Callable[[int, int, str], None] | None,
    progress_offset: int,
    progress_total: int,
    label: str,
) -> dict[str, pd.DataFrame | list[dict[str, Any]]]:
    def screen_progress(done: int, _total: int, _label: str) -> None:
        if progress_callback:
            progress_callback(progress_offset + done, progress_total, f"{label}：500组空间均衡初筛")

    screen_period, screen_fold, _ = _evaluate_config_set(
        matrix, draw_numbers, screen_folds, configs, minimum_train, tickets_per_draw,
        screening_seeds, seed, random_cache, f"{label}初筛",
        progress_callback=screen_progress, progress_start=0, progress_total=len(configs),
    )
    screen_summary = _summarize_robust_configs(screen_fold, configs)
    finalists = _select_finalists(screen_summary, configs, finalists_per_model)
    finalist_offset = progress_offset + len(configs)

    def finalist_progress(done: int, _total: int, _label: str) -> None:
        if progress_callback:
            progress_callback(finalist_offset + done, progress_total, f"{label}：优胜候选{finalist_seeds}种子复核")

    finalist_period, finalist_fold, _ = _evaluate_config_set(
        matrix, draw_numbers, finalist_folds, finalists, minimum_train, tickets_per_draw,
        finalist_seeds, seed + 37, random_cache, f"{label}优胜候选复核",
        progress_callback=finalist_progress, progress_start=0, progress_total=len(finalists),
    )
    finalist_summary = _summarize_robust_configs(finalist_fold, finalists)
    predictive_pareto, structure_pareto = _pareto_frames(finalist_summary)
    return {
        "screen_period_detail": screen_period,
        "screen_fold_detail": screen_fold,
        "screen_summary": screen_summary,
        "finalist_period_detail": finalist_period,
        "finalist_fold_detail": finalist_fold,
        "finalist_summary": finalist_summary,
        "predictive_pareto_front": predictive_pareto,
        "structure_pareto_front": structure_pareto,
        "finalists": finalists,
    }


def run_robust_research(
    frame: pd.DataFrame,
    budget: int = 500,
    tickets_per_draw: int = 10,
    minimum_train: int = 300,
    outer_folds: int = 3,
    outer_validation_draws: int = 12,
    inner_folds: int = 3,
    inner_validation_draws: int = 6,
    finalists_per_model: int = 1,
    screening_folds: int = 3,
    screening_window_draws: int = 4,
    screening_seeds: int = 1,
    finalist_seeds: int = 3,
    seed: int = 2028,
    retrospective_cutoff_draw: int = V38_RETROSPECTIVE_CUTOFF_DRAW,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run robust nested rolling evaluation without reusing the exposed V3.7 holdout.

    The historical evaluation is explicitly retrospective, not a fresh holdout.
    A separate prospective lock-and-score process is required for evidence on
    draws that occur after the model and evaluation plan have been frozen.
    """
    started = time.perf_counter()
    clean = validate_frame(frame)
    if clean.empty:
        raise ValueError("没有可用历史数据，请手动导入 CSV 或 Excel。")
    budget, tickets_per_draw = int(budget), int(tickets_per_draw)
    minimum_train = int(minimum_train)
    outer_folds, outer_validation_draws = int(outer_folds), int(outer_validation_draws)
    inner_folds, inner_validation_draws = int(inner_folds), int(inner_validation_draws)
    finalists_per_model, screening_seeds, finalist_seeds = (
        int(finalists_per_model), int(screening_seeds), int(finalist_seeds)
    )
    retrospective_cutoff_draw = int(retrospective_cutoff_draw)
    if budget < len(SEARCH_MODELS) or budget > 2000:
        raise ValueError(f"参数预算必须介于{len(SEARCH_MODELS)}至2000组之间。")
    if not (1 <= tickets_per_draw <= 20):
        raise ValueError("每期每模型模拟注数必须介于1至20注。")
    if not (1 <= outer_folds <= 8 and 1 <= inner_folds <= 8):
        raise ValueError("内外层窗口数必须介于1至8。")
    if finalists_per_model < 1 or finalists_per_model > 5:
        raise ValueError("每个模型的优胜候选复核数必须介于1至5。")
    if screening_seeds < 1 or finalist_seeds < screening_seeds:
        raise ValueError("优胜候选复核随机种子数不能少于初筛种子数。")
    screening_folds, screening_window_draws = int(screening_folds), int(screening_window_draws)
    if not (1 <= screening_folds <= 8 and 3 <= screening_window_draws <= 20):
        raise ValueError("初筛窗口数必须为1至8，每个窗口为3至20期。")
    if outer_validation_draws < 5 or inner_validation_draws < 3:
        raise ValueError("内外层每个验证窗口至少需要5期和3期。")

    safe = clean[clean["draw_no"] <= retrospective_cutoff_draw].copy().reset_index(drop=True)
    if safe.empty:
        raise ValueError(f"历史数据没有覆盖安全回溯区间（截至第{retrospective_cutoff_draw}期）。")
    draw_numbers = safe["draw_no"].to_numpy(dtype=np.int64)
    if np.any(np.diff(draw_numbers) != 1):
        raise ValueError("V3.8 回溯区间存在缺期，无法进行滚动验证。请先手动补齐期号。")
    matrix = safe[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)

    outer_total = outer_folds * outer_validation_draws
    earliest_outer_train = len(safe) - outer_total
    required_inner = minimum_train + max(
        screening_folds * screening_window_draws,
        inner_folds * inner_validation_draws,
    )
    if earliest_outer_train < required_inner:
        needed = outer_total + required_inner
        raise ValueError(
            f"V3.8回溯窗口至少需要{needed}期连续历史数据；截至第{retrospective_cutoff_draw}期目前仅有{len(safe)}期。"
        )

    configs = generate_robust_configs(budget=budget, seed=seed)
    allocations: dict[str, int] = {}
    for item in configs:
        allocations[item["model_name"]] = allocations.get(item["model_name"], 0) + 1
    finalists_max = len(SEARCH_MODELS) * finalists_per_model
    total_units = (outer_folds + 1) * (budget + finalists_max) + outer_folds + 1
    progress_offset = 0
    random_cache: dict = {}

    outer_periods: list[pd.DataFrame] = []
    outer_ticket_tables: list[pd.DataFrame] = []
    outer_selection_rows: list[dict[str, Any]] = []
    outer_start = len(safe) - outer_total

    for outer_idx in range(outer_folds):
        test_start = outer_start + outer_idx * outer_validation_draws
        test_end = test_start + outer_validation_draws
        train_prefix = test_start
        screen_folds = make_spread_inner_folds(
            train_prefix, minimum_train, screening_folds, screening_window_draws
        )
        finalist_folds = make_spread_inner_folds(
            train_prefix, minimum_train, inner_folds, inner_validation_draws
        )
        stage = _run_screen_and_refinement(
            matrix, draw_numbers, screen_folds, finalist_folds, configs, minimum_train, tickets_per_draw,
            screening_seeds, finalist_seeds, finalists_per_model, seed + outer_idx * 1009,
            random_cache, progress_callback, progress_offset, total_units,
            f"外层第{outer_idx + 1}/{outer_folds}折",
        )
        progress_offset += budget + len(stage["finalists"])
        candidate = stage["finalist_summary"].sort_values(
            ["预测综合分", "平均命中差_vs随机", "单注3+率差_vs随机", "配置编号"],
            ascending=[False, False, False, True],
        ).iloc[0]
        config = next(x for x in stage["finalists"] if int(x["config_id"]) == int(candidate["配置编号"]))
        indices = list(range(test_start, test_end))
        outer_period, outer_tickets = _evaluate_frozen_config(
            matrix, draw_numbers, indices, config, minimum_train,
            tickets_per_draw, finalist_seeds, seed + 7001 + outer_idx,
            random_cache, f"V3.8外层第{outer_idx + 1}折前瞻式回溯测试", include_tickets=False,
        )
        outer_periods.append(outer_period)
        if not outer_tickets.empty:
            outer_ticket_tables.append(outer_tickets)
        outer_selection_rows.append({
            "外层折": outer_idx + 1,
            "训练前缀截止期号": int(draw_numbers[test_start - 1]),
            "内层窗口期号范围": [
                [int(draw_numbers[idxs[0]]), int(draw_numbers[idxs[-1]])] for _, idxs in finalist_folds
            ],
            "外层测试开始期号": int(draw_numbers[test_start]),
            "外层测试结束期号": int(draw_numbers[test_end - 1]),
            "选择模型": config["model_name"],
            "配置编号": config["config_id"],
            "参数JSON": json.dumps(_jsonable(config["params"]), ensure_ascii=False, sort_keys=True),
            "预测综合分": float(candidate["预测综合分"]),
            "结构综合分（单独报告）": float(candidate["结构综合分"]),
            "三个主要点估计均高于随机": bool(candidate["是否三个主要点估计均高于随机"]),
            "选择依据": "500组空间均衡初筛→每模型优胜候选3种子复核→仅依赖外层测试前历史选型",
        })
        if progress_callback:
            progress_callback(progress_offset + 1, total_units, f"完成外层第{outer_idx + 1}/{outer_folds}折回溯验证")
        progress_offset += 1

    outer_detail = pd.concat(outer_periods, ignore_index=True) if outer_periods else pd.DataFrame()
    outer_summary = _summarize_periods(outer_detail, seed + 88_001)
    outer_selections = pd.DataFrame(outer_selection_rows)

    # Final candidate selection uses the pre-exposed historical range only.
    final_screen_folds = make_spread_inner_folds(
        len(safe), minimum_train, screening_folds, screening_window_draws
    )
    final_refinement_folds = make_spread_inner_folds(
        len(safe), minimum_train, inner_folds, inner_validation_draws
    )
    final_stage = _run_screen_and_refinement(
        matrix, draw_numbers, final_screen_folds, final_refinement_folds, configs, minimum_train, tickets_per_draw,
        screening_seeds, finalist_seeds, finalists_per_model, seed + 999_983,
        random_cache, progress_callback, progress_offset, total_units,
        "最终开发区间",
    )
    progress_offset += budget + len(final_stage["finalists"])
    selected_row = final_stage["finalist_summary"].sort_values(
        ["预测综合分", "平均命中差_vs随机", "单注3+率差_vs随机", "配置编号"],
        ascending=[False, False, False, True],
    ).iloc[0]
    selected_config = next(
        x for x in final_stage["finalists"] if int(x["config_id"]) == int(selected_row["配置编号"])
    )
    _primary_outer = outer_summary[outer_summary["指标"].isin(PRIMARY_HIT_COLUMNS)] if not outer_summary.empty else pd.DataFrame()
    predictive_signal = bool(
        selected_row["是否三个主要点估计均高于随机"]
        and len(_primary_outer) == len(PRIMARY_HIT_COLUMNS)
        and (_primary_outer["95%CI下限"] > 0).all()
        and (_primary_outer["Holm校正p值"] < 0.05).all()
    )
    evidence_status = (
        "出现需进一步前瞻检验的正向信号"
        if predictive_signal
        else "尚无预测优势证据；当前只保留相对排名靠前的研究候选"
    )

    clean_hash = hashlib.sha256(clean.to_csv(index=False).encode("utf-8")).hexdigest()
    safe_hash = hashlib.sha256(safe.to_csv(index=False).encode("utf-8")).hexdigest()
    metadata = {
        "engine_version": ENGINE_VERSION,
        "experiment_type": "v38_robust_dual_objective_research",
        "experiment_id": f"lotto645-v38-robust-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{seed}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "network_policy": "offline_only_manual_csv_excel_no_official_website_api_or_scraping",
        "budget": budget,
        "unique_configuration_budget": budget,
        "configuration_allocation": allocations,
        "parameter_space_design": "per-model Latin-hypercube sampling; unique parameter configurations",
        "screening_seed_groups": screening_seeds,
        "finalist_seed_groups": finalist_seeds,
        "finalists_per_model": finalists_per_model,
        "tickets_per_draw": tickets_per_draw,
        "minimum_training_draws": minimum_train,
        "outer_folds": outer_folds,
        "outer_validation_draws_per_fold": outer_validation_draws,
        "screening_folds": screening_folds,
        "screening_window_draws": screening_window_draws,
        "inner_folds": inner_folds,
        "inner_validation_draws_per_fold": inner_validation_draws,
        "inner_window_policy": "screening uses 3x4 distributed windows; finalist refinement uses 3x6 distributed windows; each target uses only earlier draws",
        "data_first_draw": int(draw_numbers[0]),
        "retrospective_validation_end_draw": int(draw_numbers[-1]),
        "original_data_last_draw": int(clean["draw_no"].max()),
        "v37_previously_reviewed_draw_range_excluded": [KNOWN_V37_REVIEWED_START_DRAW, KNOWN_V37_REVIEWED_END_DRAW],
        "retrospective_cutoff_draw": retrospective_cutoff_draw,
        "evaluated_rows": int(len(safe)),
        "full_data_rows_available_for_future_prediction": int(len(clean)),
        "full_data_fingerprint_sha256": clean_hash,
        "retrospective_data_fingerprint_sha256": safe_hash,
        "predictive_objectives": PREDICTIVE_OBJECTIVES,
        "structure_objectives": STRUCTURE_OBJECTIVES,
        "predictive_score_definition": "四项：平均命中差、单注3+率差、整批3+差、跨内层窗口稳定性；各自百分位排名等权平均。",
        "structure_score_definition": "两项：号码覆盖差、组间重号减少；单独排名，不参与预测综合分。",
        "selection_rule": "500组空间均衡初筛（3x4 windows, 1 seed）→每模型前1名候选以3 seeds在3x6 windows复核→按预测综合分选择；结构分仅独立展示。",
        "predictive_advantage_status": evidence_status,
        "prospective_holdout_used": False,
        "prospective_validation_required": True,
        "warning": "本实验为回溯稳健性研究，不是新的独立保留测试。V3.7已经审阅的1089-1248期没有被再次用作V3.8回溯标签。未来期必须先锁定配置和号码，再导入开奖结果开展前瞻验证。",
        "selected_configuration": {
            "config_id": int(selected_row["配置编号"]),
            "model_name": selected_row["模型"],
            "model_version": selected_row["模型版本"],
            "params": _jsonable(selected_config["params"]),
            "predictive_score": float(selected_row["预测综合分"]),
            "structure_score_reported_separately": float(selected_row["结构综合分"]),
            "predictive_evidence_status": evidence_status,
        },
        "outer_fold_selections": outer_selection_rows,
        "outer_validation_summary": outer_summary.to_dict(orient="records"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    return {
        "screen_summary": final_stage["screen_summary"],
        "screen_period_detail": final_stage["screen_period_detail"],
        "screen_fold_detail": final_stage["screen_fold_detail"],
        "finalist_summary": final_stage["finalist_summary"],
        "finalist_period_detail": final_stage["finalist_period_detail"],
        "finalist_fold_detail": final_stage["finalist_fold_detail"],
        "predictive_pareto_front": final_stage["predictive_pareto_front"],
        "structure_pareto_front": final_stage["structure_pareto_front"],
        "outer_detail": outer_detail,
        "outer_selections": outer_selections,
        "outer_summary": outer_summary,
        "selected_config": metadata["selected_configuration"],
        "metadata": metadata,
    }


def build_forward_lock(
    frame: pd.DataFrame,
    selected_config: dict[str, Any],
    tickets_per_draw: int = 10,
    seed_groups: int = 3,
    seed: int = 2028,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create an auditable pre-draw lock for the next draw using all known past."""
    clean = validate_frame(frame)
    if len(clean) < 300:
        raise ValueError("生成前瞻锁定记录至少需要300期连续历史数据。")
    draw_numbers = clean["draw_no"].to_numpy(dtype=np.int64)
    if np.any(np.diff(draw_numbers) != 1):
        raise ValueError("手动历史数据存在缺期，不能生成前瞻锁定记录。")
    if not isinstance(selected_config, dict) or "model_name" not in selected_config or "params" not in selected_config:
        raise ValueError("冻结配置结构无效。")
    matrix = clean[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)
    weights, bounds = _weights_for_config(matrix, selected_config, 300)
    target_draw = int(draw_numbers[-1] + 1)
    details: list[dict[str, Any]] = []
    lock_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config_id = int(selected_config.get("config_id", 0))
    params_json = json.dumps(_jsonable(selected_config["params"]), ensure_ascii=False, sort_keys=True)
    for seed_idx in range(int(seed_groups)):
        model_rng = np.random.default_rng(np.random.SeedSequence([int(seed), target_draw, seed_idx + 1, 384_221]))
        random_rng = np.random.default_rng(np.random.SeedSequence([int(seed), target_draw, seed_idx + 1, 777_777]))
        model_tickets = _draw_tickets_fast(model_rng, weights, int(tickets_per_draw), bounds)
        random_tickets = _draw_tickets_fast(random_rng, np.ones(45, dtype=float), int(tickets_per_draw))
        for strategy, tickets in [(selected_config["model_name"], model_tickets), (RANDOM_MODEL, random_tickets)]:
            for ticket_no, ticket in enumerate(tickets, start=1):
                row = {
                    "目标期号": target_draw,
                    "锁定时间UTC": lock_timestamp,
                    "随机种子组": seed_idx + 1,
                    "策略": strategy,
                    "配置编号": config_id if strategy != RANDOM_MODEL else 0,
                    "参数JSON": params_json if strategy != RANDOM_MODEL else "{}",
                    "注序": ticket_no,
                    "号码": "-".join(f"{n:02d}" for n in ticket),
                }
                row.update({f"n{i}": ticket[i - 1] for i in range(1, 7)})
                details.append(row)
    lock_df = pd.DataFrame(details)
    history_hash = hashlib.sha256(clean.to_csv(index=False).encode("utf-8")).hexdigest()
    metadata = {
        "engine_version": ENGINE_VERSION,
        "lock_type": "pre_draw_prediction_lock",
        "created_at_utc": lock_timestamp,
        "target_draw": target_draw,
        "history_first_draw": int(draw_numbers[0]),
        "history_last_draw": int(draw_numbers[-1]),
        "history_fingerprint_sha256": history_hash,
        "model_name": selected_config["model_name"],
        "config_id": config_id,
        "params": _jsonable(selected_config["params"]),
        "tickets_per_strategy_per_seed": int(tickets_per_draw),
        "seed_groups": int(seed_groups),
        "base_seed": int(seed),
        "paired_control": RANDOM_MODEL,
        "policy": "generated_and_locked_before importing the target draw result",
        "warning": "必须在目标期开奖号码公布前保存此文件；否则不可作为前瞻性验证证据。",
    }
    return lock_df, metadata


def evaluate_forward_locks(
    frame: pd.DataFrame,
    locks: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score only previously saved pre-draw locks whose target draw is now present."""
    clean = validate_frame(frame)
    if clean.empty or not isinstance(locks, dict) or not locks:
        return pd.DataFrame(), pd.DataFrame()
    actual_by_draw = {
        int(row["draw_no"]): set(int(row[f"n{i}"]) for i in range(1, 7))
        for _, row in clean.iterrows()
    }
    date_by_draw = {
        int(row["draw_no"]): (None if pd.isna(row.get("date")) else str(row.get("date")))
        for _, row in clean.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for target_key, lock in locks.items():
        if not isinstance(lock, dict):
            continue
        try:
            target_draw = int(target_key)
        except (TypeError, ValueError):
            continue
        if target_draw not in actual_by_draw:
            continue
        metadata = lock.get("metadata")
        detail = lock.get("detail")
        if not isinstance(metadata, dict) or not isinstance(detail, pd.DataFrame) or detail.empty:
            continue
        # A lock cannot be credited as forward-looking if the target draw was
        # already part of the history used to create it.
        history_last = int(metadata.get("history_last_draw", 0))
        if target_draw <= history_last:
            continue
        actual = actual_by_draw[target_draw]
        for _, ticket in detail.iterrows():
            numbers = [int(ticket[f"n{i}"]) for i in range(1, 7)]
            hits = len(set(numbers) & actual)
            rows.append({
                "目标期号": target_draw,
                "开奖日期": date_by_draw.get(target_draw),
                "锁定时间UTC": metadata.get("created_at_utc"),
                "锁定时历史末期": history_last,
                "锁定配置模型": metadata.get("model_name"),
                "策略": ticket.get("策略"),
                "随机种子组": int(ticket.get("随机种子组", 0)),
                "配置编号": int(ticket.get("配置编号", 0)),
                "注序": int(ticket.get("注序", 0)),
                "号码": ticket.get("号码"),
                "命中数": hits,
                "命中3个以上": int(hits >= 3),
                "命中4个以上": int(hits >= 4),
                "开奖日期可用": bool(date_by_draw.get(target_draw)),
                "开奖前保存需人工确认": True,
            })
    scored = pd.DataFrame(rows)
    if scored.empty:
        return scored, pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    for (target_draw, strategy, seed_group), group in scored.groupby(["目标期号", "策略", "随机种子组"], sort=True):
        summary_rows.append({
            "目标期号": int(target_draw),
            "策略": strategy,
            "随机种子组": int(seed_group),
            "注数": int(len(group)),
            "平均命中数": float(group["命中数"].mean()),
            "单注3+率": float(group["命中3个以上"].mean()),
            "单注4+率": float(group["命中4个以上"].mean()),
            "整批至少一注3+": int(group["命中3个以上"].any()),
            "最高命中数": int(group["命中数"].max()),
        })
    return scored, pd.DataFrame(summary_rows)


def restore_forward_locks_from_bundle(bundle_bytes: bytes) -> dict[str, dict[str, Any]]:
    """Restore only validated V3.8 forward-lock artifacts from a research ZIP.

    Other archive members are ignored. This is intentionally an allow-listed
    import path and does not import arbitrary project/session state.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(bundle_bytes), mode="r")
    except (TypeError, zipfile.BadZipFile) as exc:
        raise ValueError("前瞻锁定档案必须是有效的研究 ZIP 文件。") from exc
    restored: dict[str, dict[str, Any]] = {}
    with archive:
        names = set(archive.namelist())
        csv_paths = sorted(name for name in names if name.startswith("forward_locks/draw_") and name.endswith("/locked_tickets.csv"))
        for csv_path in csv_paths:
            folder = csv_path.rsplit("/", 1)[0]
            metadata_path = f"{folder}/lock_metadata.json"
            if metadata_path not in names:
                continue
            try:
                detail = pd.read_csv(io.BytesIO(archive.read(csv_path)))
                metadata = json.loads(archive.read(metadata_path).decode("utf-8"))
                target_draw = int(metadata["target_draw"])
                history_last = int(metadata["history_last_draw"])
            except Exception as exc:
                raise ValueError(f"锁定档案 {folder} 无法解析：{exc}") from exc
            if target_draw <= history_last:
                raise ValueError(f"第{target_draw}期锁定记录的目标期号不晚于锁定时历史末期。")
            required = {"目标期号", "策略", "随机种子组", "配置编号", "注序", "号码", *{f"n{i}" for i in range(1, 7)}}
            if not required.issubset(detail.columns):
                raise ValueError(f"第{target_draw}期锁定 CSV 缺少必要字段。")
            if detail.empty or not (detail["目标期号"].astype(int) == target_draw).all():
                raise ValueError(f"第{target_draw}期锁定 CSV 的目标期号不一致或内容为空。")
            for _, row in detail.iterrows():
                numbers = [int(row[f"n{i}"]) for i in range(1, 7)]
                if len(set(numbers)) != 6 or any(n < 1 or n > 45 for n in numbers):
                    raise ValueError(f"第{target_draw}期锁定 CSV 中存在不合法号码。")
            key = str(target_draw)
            if key in restored:
                raise ValueError(f"研究 ZIP 中重复包含第{target_draw}期锁定记录。")
            restored[key] = {"detail": detail, "metadata": metadata}
    if not restored:
        raise ValueError("ZIP 中没有找到可恢复的 V3.8 前瞻锁定记录。")
    return restored
