# -*- coding: utf-8 -*-
"""Offline parameter search with a reserved, separately evaluated holdout.

Search scores are exploratory. The function deliberately does not calculate
p-values for the winning grid entry because choosing the best entry creates
selection bias. A fixed-parameter holdout evaluation is a separate operation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import hashlib
import time
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from backtest_engine import _draw_tickets, _make_weights, _structure_bounds
from lotto_core import NUMBER_COLS, validate_frame
from model_registry import ENGINE_VERSION, MODEL_NAMES

SEARCHABLE_MODELS = [
    "近期频率模型", "贝叶斯平滑模型", "指数时间衰减模型",
    "综合评分模型", "多模型集成模型",
]


def default_parameter_grid(model_name: str) -> list[dict[str, Any]]:
    """Return a compact, explicit grid for the selected registered model."""
    if model_name == "近期频率模型":
        return [{"recent_window": x} for x in (30, 50, 80, 100, 150, 200, 300)]
    if model_name == "贝叶斯平滑模型":
        return [{"bayes_prior": x} for x in (20, 40, 60, 80, 120, 160, 240)]
    if model_name == "指数时间衰减模型":
        return [{"half_life": x} for x in (20, 30, 40, 60, 80, 120, 160)]
    if model_name == "综合评分模型":
        weights = (
            (0.20, 0.30, 0.50), (0.30, 0.30, 0.40), (0.35, 0.35, 0.30),
            (0.40, 0.35, 0.25), (0.50, 0.30, 0.20), (0.60, 0.25, 0.15),
            (0.25, 0.50, 0.25),
        )
        return [{"ensemble_weights": list(x)} for x in weights]
    if model_name == "多模型集成模型":
        weights = (
            (0.20, 0.50, 0.30), (0.30, 0.40, 0.30), (0.35, 0.35, 0.30),
            (0.40, 0.35, 0.25), (0.50, 0.30, 0.20), (0.25, 0.50, 0.25),
            (0.25, 0.35, 0.40),
        )
        return [{"multi_model_weights": list(x)} for x in weights]
    raise ValueError(f"该模型暂不支持参数搜索：{model_name}")


def _resolve_config(config: dict[str, Any], min_train: int) -> dict[str, Any]:
    out = {
        "recent_window": min_train,
        "half_life": 40.0,
        "bayes_prior": 80.0,
        "ensemble_weights": (0.40, 0.35, 0.25),
        "multi_model_weights": (0.40, 0.35, 0.25),
    }
    out.update(config)
    for key in ("ensemble_weights", "multi_model_weights"):
        out[key] = tuple(float(x) for x in out[key])
        if len(out[key]) != 3 or any(x < 0 for x in out[key]) or not np.isclose(sum(out[key]), 1.0, atol=1e-8):
            raise ValueError(f"{key} 必须由三个非负权重组成，且总和为1。")
    out["recent_window"] = int(out["recent_window"])
    out["half_life"] = float(out["half_life"])
    out["bayes_prior"] = float(out["bayes_prior"])
    if out["recent_window"] < 1 or out["half_life"] <= 0 or out["bayes_prior"] < 0:
        raise ValueError("参数必须为有效的正数；贝叶斯先验强度可以为0。")
    return out


def _group_metrics(tickets: list[tuple[int, ...]], actual: set[int]) -> dict[str, float]:
    hits = [len(set(ticket) & actual) for ticket in tickets]
    all_numbers = {n for ticket in tickets for n in ticket}
    overlaps = [
        len(set(tickets[i]) & set(tickets[j]))
        for i in range(len(tickets)) for j in range(i + 1, len(tickets))
    ]
    return {
        "平均命中数": float(np.mean(hits)),
        "3+命中率": float(np.mean(np.asarray(hits) >= 3)),
        "4+命中率": float(np.mean(np.asarray(hits) >= 4)),
        "整批至少一注3+": float(any(x >= 3 for x in hits)),
        "整批最高命中数": float(max(hits, default=0)),
        "号码覆盖数": float(len(all_numbers)),
        "平均组间重号数": float(np.mean(overlaps)) if overlaps else 0.0,
    }


def run_parameter_search(
    frame: pd.DataFrame,
    model_name: str,
    min_train: int = 200,
    validation_draws: int = 40,
    holdout_draws: int = 40,
    tickets_per_draw: int = 5,
    n_seeds: int = 2,
    seed: int = 2026,
    parameter_grid: Iterable[dict[str, Any]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Search a model's parameter grid without looking at reserved tail rows.

    The latest `holdout_draws` rows are removed before any parameter comparison.
    Every validation target uses only rows before that target. Returned ranking
    is exploratory and must not be described as an independent significance test.
    """
    started = time.perf_counter()
    clean = validate_frame(frame)
    if model_name not in SEARCHABLE_MODELS:
        raise ValueError("请选择支持参数搜索的模型。")
    min_train, validation_draws, holdout_draws = int(min_train), int(validation_draws), int(holdout_draws)
    tickets_per_draw, n_seeds = int(tickets_per_draw), int(n_seeds)
    if min(min_train, validation_draws, holdout_draws, tickets_per_draw, n_seeds) < 1:
        raise ValueError("训练期数、验证期数、保留期数、每期注数和随机种子组数必须大于0。")
    if tickets_per_draw > 20 or n_seeds > 10:
        raise ValueError("每期最多20注，随机种子组最多10组。")
    if len(clean) <= min_train + validation_draws + holdout_draws:
        raise ValueError(
            f"历史数据不足。至少需要 {min_train + validation_draws + holdout_draws + 1} 期；当前 {len(clean)} 期。"
        )
    draw_numbers = clean["draw_no"].to_numpy(dtype=np.int64)
    data_fingerprint = hashlib.sha256(clean.to_csv(index=False).encode("utf-8")).hexdigest()
    if np.any(np.diff(draw_numbers) != 1):
        raise ValueError("历史数据在文件覆盖范围内存在缺期，请先手动补齐后再进行参数搜索。")

    grid = list(parameter_grid or default_parameter_grid(model_name))
    if not grid:
        raise ValueError("参数网格不能为空。")
    configs = [_resolve_config(dict(item), min_train) for item in grid]
    # Deduplicate normalized configurations, preserving order.
    unique_configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for config in configs:
        key = json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in config.items()}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_configs.append(config)
    configs = unique_configs

    development = clean.iloc[:-holdout_draws].reset_index(drop=True)
    matrix = development[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)
    dev_draws = development["draw_no"].to_numpy(dtype=np.int64, copy=True)
    validation_start = max(min_train, len(development) - validation_draws)
    target_indices = list(range(validation_start, len(development)))
    if len(target_indices) < 2:
        raise ValueError("可用于参数比较的验证期数不足，至少需要2期。")
    model_index = MODEL_NAMES.index(model_name) + 1
    random_index = MODEL_NAMES.index("均匀随机基准") + 1
    records: list[dict[str, Any]] = []
    total = len(configs) * len(target_indices) * n_seeds
    done = 0

    for config_idx, config in enumerate(configs, start=1):
        for target_idx in target_indices:
            all_past = matrix[:target_idx]
            actual = set(int(x) for x in matrix[target_idx])
            target_draw = int(dev_draws[target_idx])
            bounds = _structure_bounds(all_past)
            strategy_weights = _make_weights(
                all_past, model_name, config["recent_window"], config["half_life"],
                config["bayes_prior"], config["ensemble_weights"], config["multi_model_weights"],
            )
            for seed_idx in range(n_seeds):
                stream = np.random.SeedSequence([int(seed), target_draw, seed_idx + 1, model_index])
                rng = np.random.default_rng(stream)
                random_stream = np.random.SeedSequence([int(seed), target_draw, seed_idx + 1, random_index])
                random_rng = np.random.default_rng(random_stream)
                tickets = _draw_tickets(rng, strategy_weights, tickets_per_draw)
                random_tickets = _draw_tickets(random_rng, np.ones(45), tickets_per_draw)
                model_metrics = _group_metrics(tickets, actual)
                random_metrics = _group_metrics(random_tickets, actual)
                config_payload = {k: list(v) if isinstance(v, tuple) else v for k, v in config.items()}
                records.append({
                    "模型": model_name,
                    "配置编号": config_idx,
                    "参数JSON": json.dumps(config_payload, ensure_ascii=False, sort_keys=True),
                    "期数": target_draw,
                    "随机种子组": seed_idx + 1,
                    **{f"模型_{k}": v for k, v in model_metrics.items()},
                    **{f"随机_{k}": v for k, v in random_metrics.items()},
                    "平均命中差": model_metrics["平均命中数"] - random_metrics["平均命中数"],
                    "3+率差": model_metrics["3+命中率"] - random_metrics["3+命中率"],
                    "整批3+差": model_metrics["整批至少一注3+"] - random_metrics["整批至少一注3+"],
                })
                done += 1
                if progress_callback is not None:
                    progress_callback(done, total)

    detail = pd.DataFrame(records)
    grouped = detail.groupby(["配置编号", "参数JSON"], sort=True)
    summary_rows: list[dict[str, Any]] = []
    for (config_id, config_json), group in grouped:
        per_draw = group.groupby("期数", as_index=False).mean(numeric_only=True)
        # All draws receive equal weight after averaging seed groups within draw.
        summary_rows.append({
            "配置编号": int(config_id),
            "参数JSON": config_json,
            "验证期数": int(group["期数"].nunique()),
            "验证批次数": int(len(group)),
            "平均命中数": float(group["模型_平均命中数"].mean()),
            "3+命中率": float(group["模型_3+命中率"].mean()),
            "4+命中率": float(group["模型_4+命中率"].mean()),
            "整批至少一注3+比例": float(group["模型_整批至少一注3+"].mean()),
            "平均每批最高命中数": float(group["模型_整批最高命中数"].mean()),
            "平均号码覆盖数": float(group["模型_号码覆盖数"].mean()),
            "平均组间重号数": float(group["模型_平均组间重号数"].mean()),
            "相对随机平均命中差": float(per_draw["平均命中差"].mean()),
            "相对随机3+率差": float(per_draw["3+率差"].mean()),
            "相对随机整批3+差": float(per_draw["整批3+差"].mean()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["平均命中数", "3+命中率", "整批至少一注3+比例"], ascending=[False, False, False]
    ).reset_index(drop=True)
    summary.insert(0, "探索排名", np.arange(1, len(summary) + 1))
    best = json.loads(summary.iloc[0]["参数JSON"])
    holdout_start_idx = len(clean) - holdout_draws
    metadata = {
        "engine_version": ENGINE_VERSION,
        "experiment_type": "parameter_search",
        "model_name": model_name,
        "experiment_id": f"param-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{int(seed)}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_first_draw": int(draw_numbers[0]),
        "data_last_draw": int(draw_numbers[-1]),
        "data_rows": int(len(clean)),
        "data_fingerprint": data_fingerprint,
        "development_last_draw": int(dev_draws[-1]),
        "validation_start_draw": int(dev_draws[validation_start]),
        "validation_end_draw": int(dev_draws[-1]),
        "reserved_holdout_start_draw": int(draw_numbers[holdout_start_idx]),
        "reserved_holdout_end_draw": int(draw_numbers[-1]),
        "reserved_holdout_draws": holdout_draws,
        "validation_draws": len(target_indices),
        "minimum_training_draws": min_train,
        "tickets_per_draw_per_seed": tickets_per_draw,
        "seed_groups": n_seeds,
        "base_seed": int(seed),
        "grid_size": len(configs),
        "best_config": best,
        "holdout_evaluated": False,
        "ranking_warning": "参数排名是探索性选择；对同一验证区间搜索多个参数存在选择偏差。不可将排名第一视为统计显著优势。",
        "holdout_policy": "最新保留区间未参与参数搜索；需要单独调用固定参数验证，并且查看后不得再用于反复调参。",
        "network_policy": "离线；仅使用用户手动提供的历史数据。",
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    detail.attrs["metadata"] = metadata
    summary.attrs["metadata"] = metadata
    return detail, summary, metadata


def evaluate_reserved_holdout(
    frame: pd.DataFrame,
    model_name: str,
    best_config: dict[str, Any],
    min_train: int,
    holdout_draws: int,
    tickets_per_draw: int = 10,
    n_seeds: int = 3,
    seed: int = 2026,
    progress_callback: Callable[[int, int], None] | None = None,
):
    """Evaluate one frozen config on the reserved tail; this consumes that holdout."""
    from backtest_engine import run_backtest

    clean = validate_frame(frame)
    holdout_draws = int(holdout_draws)
    min_train = int(min_train)
    if holdout_draws < 1 or len(clean) <= min_train + holdout_draws:
        raise ValueError("保留区间期数无效，或历史数据不足以同时满足训练窗口和保留测试期。")
    config = _resolve_config(best_config, min_train)
    data_fingerprint = hashlib.sha256(clean.to_csv(index=False).encode("utf-8")).hexdigest()
    holdout_start_draw = int(clean.iloc[-int(holdout_draws)]["draw_no"])
    holdout_end_draw = int(clean.iloc[-1]["draw_no"])
    raw, summary, metadata = run_backtest(
        clean,
        lookback=min_train,
        test_draws=holdout_draws,
        tickets_per_draw=int(tickets_per_draw),
        n_seeds=int(n_seeds),
        seed=int(seed),
        recent_window=config["recent_window"],
        half_life=config["half_life"],
        bayes_prior=config["bayes_prior"],
        ensemble_weights=config["ensemble_weights"],
        multi_model_weights=config["multi_model_weights"],
        strategies=[model_name, "均匀随机基准"],
        progress_callback=progress_callback,
    )
    metadata.update({
        "experiment_type": "frozen_parameter_holdout",
        "evaluated_model": model_name,
        "frozen_config": {k: list(v) if isinstance(v, tuple) else v for k, v in config.items()},
        "holdout_evaluated": True,
        "holdout_consumed": True,
        "data_fingerprint": data_fingerprint,
        "reserved_holdout_start_draw": holdout_start_draw,
        "reserved_holdout_end_draw": holdout_end_draw,
        "warning": "该保留区间已被查看并用于验证，不应再作为完全未触碰的最终测试集反复调参。",
    })
    return raw, summary, metadata
