# -*- coding: utf-8 -*-
"""Independent, deterministic walk-forward engine for Lotto 6/45 V3.7.

All training features for a target draw are computed only from rows before that
row. This module is deliberately offline and accepts an already supplied frame.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import time
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from lotto_core import NUMBER_COLS, validate_frame
from model_registry import ENGINE_VERSION, MODEL_DEFINITIONS, MODEL_NAMES, get_model_version

PRIOR_RATE = 6.0 / 45.0


def _smoothed_frequency(numbers: np.ndarray, prior_strength: float) -> np.ndarray:
    counts = np.bincount(numbers.reshape(-1), minlength=46)[1:46].astype(float)
    n_draws = len(numbers)
    return (counts + prior_strength * PRIOR_RATE) / max(n_draws + prior_strength, 1.0)


def _make_weights(
    all_past: np.ndarray,
    strategy: str,
    recent_window: int,
    half_life: float,
    bayes_prior: float,
    ensemble_weights: tuple[float, float, float],
    multi_model_weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
    recent_prior: float = 8.0,
    long_prior: float = 40.0,
    ensemble_short_window: int = 50,
    ensemble_mid_window: int = 200,
) -> np.ndarray:
    """Build strategy weights using strictly pre-target historical rows."""
    if len(all_past) == 0:
        return np.ones(45, dtype=float)

    if strategy in ("均匀随机基准", "历史结构约束模型"):
        return np.ones(45, dtype=float)

    if strategy == "近期频率模型":
        recent = all_past[-min(max(1, recent_window), len(all_past)):]
        return np.maximum(_smoothed_frequency(recent, recent_prior), 1e-12)

    if strategy == "长期频率模型":
        return np.maximum(_smoothed_frequency(all_past, long_prior), 1e-12)

    if strategy == "贝叶斯平滑模型":
        return np.maximum(_smoothed_frequency(all_past, bayes_prior), 1e-12)

    if strategy == "指数时间衰减模型":
        hl = max(float(half_life), 1.0)
        ages = np.arange(len(all_past) - 1, -1, -1, dtype=float)
        decay = np.exp(-ages / hl)
        weighted = np.zeros(45, dtype=float)
        for row, weight in zip(all_past, decay):
            weighted[row - 1] += weight
        # A light uniform prior prevents a zero or numerically negligible weight.
        return np.maximum(weighted + 40.0 * PRIOR_RATE, 1e-12)

    if strategy == "综合评分模型":
        short = all_past[-min(max(1, int(ensemble_short_window)), len(all_past)):]
        medium = all_past[-min(max(1, int(ensemble_mid_window)), len(all_past)):]
        long = all_past
        a, b, c = ensemble_weights
        if not math.isclose(a + b + c, 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError("综合评分模型的三个权重之和必须等于1。")
        result = (
            a * _smoothed_frequency(short, 20.0)
            + b * _smoothed_frequency(medium, 40.0)
            + c * _smoothed_frequency(long, 80.0)
        )
        return np.maximum(result, 1e-12)

    if strategy == "多模型集成模型":
        recent = all_past[-min(max(1, recent_window), len(all_past)):]
        recent_weights = _smoothed_frequency(recent, recent_prior)
        bayes_weights = _smoothed_frequency(all_past, bayes_prior)
        hl = max(float(half_life), 1.0)
        ages = np.arange(len(all_past) - 1, -1, -1, dtype=float)
        decay = np.exp(-ages / hl)
        decayed = np.zeros(45, dtype=float)
        for row, weight in zip(all_past, decay):
            decayed[row - 1] += weight
        decayed += 40.0 * PRIOR_RATE
        a, b, c = multi_model_weights
        if not math.isclose(a + b + c, 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError("多模型集成的三个权重之和必须等于1。")
        return np.maximum(a * recent_weights + b * bayes_weights + c * decayed, 1e-12)

    raise ValueError(f"未注册的回测策略：{strategy}")


def _structure_bounds(
    train: np.ndarray, lower_quantile: float = 0.02, upper_quantile: float = 0.98
) -> tuple[float, float]:
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("结构和值分位数必须满足 0 < lower < upper < 1。")
    sums = train.sum(axis=1)
    return tuple(float(x) for x in np.quantile(sums, [lower_quantile, upper_quantile]))


def _passes_structure(ticket: tuple[int, ...], sum_bounds: tuple[float, float]) -> bool:
    total = sum(ticket)
    odd = sum(n % 2 for n in ticket)
    low = sum(n <= 15 for n in ticket)
    middle = sum(16 <= n <= 30 for n in ticket)
    high = sum(n >= 31 for n in ticket)
    return (
        sum_bounds[0] <= total <= sum_bounds[1]
        and 1 <= odd <= 5
        and low >= 1 and middle >= 1 and high >= 1
    )


def _draw_tickets(
    rng: np.random.Generator,
    weights: np.ndarray,
    count: int,
    structure_bounds: tuple[float, float] | None = None,
) -> list[tuple[int, ...]]:
    probs = np.asarray(weights, dtype=float).copy()
    if probs.shape != (45,) or not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("模型权重必须是45个有限且非负的数值。")
    if probs.sum() <= 0:
        raise ValueError("模型权重总和必须大于0。")
    probs /= probs.sum()

    chosen: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = max(20_000, count * 5_000)
    for _ in range(max_attempts):
        ticket = tuple(sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False, p=probs)))
        if ticket in seen:
            continue
        if structure_bounds is not None and not _passes_structure(ticket, structure_bounds):
            continue
        seen.add(ticket)
        chosen.append(ticket)
        if len(chosen) == count:
            return chosen
    raise RuntimeError(
        f"候选生成未能在{max_attempts:,}次尝试内凑齐{count}组；"
        "请检查训练历史是否足够，或降低每期模拟注数。"
    )


def _paired_signflip_p(values: np.ndarray, rng: np.random.Generator, reps: int) -> float:
    """Two-sided paired sign-flip randomization p-value for a mean difference."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    if np.allclose(values, 0.0):
        return 1.0
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(values)), replace=True)
    simulated = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(simulated >= observed)) / (reps + 1))


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down family-wise error adjustment; preserves input order."""
    if not p_values:
        return []
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.empty(len(p), dtype=float)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    result = np.empty(len(p), dtype=float)
    for rank, idx in enumerate(order):
        result[idx] = adjusted_sorted[rank]
    return result.tolist()


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _batch_metrics(group: pd.DataFrame) -> pd.Series:
    tickets = group[[f"n{i}" for i in range(1, 7)]].to_numpy(dtype=np.int16)
    unique_cover = int(np.unique(tickets).size)
    overlaps: list[int] = []
    for i in range(len(tickets)):
        a = set(int(x) for x in tickets[i])
        for j in range(i + 1, len(tickets)):
            overlaps.append(len(a.intersection(int(x) for x in tickets[j])))
    max_hits = int(group["命中数"].max())
    p3plus = bool((group["命中数"] >= 3).any())
    return pd.Series({
        "每批最高命中数": max_hits,
        "每批至少一注3+": int(p3plus),
        "每批号码覆盖数": unique_cover,
        "每批平均两两重号数": float(np.mean(overlaps)) if overlaps else 0.0,
    })


def run_backtest(
    frame: pd.DataFrame,
    lookback: int = 200,
    test_draws: int = 100,
    tickets_per_draw: int = 10,
    n_seeds: int = 3,
    seed: int = 2026,
    recent_window: int | None = None,
    half_life: float = 40.0,
    bayes_prior: float = 80.0,
    ensemble_weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
    multi_model_weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
    recent_prior: float = 8.0,
    long_prior: float = 40.0,
    ensemble_short_window: int = 50,
    ensemble_mid_window: int = 200,
    structure_lower_quantile: float = 0.02,
    structure_upper_quantile: float = 0.98,
    bootstrap_reps: int = 5000,
    signflip_reps: int = 10000,
    data_source_label: str = "用户手动提供的历史数据",
    progress_callback: Callable[[int, int], None] | None = None,
    strategies: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run deterministic walk-forward simulation and return detail, summary, metadata.

    `lookback` is the minimum number of historical draws required before testing
    begins. Long-term and Bayesian strategies use all rows before the target;
    recent frequency uses `recent_window` (default: lookback).
    """
    started = time.perf_counter()
    clean = validate_frame(frame)
    lookback = int(lookback)
    test_draws = int(test_draws)
    tickets_per_draw = int(tickets_per_draw)
    n_seeds = int(n_seeds)
    recent_window = lookback if recent_window is None else int(recent_window)
    bootstrap_reps = int(bootstrap_reps)
    signflip_reps = int(signflip_reps)
    experiment_id = f"lotto645-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{int(seed)}"
    created_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if clean.empty:
        raise ValueError("没有可用于回测的历史记录，请手动导入 CSV 或 Excel。")
    if min(lookback, test_draws, tickets_per_draw, n_seeds, recent_window, bootstrap_reps, signflip_reps) < 1:
        raise ValueError("训练窗口、测试期数、注数、随机种子、近期窗口和统计重采样次数必须为正整数。")
    if tickets_per_draw > 20:
        raise ValueError("单期每个模型最多模拟20注。")
    if n_seeds > 10:
        raise ValueError("随机种子组数最多10组。")
    selected = list(strategies or MODEL_NAMES)
    unknown = [name for name in selected if name not in MODEL_NAMES]
    if unknown:
        raise ValueError("存在未注册策略：" + ", ".join(unknown))
    if "均匀随机基准" not in selected:
        raise ValueError("回测必须包含均匀随机基准。")
    if len(clean) <= lookback:
        raise ValueError(f"至少需要{lookback + 1}期有效历史记录；当前只有{len(clean)}期。")

    number_matrix = clean[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)
    draw_numbers = clean["draw_no"].to_numpy(dtype=np.int32, copy=True)
    draw_gaps = np.diff(draw_numbers)
    if np.any(draw_gaps != 1):
        missing_examples = []
        for left, right in zip(draw_numbers[:-1], draw_numbers[1:]):
            if right - left != 1:
                missing_examples.append(int(left + 1))
            if len(missing_examples) >= 10:
                break
        raise ValueError(
            "所提供历史文件在其覆盖范围内存在缺失期号，无法进行可靠的连续滚动回测。"
            f"缺失期号示例：{missing_examples}。请先手动补齐历史数据。"
        )
    draw_dates = clean["date"].astype(str).to_numpy()
    start_idx = max(lookback, len(clean) - min(test_draws, len(clean) - lookback))
    test_indices = range(start_idx, len(clean))
    work_total = len(test_indices) * n_seeds * len(selected)
    rows: list[dict[str, Any]] = []

    for target_idx in test_indices:
        all_past = number_matrix[:target_idx]
        target_draw = int(draw_numbers[target_idx])
        target_date = str(draw_dates[target_idx])
        actual = set(int(x) for x in number_matrix[target_idx])
        train_start_draw = int(draw_numbers[0])
        train_end_draw = int(draw_numbers[target_idx - 1])
        structure_train = all_past[-min(lookback, len(all_past)):]
        structure_bounds = _structure_bounds(
            structure_train, structure_lower_quantile, structure_upper_quantile
        )

        for seed_idx in range(n_seeds):
            for strategy_idx, strategy in enumerate(selected):
                # Derive an independent but reproducible stream for each period,
                # seed group and strategy. Strategy order changes won't alter others.
                stable_strategy_idx = MODEL_NAMES.index(strategy) + 1
                strategy_seed = np.random.SeedSequence(
                    [int(seed), target_draw, seed_idx + 1, stable_strategy_idx]
                )
                rng = np.random.default_rng(strategy_seed)
                weights = _make_weights(
                    all_past, strategy, recent_window, half_life, bayes_prior, ensemble_weights,
                    multi_model_weights, recent_prior, long_prior, ensemble_short_window, ensemble_mid_window,
                )
                bounds = structure_bounds if strategy == "历史结构约束模型" else None
                tickets = _draw_tickets(rng, weights, tickets_per_draw, bounds)
                model_version = get_model_version(strategy)
                if strategy == "近期频率模型":
                    strategy_start_idx = max(0, target_idx - min(recent_window, target_idx))
                    strategy_start_draw = int(draw_numbers[strategy_start_idx])
                elif strategy == "历史结构约束模型":
                    strategy_start_idx = max(0, target_idx - min(lookback, target_idx))
                    strategy_start_draw = int(draw_numbers[strategy_start_idx])
                elif strategy == "均匀随机基准":
                    strategy_start_draw = None
                else:
                    strategy_start_draw = train_start_draw
                for ticket_idx, ticket in enumerate(tickets, start=1):
                    hits = len(set(ticket).intersection(actual))
                    row: dict[str, Any] = {
                        "实验编号": experiment_id,
                        "实验引擎版本": ENGINE_VERSION,
                        "期数": target_draw,
                        "开奖日期": target_date,
                        "训练开始期号": train_start_draw,
                        "策略实际训练起始期号": strategy_start_draw,
                        "训练截止期号": train_end_draw,
                        "训练可用期数": int(target_idx),
                        "最低训练窗口": lookback,
                        "近期频率窗口": recent_window,
                        "随机种子组": seed_idx + 1,
                        "随机种子": int(seed),
                        "策略": strategy,
                        "模型版本": model_version,
                        "注序": ticket_idx,
                        "号码": "-".join(f"{n:02d}" for n in ticket),
                        "号码JSON": json.dumps(list(ticket), ensure_ascii=False),
                        "命中数": hits,
                        "命中3个以上": int(hits >= 3),
                        "命中4个以上": int(hits >= 4),
                        "命中5个以上": int(hits >= 5),
                        "命中6个": int(hits == 6),
                    }
                    row.update({f"n{i}": ticket[i - 1] for i in range(1, 7)})
                    rows.append(row)
        if progress_callback is not None:
            done = (target_idx - start_idx + 1) * n_seeds * len(selected)
            progress_callback(done, work_total)

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, pd.DataFrame(), {}

    group_keys = ["期数", "随机种子组", "策略"]
    batch_rows: list[dict[str, Any]] = []
    for key, group in raw.groupby(group_keys, sort=False):
        metrics = _batch_metrics(group)
        batch_rows.append({
            "期数": int(key[0]), "随机种子组": int(key[1]), "策略": str(key[2]),
            "每期平均命中": float(group["命中数"].mean()),
            "每期3+率": float(group["命中3个以上"].mean()),
            **metrics.to_dict(),
        })
    batches = pd.DataFrame(batch_rows)

    # Average multiple seeds within each draw first: the draw, not each ticket,
    # is the inferential unit for paired comparisons against the random baseline.
    period = batches.groupby(["期数", "策略"], as_index=False).mean(numeric_only=True)
    baseline = period[period["策略"] == "均匀随机基准"].set_index("期数")
    rng_ci = np.random.default_rng(int(seed) + 731_991)
    rng_test = np.random.default_rng(int(seed) + 941_107)
    summaries: list[dict[str, Any]] = []
    p_value_rows: dict[str, dict[str, float]] = {}
    for strategy in selected:
        subset = raw[raw["策略"] == strategy]
        current_period = period[period["策略"] == strategy].set_index("期数")
        aligned = current_period.join(
            baseline[["每期平均命中", "每期3+率", "每批至少一注3+", "每批最高命中数"]],
            rsuffix="_随机", how="inner"
        )
        hit_delta = (aligned["每期平均命中"] - aligned["每期平均命中_随机"]).to_numpy()
        p3_delta = (aligned["每期3+率"] - aligned["每期3+率_随机"]).to_numpy()
        batch_delta = (aligned["每批至少一注3+"] - aligned["每批至少一注3+_随机"]).to_numpy()
        hit_lo, hit_hi = (0.0, 0.0) if strategy == "均匀随机基准" else _bootstrap_ci(hit_delta, rng_ci, bootstrap_reps)
        p3_lo, p3_hi = (0.0, 0.0) if strategy == "均匀随机基准" else _bootstrap_ci(p3_delta, rng_ci, bootstrap_reps)
        batch_lo, batch_hi = (0.0, 0.0) if strategy == "均匀随机基准" else _bootstrap_ci(batch_delta, rng_ci, bootstrap_reps)
        batch_for_strategy = batches[batches["策略"] == strategy]
        p_value_rows[strategy] = {
            "hit_p": 1.0 if strategy == "均匀随机基准" else _paired_signflip_p(hit_delta, rng_test, signflip_reps),
            "p3_p": 1.0 if strategy == "均匀随机基准" else _paired_signflip_p(p3_delta, rng_test, signflip_reps),
            "batch_p": 1.0 if strategy == "均匀随机基准" else _paired_signflip_p(batch_delta, rng_test, signflip_reps),
        }
        summaries.append({
            "策略": strategy,
            "模型版本": get_model_version(strategy),
            "测试注数": int(len(subset)),
            "测试期数": int(subset["期数"].nunique()),
            "平均命中数": float(subset["命中数"].mean()),
            "3个以上命中率": float(subset["命中3个以上"].mean()),
            "4个以上命中率": float(subset["命中4个以上"].mean()),
            "5个以上命中率": float(subset["命中5个以上"].mean()),
            "6个命中率": float(subset["命中6个"].mean()),
            "每批至少一注3+比例": float(batch_for_strategy["每批至少一注3+"].mean()),
            "平均每批最高命中数": float(batch_for_strategy["每批最高命中数"].mean()),
            "平均每批号码覆盖数": float(batch_for_strategy["每批号码覆盖数"].mean()),
            "平均组间重号数": float(batch_for_strategy["每批平均两两重号数"].mean()),
            "平均命中差_vs随机": 0.0 if strategy == "均匀随机基准" else float(hit_delta.mean()),
            "平均命中差95%CI下限": hit_lo,
            "平均命中差95%CI上限": hit_hi,
            "3+率差_vs随机": 0.0 if strategy == "均匀随机基准" else float(p3_delta.mean()),
            "3+率差95%CI下限": p3_lo,
            "3+率差95%CI上限": p3_hi,
            "整批3+率差_vs随机": 0.0 if strategy == "均匀随机基准" else float(batch_delta.mean()),
            "整批3+率差95%CI下限": batch_lo,
            "整批3+率差95%CI上限": batch_hi,
            "平均命中差_配对检验p值": p_value_rows[strategy]["hit_p"],
            "3+率差_配对检验p值": p_value_rows[strategy]["p3_p"],
            "整批3+率差_配对检验p值": p_value_rows[strategy]["batch_p"],
        })
    summary = pd.DataFrame(summaries)
    non_baseline = [name for name in selected if name != "均匀随机基准"]
    for metric_key, output_col in [
        ("hit_p", "平均命中差_Holm校正p值"),
        ("p3_p", "3+率差_Holm校正p值"),
        ("batch_p", "整批3+率差_Holm校正p值"),
    ]:
        adjusted = _holm_adjust([p_value_rows[name][metric_key] for name in non_baseline])
        adjusted_map = dict(zip(non_baseline, adjusted))
        summary[output_col] = summary["策略"].map(
            lambda name: 1.0 if name == "均匀随机基准" else adjusted_map.get(name, float("nan"))
        )
    summary = summary.copy()

    test_start = int(draw_numbers[start_idx])
    metadata = {
        "engine_version": ENGINE_VERSION,
        "experiment_id": experiment_id,
        "created_at_utc": created_at_utc,
        "data_source_label": str(data_source_label),
        "data_first_draw": int(draw_numbers[0]),
        "data_last_draw": int(draw_numbers[-1]),
        "data_rows": int(len(clean)),
        "test_start_draw": test_start,
        "test_end_draw": int(draw_numbers[-1]),
        "test_draws": int(len(test_indices)),
        "lookback_minimum": lookback,
        "recent_window": recent_window,
        "tickets_per_draw_per_seed": tickets_per_draw,
        "seed_groups": n_seeds,
        "base_seed": int(seed),
        "strategies": selected,
        "model_versions": {name: get_model_version(name) for name in selected},
        "half_life": float(half_life),
        "bayes_prior": float(bayes_prior),
        "ensemble_weights": list(ensemble_weights),
        "multi_model_weights": list(multi_model_weights),
        "recent_prior": float(recent_prior),
        "long_prior": float(long_prior),
        "ensemble_short_window": int(ensemble_short_window),
        "ensemble_mid_window": int(ensemble_mid_window),
        "structure_lower_quantile": float(structure_lower_quantile),
        "structure_upper_quantile": float(structure_upper_quantile),
        "bootstrap_reps": bootstrap_reps,
        "signflip_reps": signflip_reps,
        "multiple_testing_correction": "Holm；对各非随机模型分别按平均命中差、单注3+率差、整批3+率差三项进行家族内校正。",
        "training_rule": "每个测试期只使用其之前的历史行；长期/贝叶斯/衰减模型使用此前全部历史，近期模型使用指定近期窗口。",
        "network_policy": "离线；只分析用户手动导入的文件或仓库内手动维护的历史文件。",
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    raw.attrs["metadata"] = metadata
    summary.attrs["metadata"] = metadata
    return raw, summary, metadata


def run_multi_window_backtest(
    frame: pd.DataFrame,
    windows: Iterable[int] = (100, 200, 300, 500),
    test_draws: int = 100,
    tickets_per_draw: int = 10,
    n_seeds: int = 3,
    seed: int = 2026,
    progress_callback: Callable[[int, int], None] | None = None,
    bootstrap_reps: int = 3000,
    data_source_label: str = "用户手动提供的历史数据",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Compare minimum training windows; returns concatenated ticket-level runs."""
    window_list = list(dict.fromkeys(int(w) for w in windows))
    if not window_list or any(w < 1 for w in window_list):
        raise ValueError("多窗口回测至少需要一个正整数窗口。")
    total = len(window_list)
    all_raw: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    all_metadata: list[dict[str, Any]] = []
    for idx, window in enumerate(window_list):
        def child_progress(done: int, child_total: int, *, _idx: int = idx) -> None:
            if progress_callback is not None:
                scaled = int((done / max(child_total, 1)) * 1_000_000)
                progress_callback(_idx * 1_000_000 + scaled, total * 1_000_000)
        raw, summary, metadata = run_backtest(
            frame,
            lookback=window,
            recent_window=window,
            test_draws=test_draws,
            tickets_per_draw=tickets_per_draw,
            n_seeds=n_seeds,
            seed=seed,
            bootstrap_reps=bootstrap_reps,
            data_source_label=data_source_label,
            progress_callback=child_progress,
        )
        raw = raw.copy()
        summary = summary.copy()
        raw["比较训练窗口"] = window
        summary["比较训练窗口"] = window
        all_raw.append(raw)
        all_summary.append(summary)
        all_metadata.append(metadata)
    combined_raw = pd.concat(all_raw, ignore_index=True) if all_raw else pd.DataFrame()
    combined_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    return combined_raw, combined_summary, all_metadata
