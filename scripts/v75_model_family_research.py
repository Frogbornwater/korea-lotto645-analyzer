# -*- coding: utf-8 -*-
"""V7.5 predeclared comparison of alternative Lotto 6/45 model families.

Research-only. Reads only manually maintained data/lotto645.csv.
All predictions are one-step-ahead from historical data available before target.
The 100-draw block selector only sees fully completed prior blocks.
No production model, Streamlit UI, historical CSV, or forward-lock archive is changed.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path
from math import comb

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lotto_core import NUMBER_COLS, validate_frame
from v50_backtest import _v4_long_frequency_probabilities
from v50_predictive_engine import (
    build_number_features,
    _fit_from_feature_cache,
    generate_tickets,
)
from scripts.v74_recent1000_research import (
    _draw_metrics,
    moving_block_ci,
    uniform_random_expectations,
)

TARGET_DRAWS = 1000
BLOCK_SIZE = 100
MIN_HISTORY = 180
MIN_TRAINING_DRAWS = 60
TICKETS_PER_DRAW = 10
MASTER_SEED = 75026
BOOTSTRAP_REPS = 20000
BLOCK_BOOTSTRAP_LENGTH = 10
BASE_RATE = 6.0 / 45.0
V4_PRIOR = 174.941214
V5_BASELINE_KEY = "v5_w500_a1_p0.5"
V4_BASELINE_KEY = "v4_long_prior174.941214"
UNIFORM_KEY = "uniform"
PRIMARY_METRICS = (
    "top6_hits", "top10_hits", "top15_hits", "Brier", "LogLoss",
    "mean_hits", "ticket_3plus", "ticket_4plus", "batch_3plus", "batch_4plus",
)


def build_configurations() -> tuple[dict, ...]:
    configs: list[dict] = [{"family": "uniform", "key": UNIFORM_KEY}]
    configs.append({
        "family": "v4_long_frequency",
        "key": V4_BASELINE_KEY,
        "prior": V4_PRIOR,
    })
    for window, prior in itertools.product((100, 300, 500, 800, 1000), (80.0, 175.0)):
        configs.append({
            "family": "rolling_frequency",
            "window": int(window),
            "prior": float(prior),
            "key": f"rolling_w{window}_p{prior:g}",
        })
    for half_life, prior in itertools.product((30, 60, 120, 240, 480), (80.0, 175.0)):
        configs.append({
            "family": "exponential_frequency",
            "half_life": int(half_life),
            "prior": float(prior),
            "key": f"ewma_h{half_life}_p{prior:g}",
        })
    for window, pair_strength in itertools.product((300, 500, 800), (0.0, 0.5, 1.0)):
        configs.append({
            "family": "v5_ridge",
            "training_window": int(window),
            "ridge_alpha": 1.0,
            "pair_strength": float(pair_strength),
            "key": f"v5_w{window}_a1_p{pair_strength:g}",
        })
    return tuple(configs)


CONFIGS = build_configurations()
CONFIG_BY_KEY = {x["key"]: x for x in CONFIGS}
UNIFORM_BASELINES = uniform_random_expectations()


def frequency_probabilities(
    matrix: np.ndarray,
    target_idx: int,
    *,
    window: int | None = None,
    half_life: int | None = None,
    prior: float = 80.0,
) -> np.ndarray:
    """Return a 45-number inclusion vector whose components sum exactly to six."""
    if (window is None) == (half_life is None):
        raise ValueError("必须且只能指定 window 或 half_life。")
    if prior <= 0:
        raise ValueError("prior必须大于0。")
    past = np.asarray(matrix[:int(target_idx)], dtype=np.int16)
    if len(past) < 1:
        raise ValueError("至少需要一期历史开奖。")
    flat = past.reshape(-1)
    if window is not None:
        if int(window) < 1:
            raise ValueError("window必须为正整数。")
        segment = past[-int(window):]
        counts = np.bincount(segment.reshape(-1), minlength=46)[1:46].astype(float)
        denominator = float(len(segment))
        probs = (counts + float(prior) * BASE_RATE) / (denominator + float(prior))
    else:
        if int(half_life) < 1:
            raise ValueError("half_life必须为正整数。")
        weights = np.exp2(-np.arange(len(past) - 1, -1, -1, dtype=float) / float(half_life))
        weighted_counts = np.bincount(
            flat,
            weights=np.repeat(weights, 6),
            minlength=46,
        )[1:46].astype(float)
        denominator = float(weights.sum())
        probs = (weighted_counts + float(prior) * BASE_RATE) / (denominator + float(prior))
    if probs.shape != (45,) or np.any(~np.isfinite(probs)) or np.any(probs <= 0):
        raise RuntimeError("频率模型生成了无效概率向量。")
    if not np.isclose(float(probs.sum()), 6.0, atol=1e-9):
        raise RuntimeError("频率模型的45维入选概率之和不等于6。")
    return probs


def rank_configurations(prior_rows: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Rank candidate models on completed target draws only."""
    if prior_rows.empty:
        return dict(CONFIG_BY_KEY[V5_BASELINE_KEY]), []
    metrics = ("top10_hits", "Brier", "LogLoss", "mean_hits")
    records = []
    for key, group in prior_rows.groupby("config_key", sort=True):
        means = group[list(metrics)].mean()
        records.append({"config_key": key, **{m: float(means[m]) for m in metrics}})
    table = pd.DataFrame(records)
    table["rank_top10"] = table["top10_hits"].rank(ascending=False, method="average")
    table["rank_Brier"] = table["Brier"].rank(ascending=True, method="average")
    table["rank_LogLoss"] = table["LogLoss"].rank(ascending=True, method="average")
    table["rank_mean_hits"] = table["mean_hits"].rank(ascending=False, method="average")
    table["mean_rank"] = table[
        ["rank_top10", "rank_Brier", "rank_LogLoss", "rank_mean_hits"]
    ].mean(axis=1)
    winner = table.sort_values(["mean_rank", "config_key"], kind="mergesort").iloc[0]["config_key"]
    return dict(CONFIG_BY_KEY[winner]), table.sort_values(
        ["mean_rank", "config_key"]
    ).to_dict(orient="records")


def _metric_row(
    probabilities: np.ndarray,
    tickets: list[list[int]],
    actual: set[int],
    random_tickets: list[list[int]],
) -> dict[str, float]:
    row = {
        "V5_45号码概率_JSON": json.dumps([float(x) for x in probabilities], separators=(",", ":")),
        "实际主号码": json.dumps(sorted(actual)),
        "V5_号码列表": "|".join(str(x) for x in tickets),
        "随机_号码列表": "|".join(str(x) for x in random_tickets),
    }
    return _draw_metrics(row)


def _predict_config(
    config: dict,
    matrix: np.ndarray,
    feature_cache: dict,
    target_idx: int,
    target_seed: int,
    random_tickets: list[list[int]],
    actual: set[int],
) -> dict:
    family = config["family"]
    pair_scores = None
    training_draws = int(target_idx)
    effective_sample_draws = float(target_idx)

    if family == "uniform":
        probabilities = np.full(45, BASE_RATE, dtype=float)
        training_draws = 0
        effective_sample_draws = 0.0
        pair_strength = 0.0
        tickets = generate_tickets(
            probabilities, None, TICKETS_PER_DRAW, target_seed, pair_strength=0.0
        )
    elif family == "v4_long_frequency":
        probabilities = _v4_long_frequency_probabilities(matrix[:target_idx], V4_PRIOR)
        pair_strength = 0.0
        tickets = generate_tickets(
            probabilities, None, TICKETS_PER_DRAW, target_seed, pair_strength=0.0
        )
    elif family == "rolling_frequency":
        probabilities = frequency_probabilities(
            matrix, target_idx, window=config["window"], prior=config["prior"]
        )
        training_draws = min(int(config["window"]), int(target_idx))
        effective_sample_draws = float(training_draws)
        pair_strength = 0.0
        tickets = generate_tickets(
            probabilities, None, TICKETS_PER_DRAW, target_seed, pair_strength=0.0
        )
    elif family == "exponential_frequency":
        probabilities = frequency_probabilities(
            matrix, target_idx, half_life=config["half_life"], prior=config["prior"]
        )
        weights = np.exp2(
            -np.arange(int(target_idx) - 1, -1, -1, dtype=float)
            / float(config["half_life"])
        )
        effective_sample_draws = float(weights.sum() ** 2 / np.square(weights).sum())
        pair_strength = 0.0
        tickets = generate_tickets(
            probabilities, None, TICKETS_PER_DRAW, target_seed, pair_strength=0.0
        )
    elif family == "v5_ridge":
        probabilities, pair_scores, diagnostics = _fit_from_feature_cache(
            matrix,
            feature_cache,
            target_idx,
            config["training_window"],
            config["ridge_alpha"],
        )
        training_draws = int(diagnostics["training_draws"])
        effective_sample_draws = float(training_draws)
        if training_draws < MIN_TRAINING_DRAWS:
            raise RuntimeError(
                f"V5训练样本不足：{training_draws} < {MIN_TRAINING_DRAWS}。"
            )
        pair_strength = float(config["pair_strength"])
        tickets = generate_tickets(
            probabilities, pair_scores, TICKETS_PER_DRAW, target_seed,
            pair_strength=pair_strength,
        )
    else:
        raise ValueError(f"未知模型族：{family}")

    metrics = _metric_row(probabilities, tickets, actual, random_tickets)
    return {
        "training_draws": training_draws,
        "effective_sample_draws": effective_sample_draws,
        "probability_sum": float(probabilities.sum()),
        "pair_strength": float(pair_strength),
        **metrics,
    }


def _compare_to_uniform(frame: pd.DataFrame, seed_offset: int) -> list[dict]:
    specs = [
        ("top6_hits", "uniform_top6_expected", "higher"),
        ("top10_hits", "uniform_top10_expected", "higher"),
        ("top15_hits", "uniform_top15_expected", "higher"),
        ("Brier", "uniform_Brier_expected", "lower"),
        ("LogLoss", "uniform_LogLoss_expected", "lower"),
        ("mean_hits", "uniform_mean_hits_expected", "higher"),
        ("ticket_3plus", "uniform_ticket_3plus_expected", "higher"),
        ("ticket_4plus", "uniform_ticket_4plus_expected", "higher"),
        ("batch_3plus", "uniform_batch_3plus_expected", "higher"),
        ("batch_4plus", "uniform_batch_4plus_expected", "higher"),
    ]
    results = []
    for idx, (metric, control, direction) in enumerate(specs):
        difference = frame[metric].to_numpy(float) - frame[control].to_numpy(float)
        if direction == "lower":
            difference = -difference
        low, high = moving_block_ci(
            difference,
            MASTER_SEED + seed_offset + idx * 101,
            alpha=0.05 / len(specs),
        )
        results.append({
            "metric": metric,
            "direction": direction,
            "model_mean": float(frame[metric].mean()),
            "uniform_exact_mean": float(frame[control].mean()),
            "improvement_positive_is_better": float(difference.mean()),
            "bonferroni_familywise_95ci_low": low,
            "bonferroni_familywise_95ci_high": high,
            "draws_better": int(np.sum(difference > 0)),
            "draws_worse": int(np.sum(difference < 0)),
            "draws_tied": int(np.sum(difference == 0)),
        })
    return results


def _compare_paired(
    left: pd.DataFrame,
    right: pd.DataFrame,
    label: str,
    seed_offset: int,
) -> list[dict]:
    results = []
    for idx, metric in enumerate(PRIMARY_METRICS):
        difference = left[metric].to_numpy(float) - right[metric].to_numpy(float)
        if metric in ("Brier", "LogLoss"):
            difference = -difference
        low, high = moving_block_ci(
            difference,
            MASTER_SEED + seed_offset + idx * 101,
            alpha=0.05 / len(PRIMARY_METRICS),
        )
        results.append({
            "metric": f"{label}_{metric}",
            "improvement_positive_is_better": float(difference.mean()),
            "bonferroni_familywise_95ci_low": low,
            "bonferroni_familywise_95ci_high": high,
            "draws_better": int(np.sum(difference > 0)),
            "draws_worse": int(np.sum(difference < 0)),
            "draws_tied": int(np.sum(difference == 0)),
        })
    return results


def run_research(frame: pd.DataFrame) -> dict:
    history = validate_frame(frame).sort_values("draw_no").reset_index(drop=True)
    draws = history["draw_no"].to_numpy(dtype=int)
    if len(history) < MIN_HISTORY + TARGET_DRAWS:
        raise ValueError(
            f"最近{TARGET_DRAWS}期研究至少需要{MIN_HISTORY + TARGET_DRAWS}期连续历史；"
            f"当前只有{len(history)}期。"
        )
    if np.any(np.diff(draws) != 1):
        raise ValueError("研究要求手动导入数据在所用范围内期号连续。")

    matrix = history[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)
    first_target_idx = len(history) - TARGET_DRAWS
    feature_cache = {
        idx: build_number_features(matrix[:idx])
        for idx in range(MIN_HISTORY, len(history))
    }
    candidate_rows: list[dict] = []

    for target_idx in range(first_target_idx, len(history)):
        target_draw = int(draws[target_idx])
        actual = set(int(x) for x in matrix[target_idx])
        target_seed = int(
            np.random.SeedSequence([MASTER_SEED, target_draw]).generate_state(1)[0]
        )
        random_tickets = generate_tickets(
            np.full(45, BASE_RATE), None, TICKETS_PER_DRAW,
            target_seed ^ 0x5A17, pair_strength=0.0,
        )
        for config in CONFIGS:
            prediction = _predict_config(
                config, matrix, feature_cache, target_idx,
                target_seed, random_tickets, actual,
            )
            candidate_rows.append({
                "target_draw": target_draw,
                "config_key": config["key"],
                "model_family": config["family"],
                "config_json": json.dumps(config, sort_keys=True),
                **prediction,
            })

    candidates = pd.DataFrame(candidate_rows)
    expected_targets = list(range(int(draws[-TARGET_DRAWS]), int(draws[-1]) + 1))
    complete = candidates.groupby("config_key")["target_draw"].apply(list)
    if not bool(complete.map(lambda values: values == expected_targets).all()):
        raise RuntimeError("候选模型的目标期序列不完整或不一致。")
    if len(CONFIGS) != len(CONFIG_BY_KEY):
        raise RuntimeError("候选配置存在重复key。")

    selected_rows = []
    block_reports = []
    for block_no, start in enumerate(
        range(expected_targets[0], expected_targets[-1] + 1, BLOCK_SIZE), 1
    ):
        end = min(start + BLOCK_SIZE - 1, expected_targets[-1])
        prior = candidates.loc[candidates["target_draw"] < start].copy()
        chosen, ranking = rank_configurations(prior)
        if block_no == 1:
            chosen = dict(CONFIG_BY_KEY[V5_BASELINE_KEY])
            ranking = []
            method = "predeclared_fixed_v5_baseline_no_prior_block"
        else:
            method = "rank_all_31_model_configs_on_completed_prior_draws_only"
        block = candidates.loc[
            (candidates["config_key"] == chosen["key"])
            & candidates["target_draw"].between(start, end)
        ].copy()
        if len(block) != end - start + 1:
            raise RuntimeError("当前区块选中的模型结果不完整。")
        selected_rows.extend(block.to_dict(orient="records"))
        block_reports.append({
            "block": block_no,
            "target_range": [start, end],
            "selected_config": chosen,
            "selection_method": method,
            "selection_prior_draws": int(prior["target_draw"].nunique()),
            "selected_training_draws_min": int(block["training_draws"].min()),
            "selected_training_draws_max": int(block["training_draws"].max()),
            "selected_training_draws_mean": float(block["training_draws"].mean()),
            "selected_effective_sample_draws_min": float(block["effective_sample_draws"].min()),
            "selected_effective_sample_draws_max": float(block["effective_sample_draws"].max()),
            "selected_effective_sample_draws_mean": float(block["effective_sample_draws"].mean()),
            "prior_config_ranking_top10": ranking[:10],
        })

    selected = pd.DataFrame(selected_rows).sort_values("target_draw").reset_index(drop=True)
    fixed_v5 = candidates.loc[
        candidates["config_key"] == V5_BASELINE_KEY
    ].sort_values("target_draw").reset_index(drop=True)
    fixed_v4 = candidates.loc[
        candidates["config_key"] == V4_BASELINE_KEY
    ].sort_values("target_draw").reset_index(drop=True)
    uniform = candidates.loc[
        candidates["config_key"] == UNIFORM_KEY
    ].sort_values("target_draw").reset_index(drop=True)
    if len(selected) != TARGET_DRAWS or selected["target_draw"].tolist() != expected_targets:
        raise RuntimeError("V7.5最终逐期选择结果不是完整的最近1000期。")

    selected_vs_uniform = _compare_to_uniform(selected, 0)
    fixed_v5_vs_uniform = _compare_to_uniform(fixed_v5, 3001)
    fixed_v4_vs_uniform = _compare_to_uniform(fixed_v4, 6001)
    adaptive_vs_v5 = _compare_paired(selected, fixed_v5, "adaptive_vs_fixed_v5", 9001)
    adaptive_vs_v4 = _compare_paired(selected, fixed_v4, "adaptive_vs_fixed_v4", 12001)

    family_summary = candidates.groupby(["model_family", "config_key"], as_index=False).agg(
        top6_hits=("top6_hits", "mean"),
        top10_hits=("top10_hits", "mean"),
        top15_hits=("top15_hits", "mean"),
        Brier=("Brier", "mean"),
        LogLoss=("LogLoss", "mean"),
        mean_hits=("mean_hits", "mean"),
        ticket_3plus=("ticket_3plus", "mean"),
        ticket_4plus=("ticket_4plus", "mean"),
        batch_3plus=("batch_3plus", "mean"),
        batch_4plus=("batch_4plus", "mean"),
        mean_training_draws=("training_draws", "mean"),
        mean_effective_sample_draws=("effective_sample_draws", "mean"),
    )
    family_summary = family_summary.sort_values(
        ["Brier", "LogLoss", "top10_hits"], ascending=[True, True, False]
    ).reset_index(drop=True)

    out = ROOT / "artifacts" / "v75"
    out.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out / "v75_all_candidate_predictions.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(out / "v75_prequential_selected_1000_detail.csv", index=False, encoding="utf-8-sig")
    family_summary.to_csv(out / "v75_model_family_summary.csv", index=False, encoding="utf-8-sig")

    fingerprint = hashlib.sha256(history.to_csv(index=False).encode("utf-8")).hexdigest()
    report = {
        "experiment": "V7.5 alternative model-family expanding prequential research",
        "status": "historical_research_screen_not_independent_or_prospective_evidence",
        "manual_history_range": [int(draws[0]), int(draws[-1])],
        "history_rows": int(len(history)),
        "target_range": [expected_targets[0], expected_targets[-1]],
        "target_draws": TARGET_DRAWS,
        "block_size": BLOCK_SIZE,
        "candidate_configurations": len(CONFIGS),
        "candidate_families": sorted({x["family"] for x in CONFIGS}),
        "candidate_counts_by_family": {
            family: sum(x["family"] == family for x in CONFIGS)
            for family in sorted({x["family"] for x in CONFIGS})
        },
        "rolling_frequency_windows": [100, 300, 500, 800, 1000],
        "rolling_frequency_prior_draws": [80.0, 175.0],
        "exponential_half_lives": [30, 60, 120, 240, 480],
        "exponential_prior_draws": [80.0, 175.0],
        "v5_ridge_training_windows": [300, 500, 800],
        "v5_ridge_alpha": 1.0,
        "v5_pair_strengths": [0.0, 0.5, 1.0],
        "minimum_v5_training_draws": MIN_TRAINING_DRAWS,
        "selected_effective_sample_draws_range": [float(selected["effective_sample_draws"].min()), float(selected["effective_sample_draws"].max())],
        "tickets_per_draw": TICKETS_PER_DRAW,
        "selection": "First block uses preregistered fixed V5; each later 100-draw block selects one of 31 configurations by mean rank over Top10, Brier, LogLoss and mean hits, using only all candidate results from completed earlier blocks.",
        "uncertainty": f"{BOOTSTRAP_REPS} moving-block bootstrap replicates, block length {BLOCK_BOOTSTRAP_LENGTH}; Bonferroni family-wise 95% intervals across 10 metrics per comparison family.",
        "uniform_random_baseline_exact": uniform_random_expectations(),
        "selected_prequential_vs_uniform": selected_vs_uniform,
        "fixed_v5_vs_uniform": fixed_v5_vs_uniform,
        "fixed_v4_vs_uniform": fixed_v4_vs_uniform,
        "selected_prequential_vs_fixed_v5": adaptive_vs_v5,
        "selected_prequential_vs_fixed_v4": adaptive_vs_v4,
        "block_reports": block_reports,
        "candidate_full_window_ranking_diagnostic_only": family_summary.to_dict(orient="records"),
        "data_fingerprint_sha256": fingerprint,
        "limitations": [
            "最近1000期是已开奖历史，与V7.2、V7.3、V7.4研究区间高度重叠；不能视为独立盲测或前瞻证据。",
            "每个目标期预测只用该期之前的历史开奖；每个100期块的模型族和参数选择只用此前已完成区块。",
            "全窗口候选排名使用了全部目标期结果，只作为描述性诊断，不能作为选型依据。",
            "31组候选配置存在研究者自由度；嵌套区块选择及Bonferroni区间可控制一部分偏差，但不能消除所有重复研究偏差。",
            "Top-k号码排序和边际概率分数不等价于六号码头奖概率；任何历史差异都不能保证下一期表现。",
            "不修改生产选号器、Streamlit界面、原始历史CSV或1251期锁定档案。",
            "数据仅来自仓库内用户手动维护的CSV；不访问外部开奖网站或API。",
        ],
    }
    (out / "v75_model_family_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    run_research(pd.read_csv(ROOT / "data" / "lotto645.csv"))


if __name__ == "__main__":
    main()
