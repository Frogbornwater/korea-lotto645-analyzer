# -*- coding: utf-8 -*-
"""V7.2 recent-500 expanding prequential hyperparameter study.

Research-only. Reads only the manually maintained data/lotto645.csv.
No network access, no official lottery integration, no production changes,
and no edits to forward-lock archives. Every prediction is generated from
strictly earlier draws. Hyperparameters for each 100-draw block are selected
only from completed earlier blocks.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lotto_core import validate_frame
from v50_backtest import run_v50_walk_forward

TARGET_DRAWS = 500
BLOCK_SIZE = 100
MIN_HISTORY = 180
TICKETS_PER_DRAW = 10
MASTER_SEED = 72026
BOOTSTRAP_REPS = 20000
BLOCK_BOOTSTRAP_LENGTH = 10
# Predeclared, deliberately compact grid. Training history is capped at 1,000
# by the current V5 engine; 300/500/800/1000 test shorter vs longer memory.
CONFIGS = tuple(
    {"training_window": w, "ridge_alpha": a, "pair_strength": p}
    for w, a, p in itertools.product(
        (300, 500, 800, 1000), (0.3, 1.0), (0.0, 0.5, 1.0)
    )
)
BASELINE = {"training_window": 500, "ridge_alpha": 1.0, "pair_strength": 0.5}
PRIMARY_METRICS = ("top10_hits", "Brier", "LogLoss", "mean_hits", "ticket_3plus", "batch_3plus")


def _decode_numbers(value: str) -> list[int]:
    return [int(x) for x in json.loads(value)]


def _draw_metrics(row: dict) -> dict[str, float]:
    probs = np.asarray(json.loads(row["V5_45号码概率_JSON"]), dtype=float)
    actual = set(_decode_numbers(row["实际主号码"]))
    if probs.shape != (45,) or not np.all(np.isfinite(probs)):
        raise ValueError("概率向量不是45个有限数值。")
    if np.any(probs < 0) or np.any(probs > 1) or not np.isclose(probs.sum(), 6.0, atol=1e-5):
        raise ValueError("概率向量越界或概率总和不为6。")
    if len(actual) != 6 or min(actual) < 1 or max(actual) > 45:
        raise ValueError("实际主号码无效。")
    y = np.zeros(45, dtype=float)
    y[np.asarray(sorted(actual)) - 1] = 1.0
    clipped = np.clip(probs, 1e-9, 1.0 - 1e-9)
    chosen = np.argsort(-probs, kind="mergesort")
    top10 = len(set((chosen[:10] + 1).tolist()) & actual)
    tickets = [list(map(int, ast.literal_eval(ticket))) for ticket in row["V5_号码列表"].split("|")]
    random_tickets = [list(map(int, ast.literal_eval(ticket))) for ticket in row["随机_号码列表"].split("|")]
    model_hits = np.asarray([len(set(t) & actual) for t in tickets], dtype=float)
    random_hits = np.asarray([len(set(t) & actual) for t in random_tickets], dtype=float)
    return {
        "top6_hits": float(len(set((chosen[:6] + 1).tolist()) & actual)),
        "top10_hits": float(top10),
        "top15_hits": float(len(set((chosen[:15] + 1).tolist()) & actual)),
        "Brier": float(np.mean((probs - y) ** 2)),
        "LogLoss": float(-np.mean(y * np.log(clipped) + (1-y) * np.log(1-clipped))),
        "mean_hits": float(model_hits.mean()),
        "ticket_3plus": float(np.mean(model_hits >= 3)),
        "ticket_4plus": float(np.mean(model_hits >= 4)),
        "batch_3plus": float(np.any(model_hits >= 3)),
        "batch_4plus": float(np.any(model_hits >= 4)),
        "random_mean_hits": float(random_hits.mean()),
        "random_ticket_3plus": float(np.mean(random_hits >= 3)),
        "random_ticket_4plus": float(np.mean(random_hits >= 4)),
        "random_batch_3plus": float(np.any(random_hits >= 3)),
        "random_batch_4plus": float(np.any(random_hits >= 4)),
        "random_Brier": float(6/45 * (1-6/45)),
        "random_LogLoss": float(-(6/45)*np.log(6/45) -(39/45)*np.log(39/45)),
    }


def _rank_configurations(prior: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Rank only using already completed target blocks; lower rank score is better."""
    if prior.empty:
        return dict(BASELINE), []
    metrics = ("top10_hits", "Brier", "LogLoss", "mean_hits")
    scored = []
    for key, group in prior.groupby("config_key", sort=True):
        means = group[list(metrics)].mean()
        # Each metric contributes one rank, avoiding arbitrary unit scaling.
        scored.append({"config_key": key, **{m: float(means[m]) for m in metrics}})
    table = pd.DataFrame(scored)
    table["rank_top10"] = table["top10_hits"].rank(ascending=False, method="average")
    table["rank_Brier"] = table["Brier"].rank(ascending=True, method="average")
    table["rank_LogLoss"] = table["LogLoss"].rank(ascending=True, method="average")
    table["rank_mean_hits"] = table["mean_hits"].rank(ascending=False, method="average")
    table["mean_rank"] = table[["rank_top10", "rank_Brier", "rank_LogLoss", "rank_mean_hits"]].mean(axis=1)
    winner = table.sort_values(["mean_rank", "config_key"], kind="mergesort").iloc[0]["config_key"]
    selected = next(c for c in CONFIGS if config_key(c) == winner)
    return dict(selected), table.sort_values(["mean_rank", "config_key"]).to_dict(orient="records")


def config_key(config: dict) -> str:
    return f"w{int(config['training_window'])}_a{float(config['ridge_alpha']):g}_p{float(config['pair_strength']):g}"


def moving_block_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / BLOCK_BOOTSTRAP_LENGTH))
    starts = rng.integers(0, n, size=(BOOTSTRAP_REPS, blocks))
    offsets = np.arange(BLOCK_BOOTSTRAP_LENGTH)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    samples = x[indices.reshape(BOOTSTRAP_REPS, -1)[:, :n]]
    means = samples.mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


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

    candidate_results = []
    for i, config in enumerate(CONFIGS):
        result = run_v50_walk_forward(
            history,
            test_draws=TARGET_DRAWS,
            training_window=config["training_window"],
            min_history=MIN_HISTORY,
            ridge_alpha=config["ridge_alpha"],
            tickets_per_draw=TICKETS_PER_DRAW,
            seed=MASTER_SEED,
            bootstrap_reps=100,
            pair_sampling_strength=config["pair_strength"],
        )
        detail = result["detail"].copy()
        if len(detail) != TARGET_DRAWS or not (detail["训练结束期"] < detail["目标期号"]).all():
            raise RuntimeError("候选配置未生成完整的严格前瞻式滚动预测。")
        key = config_key(config)
        rows = []
        for row in detail.to_dict(orient="records"):
            rows.append({
                "target_draw": int(row["目标期号"]),
                "config_key": key,
                **_draw_metrics(row),
            })
        candidate_results.extend(rows)

    candidates = pd.DataFrame(candidate_results)
    expected_targets = list(range(int(draws[-TARGET_DRAWS]), int(draws[-1]) + 1))
    if not bool(candidates.groupby("config_key")["target_draw"].apply(list).map(lambda x: x == expected_targets).all()):
        raise RuntimeError("候选配置的目标期序列不完整或不一致。")

    # The first block is a preregistered fixed baseline. For every later block,
    # select only from predictions whose target draws precede that block.
    selected_rows = []
    block_reports = []
    for block_no, start in enumerate(range(expected_targets[0], expected_targets[-1] + 1, BLOCK_SIZE), 1):
        end = min(start + BLOCK_SIZE - 1, expected_targets[-1])
        prior = pd.DataFrame(selected_rows)
        chosen, ranking = _rank_configurations(prior)
        if block_no == 1:
            chosen = dict(BASELINE)
            ranking = []
            selection_method = "predeclared_baseline_no_prior_block"
        else:
            selection_method = "best_mean_rank_on_completed_prior_blocks_only"
        key = config_key(chosen)
        block = candidates.loc[
            (candidates["config_key"] == key) &
            candidates["target_draw"].between(start, end)
        ].copy()
        if len(block) != end - start + 1:
            raise RuntimeError("所选配置缺少当前测试块预测。")
        selected_rows.extend(block.to_dict(orient="records"))
        block_reports.append({
            "block": block_no,
            "target_range": [start, end],
            "selected_config": chosen,
            "selection_method": selection_method,
            "selection_training_draws": int(len(prior)),
            "prior_config_ranking": ranking[:10],
        })

    selected = pd.DataFrame(selected_rows).sort_values("target_draw").reset_index(drop=True)
    baseline_key = config_key(BASELINE)
    baseline = candidates.loc[candidates["config_key"] == baseline_key].sort_values("target_draw").reset_index(drop=True)
    if len(selected) != TARGET_DRAWS or selected["target_draw"].tolist() != expected_targets:
        raise RuntimeError("V7.2最终逐期选择结果不是完整的最近500期。")

    summaries = []
    compare_specs = [
        ("top10_hits", "top10_hits", "higher"),
        ("Brier", "random_Brier", "lower"),
        ("LogLoss", "random_LogLoss", "lower"),
        ("mean_hits", "random_mean_hits", "higher"),
        ("ticket_3plus", "random_ticket_3plus", "higher"),
        ("batch_3plus", "random_batch_3plus", "higher"),
    ]
    for idx, (metric, control, direction) in enumerate(compare_specs):
        diff = selected[metric].to_numpy(float) - selected[control].to_numpy(float)
        if direction == "lower":
            diff = -diff
        low, high = moving_block_ci(diff, MASTER_SEED + idx * 101)
        summaries.append({
            "metric": metric,
            "direction": direction,
            "selected_prequential_mean": float(selected[metric].mean()),
            "control_mean": float(selected[control].mean()),
            "improvement_positive_is_better": float(diff.mean()),
            "moving_block_bootstrap_95ci_low": low,
            "moving_block_bootstrap_95ci_high": high,
            "draws_better": int(np.sum(diff > 0)),
            "draws_worse": int(np.sum(diff < 0)),
            "draws_tied": int(np.sum(diff == 0)),
        })

    # Compare adaptive selection with the fixed V5 baseline on the exact same targets.
    for idx, metric in enumerate(("top10_hits", "Brier", "LogLoss", "mean_hits", "ticket_3plus", "batch_3plus")):
        diff = selected[metric].to_numpy(float) - baseline[metric].to_numpy(float)
        if metric in ("Brier", "LogLoss"):
            diff = -diff
        low, high = moving_block_ci(diff, MASTER_SEED + 2001 + idx * 101)
        summaries.append({
            "metric": f"adaptive_vs_fixed_{metric}",
            "direction": "higher_is_better_after_sign_adjustment",
            "selected_prequential_mean": float(selected[metric].mean()),
            "control_mean": float(baseline[metric].mean()),
            "improvement_positive_is_better": float(diff.mean()),
            "moving_block_bootstrap_95ci_low": low,
            "moving_block_bootstrap_95ci_high": high,
            "draws_better": int(np.sum(diff > 0)),
            "draws_worse": int(np.sum(diff < 0)),
            "draws_tied": int(np.sum(diff == 0)),
        })

    candidate_summary = candidates.groupby("config_key").agg(
        top10_hits=("top10_hits", "mean"),
        Brier=("Brier", "mean"),
        LogLoss=("LogLoss", "mean"),
        mean_hits=("mean_hits", "mean"),
        ticket_3plus=("ticket_3plus", "mean"),
        batch_3plus=("batch_3plus", "mean"),
    ).reset_index()
    candidate_summary["training_window"] = candidate_summary["config_key"].str.extract(r"w(\d+)").astype(int)
    candidate_summary["ridge_alpha"] = candidate_summary["config_key"].str.extract(r"_a([0-9.]+)").astype(float)
    candidate_summary["pair_strength"] = candidate_summary["config_key"].str.extract(r"_p([0-9.]+)").astype(float)
    candidate_summary = candidate_summary.sort_values(
        ["Brier", "LogLoss", "top10_hits"], ascending=[True, True, False]
    ).reset_index(drop=True)

    out = ROOT / "artifacts" / "v72"
    out.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out / "v72_all_candidate_predictions.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(out / "v72_prequential_selected_500_detail.csv", index=False, encoding="utf-8-sig")
    candidate_summary.to_csv(out / "v72_candidate_summary.csv", index=False, encoding="utf-8-sig")
    fingerprint = hashlib.sha256(history.to_csv(index=False).encode("utf-8")).hexdigest()
    report = {
        "experiment": "V7.2 recent-500 expanding prequential hyperparameter study",
        "status": "historical_research_screen_not_independent_or_prospective_evidence",
        "manual_history_range": [int(draws[0]), int(draws[-1])],
        "history_rows": int(len(history)),
        "target_range": [expected_targets[0], expected_targets[-1]],
        "target_draws": TARGET_DRAWS,
        "block_size": BLOCK_SIZE,
        "training_window_candidates": [300, 500, 800, 1000],
        "ridge_alpha_candidates": [0.3, 1.0],
        "pair_sampling_strength_candidates": [0.0, 0.5, 1.0],
        "candidate_configurations": len(CONFIGS),
        "tickets_per_draw": TICKETS_PER_DRAW,
        "selection": "Each 100-draw block selects from completed earlier blocks only using mean rank across Top10, Brier, LogLoss, and mean ticket hits; first block uses the preregistered V5 baseline.",
        "primary_metrics": list(PRIMARY_METRICS),
        "uncertainty": f"{BOOTSTRAP_REPS} moving-block bootstrap replicates, block length {BLOCK_BOOTSTRAP_LENGTH}",
        "block_reports": block_reports,
        "selected_prequential_vs_random": summaries[:6],
        "selected_prequential_vs_fixed_baseline": summaries[6:],
        "candidate_full_window_ranking": candidate_summary.to_dict(orient="records"),
        "data_fingerprint_sha256": fingerprint,
        "limitations": [
            "最近500期是已开奖历史，且部分目标期已参与过此前研究；本报告属于历史研究筛查，不是独立盲测或前瞻证据。",
            "每个目标期的预测均只使用此前开奖；每个100期块的参数选择只使用此前已完成块。",
            "24组候选参数仍存在研究者自由度；区块式前瞻选型可减少但不能完全消除重复研究偏差。",
            "Top-k号码排序与六号码组合头奖概率不同；历史差异不能证明未来优势。",
            "没有自动修改生产选号器、1251期锁定档案或任何原始开奖数据。",
            "所有数据只来自仓库内用户手动维护的CSV；本研究不访问任何外部开奖网站或API。",
        ],
    }
    (out / "v72_recent500_research_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    run_research(pd.read_csv(ROOT / "data" / "lotto645.csv"))


if __name__ == "__main__":
    main()
