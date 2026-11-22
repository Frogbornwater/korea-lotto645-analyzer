# -*- coding: utf-8 -*-
"""V9.0 prospective probability-score audit; offline and read-only."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from v89_prospective_cohort import load_score_cohort

VERSION = "9.0.0-prospective-probability-audit"
METRICS = ("brier", "logloss")
TOP_K = (6, 10, 15)
UNIFORM_P = 6.0 / 45.0
UNIFORM_BRIER = (6 * (1 - UNIFORM_P) ** 2 + 39 * UNIFORM_P ** 2) / 45.0
UNIFORM_LOGLOSS = -(6 * math.log(UNIFORM_P) + 39 * math.log(1 - UNIFORM_P)) / 45.0


def _finite_metric(metrics: dict[str, Any], key: str, draw: int) -> float:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"第{draw}期概率指标 {key} 缺失或无效。")
    return float(value)


def _bootstrap_mean_ci(values: list[float], *, seed: int, reps: int = 10000) -> list[float]:
    if not values:
        raise ValueError("bootstrap requires at least one draw")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [means[int(0.025 * (reps - 1))], means[int(0.975 * (reps - 1))]]


def analyze_v90_probability(records: list[dict[str, Any]], *, bootstrap_reps: int = 10000) -> dict[str, Any]:
    if not records:
        raise ValueError("没有可用于概率审计的前瞻性评分记录。")
    if bootstrap_reps < 1000:
        raise ValueError("bootstrap_reps 必须至少为1000。")
    draws = [int(row["target_draw"]) for row in records]
    if len(draws) != len(set(draws)) or draws != sorted(draws):
        raise ValueError("期号必须唯一且按升序排列。")

    per_draw = []
    for row in records:
        draw = int(row["target_draw"])
        score = row["score"]
        probability = score.get("v5_probability_metrics")
        if not isinstance(probability, dict):
            raise ValueError(f"第{draw}期缺少 v5_probability_metrics。")
        item = {"target_draw": draw}
        for metric in METRICS:
            item[metric] = _finite_metric(probability, metric, draw)
        for k in TOP_K:
            item[f"top{k}_hits"] = _finite_metric(probability, f"top{k}_hits", draw)
        per_draw.append(item)

    n = len(per_draw)
    result_metrics = {}
    for metric, baseline in (("brier", UNIFORM_BRIER), ("logloss", UNIFORM_LOGLOSS)):
        values = [x[metric] for x in per_draw]
        differences = [x - baseline for x in values]
        result_metrics[metric] = {
            "mean": sum(values) / n,
            "uniform_reference": baseline,
            "mean_difference_vs_uniform": sum(differences) / n,
            "difference_bootstrap_95ci": _bootstrap_mean_ci(
                differences, seed=9000 + (1 if metric == "brier" else 2), reps=bootstrap_reps
            ),
            "interpretation": "lower_is_better",
        }

    top_k = {}
    for k in TOP_K:
        values = [x[f"top{k}_hits"] for x in per_draw]
        expected = k * 6.0 / 45.0
        differences = [x - expected for x in values]
        top_k[str(k)] = {
            "mean_hits": sum(values) / n,
            "uniform_expected_hits": expected,
            "mean_difference_vs_uniform": sum(differences) / n,
            "difference_bootstrap_95ci": _bootstrap_mean_ci(
                differences, seed=9100 + k, reps=bootstrap_reps
            ),
            "interpretation": "higher_is_better",
        }

    return {
        "research_version": VERSION,
        "status": "prospective_probability_audit",
        "first_target_draw": min(draws),
        "last_target_draw": max(draws),
        "scored_draws": n,
        "bootstrap": {"method": "draw-level paired percentile bootstrap", "replicates": bootstrap_reps, "seed_policy": "fixed_metric_specific_seeds"},
        "probability_metrics": result_metrics,
        "ranking_metrics": {"top_k_hits": top_k},
        "limitations": {
            "historical_replay_excluded": True,
            "only_pre_draw_verified_locks_included": True,
            "brier_and_logloss_are_draw_level_scores": True,
            "uniform_reference_uses_marginal_probability_6_over_45": True,
            "raw_45d_probability_vectors_not_retained_in_score_schema": True,
            "reliability_diagram_and_calibration_slope_not_estimable_from_current_score_files": True,
            "bootstrap_intervals_are_descriptive_not_proof_of_superiority": True,
            "production_model_changed": False,
            "official_lottery_site_or_api_accessed": False,
        },
        "interpretation": "该报告只分析开奖前锁定且时间与历史指纹核验通过的逐期概率评分。Brier/LogLoss低于均匀边际概率参考值才是较好方向；Top-K命中高于理论随机期望才是较好方向。置信区间跨越0时，不应宣称有优势。"
    }


def run_v90(scores_dir: str | Path, output_path: str | Path, *, bootstrap_reps: int = 10000) -> dict[str, Any]:
    root = Path(scores_dir)
    accepted, excluded = load_score_cohort(sorted(root.glob("*.json")))
    report = analyze_v90_probability(accepted, bootstrap_reps=bootstrap_reps)
    report["excluded_records"] = excluded
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"V9.0报告已存在，拒绝覆盖：{target}")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V9.0 prospective probability audit; offline only")
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()
    report = run_v90(args.scores_dir, args.output, bootstrap_reps=args.bootstrap_reps)
    print(json.dumps({"status": report["status"], "scored_draws": report["scored_draws"], "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
