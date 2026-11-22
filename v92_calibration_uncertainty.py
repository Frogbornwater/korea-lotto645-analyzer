# -*- coding: utf-8 -*-
"""V9.2: draw-clustered uncertainty for prospective calibration diagnostics."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from v89_prospective_cohort import load_score_cohort
from v91_probability_vector import _draw_metrics

VERSION = "9.2.0-calibration-uncertainty"
BINS = 10


def _bin_summaries(rows: list[dict[str, Any]], bins: int = BINS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros((len(rows), bins), dtype=float)
    pred = np.zeros((len(rows), bins), dtype=float)
    observed = np.zeros((len(rows), bins), dtype=float)
    for j, row in enumerate(rows):
        p = np.asarray(row["p"], dtype=float)
        y = np.asarray(row["y"], dtype=float)
        indices = np.minimum((p * bins).astype(int), bins - 1)
        for b in range(bins):
            mask = indices == b
            counts[j, b] = int(mask.sum())
            if counts[j, b]:
                pred[j, b] = float(p[mask].sum())
                observed[j, b] = float(y[mask].sum())
    return counts, pred, observed


def _ece(counts: np.ndarray, pred: np.ndarray, observed: np.ndarray) -> float:
    c = counts.sum(axis=0)
    p = pred.sum(axis=0)
    y = observed.sum(axis=0)
    total = float(c.sum())
    if total <= 0:
        raise ValueError("ECE requires non-empty probability outcomes")
    valid = c > 0
    return float(np.sum(c[valid] / total * np.abs(p[valid] / c[valid] - y[valid] / c[valid])))


def _cluster_bootstrap(
    rows: list[dict[str, Any]], *, reps: int = 10000, seed: int = 9201, bins: int = BINS
) -> dict[str, list[float]]:
    if reps < 1000:
        raise ValueError("bootstrap_reps必须至少1000")
    n = len(rows)
    if n == 0:
        raise ValueError("bootstrap requires at least one draw")
    counts, pred, observed = _bin_summaries(rows, bins)
    gap_by_draw = np.asarray([
        (sum(row["y"]) - sum(row["p"])) / 45.0 for row in rows
    ], dtype=float)
    rng = np.random.default_rng(seed)
    ece_values = np.empty(reps, dtype=float)
    gap_values = np.empty(reps, dtype=float)
    for r in range(reps):
        sample = rng.integers(0, n, size=n)
        ece_values[r] = _ece(counts[sample], pred[sample], observed[sample])
        gap_values[r] = float(gap_by_draw[sample].mean())
    return {
        "ece_95ci": [float(x) for x in np.quantile(ece_values, [0.025, 0.975])],
        "calibration_in_the_large_gap_95ci": [float(x) for x in np.quantile(gap_values, [0.025, 0.975])],
    }


def analyze_v92_uncertainty(
    records: list[dict[str, Any]], *, bootstrap_reps: int = 10000, minimum_ready_draws: int = 100
) -> dict[str, Any]:
    if not records:
        raise ValueError("没有可用于V9.2校准不确定性分析的前瞻评分记录。")
    if minimum_ready_draws < 2:
        raise ValueError("minimum_ready_draws必须至少2")
    draws = [int(x["target_draw"]) for x in records]
    if draws != sorted(draws) or len(draws) != len(set(draws)):
        raise ValueError("期号必须唯一且按升序排列。")
    rows = [_draw_metrics(x) for x in records]
    counts, pred, observed = _bin_summaries(rows)
    total = float(counts.sum())
    mean_p = float(pred.sum() / total)
    observed_rate = float(observed.sum() / total)
    gap = observed_rate - mean_p
    ece = _ece(counts, pred, observed)
    uncertainty = _cluster_bootstrap(rows, reps=bootstrap_reps)
    return {
        "research_version": VERSION,
        "status": "prospective_calibration_uncertainty_audit",
        "first_target_draw": min(draws),
        "last_target_draw": max(draws),
        "scored_draws": len(draws),
        "calibration_uncertainty_ready": len(draws) >= minimum_ready_draws,
        "minimum_draws_for_uncertainty_readiness": minimum_ready_draws,
        "metrics": {
            "mean_predicted_inclusion": mean_p,
            "observed_inclusion_frequency": observed_rate,
            "calibration_in_the_large_gap_observed_minus_predicted": gap,
            "expected_calibration_error_equal_width": ece,
            **uncertainty,
        },
        "design": {
            "bootstrap_unit": "target draw; all 45 outcomes are resampled together",
            "bootstrap_repetitions": bootstrap_reps,
            "reliability_bins": BINS,
            "calibration_gap_direction": "positive means underprediction; negative means overprediction",
            "minimum_draw_gate_is_not_a_power_guarantee": True,
            "only_pre_draw_verified_locks_included": True,
            "historical_replay_excluded": True,
            "production_model_changed": False,
            "official_lottery_site_or_api_accessed": False,
        },
        "interpretation": "ECE和校准差距的区间描述的是跨期抽样不确定性；样本门槛不等于预测优势证明，也不能替代独立前瞻验证。"
    }


def run_v92(scores_dir: str | Path, output_path: str | Path, *, bootstrap_reps: int = 10000) -> dict[str, Any]:
    accepted, excluded = load_score_cohort(sorted(Path(scores_dir).glob("*.json")))
    eligible = []
    for item in accepted:
        if isinstance(item["score"].get("probability_forecast"), dict):
            eligible.append(item)
        else:
            excluded.append({"target_draw": item["target_draw"], "file": item["file"], "reason": "missing_probability_forecast_vector"})
    report = analyze_v92_uncertainty(eligible, bootstrap_reps=bootstrap_reps)
    report["excluded_records"] = excluded
    report["included_score_files"] = [
        {"target_draw": x["target_draw"], "file": x["file"], "sha256": x["score_sha256"]}
        for x in eligible
    ]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"V9.2报告已存在，拒绝覆盖：{target}")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V9.2 draw-clustered calibration uncertainty audit")
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()
    report = run_v92(args.scores_dir, args.output, bootstrap_reps=args.bootstrap_reps)
    print(json.dumps({"status": report["status"], "scored_draws": report["scored_draws"], "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
