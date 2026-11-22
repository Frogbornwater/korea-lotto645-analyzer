# -*- coding: utf-8 -*-
"""V9.1 prospective 45-dimensional probability-vector calibration audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from v89_prospective_cohort import load_score_cohort

VERSION = "9.1.0-prospective-vector-calibration"
UNIFORM_P = 6.0 / 45.0


def _canonical_vector_hash(values: list[float]) -> str:
    raw = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _bootstrap_ci(values: list[float], seed: int, reps: int = 10000) -> list[float]:
    if not values:
        raise ValueError("bootstrap requires at least one draw")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return [means[int(.025 * (reps - 1))], means[int(.975 * (reps - 1))]]


def _draw_metrics(record: dict[str, Any]) -> dict[str, float]:
    score = record["score"]
    forecast = score.get("probability_forecast")
    if not isinstance(forecast, dict) or forecast.get("source") != "immutable_pre_draw_lock_bundle":
        raise ValueError(f'第{record["target_draw"]}期缺少可信概率向量。')
    p = forecast.get("marginal_inclusion_probabilities")
    if not isinstance(p, list) or len(p) != 45:
        raise ValueError(f'第{record["target_draw"]}期概率向量长度不是45。')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) for x in p):
        raise ValueError(f'第{record["target_draw"]}期概率向量含非有限值。')
    p = [float(x) for x in p]
    if any(x < 0 or x > 1 for x in p) or not math.isclose(sum(p), 6.0, abs_tol=1e-6):
        raise ValueError(f'第{record["target_draw"]}期概率向量不满足边际概率约束。')
    if forecast.get("vector_sha256") != _canonical_vector_hash(p):
        raise ValueError(f'第{record["target_draw"]}期概率向量哈希不匹配。')
    actual = score.get("actual_main_numbers", [])
    if len(actual) != 6 or len(set(actual)) != 6 or any(int(x) < 1 or int(x) > 45 for x in actual):
        raise ValueError(f'第{record["target_draw"]}期实际号码无效。')
    y = [1.0 if i + 1 in set(int(x) for x in actual) else 0.0 for i in range(45)]
    brier = sum((a - b) ** 2 for a, b in zip(p, y)) / 45.0
    logloss = -sum(b * math.log(max(a, 1e-15)) + (1-b) * math.log(max(1-a, 1e-15)) for a, b in zip(p, y)) / 45.0
    return {"p": p, "y": y, "brier": brier, "logloss": logloss, "mean_probability": sum(p)/45.0}


def analyze_v91_calibration(records: list[dict[str, Any]], *, bootstrap_reps: int = 10000, bins: int = 10) -> dict[str, Any]:
    if not records:
        raise ValueError("没有通过前瞻锁定验证的概率向量评分记录。")
    if bootstrap_reps < 1000 or bins < 2:
        raise ValueError("bootstrap_reps必须至少1000，bins必须至少2。")
    draws = [int(x["target_draw"]) for x in records]
    if draws != sorted(draws) or len(draws) != len(set(draws)):
        raise ValueError("期号必须唯一且按升序排列。")
    per_draw = [_draw_metrics(x) for x in records]
    n = len(per_draw)
    brier_ref = (6 * (1-UNIFORM_P)**2 + 39 * UNIFORM_P**2) / 45.0
    logloss_ref = -(6*math.log(UNIFORM_P) + 39*math.log(1-UNIFORM_P)) / 45.0
    brier_diff = [x["brier"] - brier_ref for x in per_draw]
    logloss_diff = [x["logloss"] - logloss_ref for x in per_draw]

    flat_p = [p for row in per_draw for p in row["p"]]
    flat_y = [y for row in per_draw for y in row["y"]]
    reliability = []
    ece = 0.0
    total = len(flat_p)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(flat_p) if lo <= p < hi or (b == bins-1 and p == 1.0)]
        if not idx:
            reliability.append({"bin": b, "lower": lo, "upper": hi, "count": 0, "mean_predicted": None, "observed_frequency": None})
            continue
        mp = sum(flat_p[i] for i in idx) / len(idx)
        obs = sum(flat_y[i] for i in idx) / len(idx)
        ece += len(idx) / total * abs(mp - obs)
        reliability.append({"bin": b, "lower": lo, "upper": hi, "count": len(idx), "mean_predicted": mp, "observed_frequency": obs})

    return {
        "research_version": VERSION,
        "status": "prospective_vector_calibration_audit",
        "first_target_draw": min(draws),
        "last_target_draw": max(draws),
        "scored_draws": n,
        "probability_vector_count": total,
        "metrics": {
            "brier_mean": sum(x["brier"] for x in per_draw)/n,
            "brier_uniform_reference": brier_ref,
            "brier_difference_vs_uniform": sum(brier_diff)/n,
            "brier_difference_bootstrap_95ci": _bootstrap_ci(brier_diff, 9201, bootstrap_reps),
            "logloss_mean": sum(x["logloss"] for x in per_draw)/n,
            "logloss_uniform_reference": logloss_ref,
            "logloss_difference_vs_uniform": sum(logloss_diff)/n,
            "logloss_difference_bootstrap_95ci": _bootstrap_ci(logloss_diff, 9202, bootstrap_reps),
            "mean_marginal_probability": sum(flat_p)/total,
            "observed_inclusion_frequency": sum(flat_y)/total,
            "expected_calibration_error_equal_width": ece,
        },
        "reliability_bins": reliability,
        "design": {
            "unit_of_bootstrap": "target draw; all 45 number outcomes remain clustered within draw",
            "binning": "fixed equal-width bins on [0,1]",
            "uniform_reference": "marginal inclusion probability 6/45",
            "only_pre_draw_verified_locks_included": True,
            "historical_replay_excluded": True,
            "production_model_changed": False,
            "official_lottery_site_or_api_accessed": False,
        },
        "interpretation": "可靠性分箱和ECE是描述性校准诊断；只有跨期置信区间、样本量和预先定义的检验共同支持时，才可讨论预测优势。"
    }


def run_v91(scores_dir: str | Path, output_path: str | Path, *, bootstrap_reps: int = 10000, bins: int = 10) -> dict[str, Any]:
    accepted, excluded = load_score_cohort(sorted(Path(scores_dir).glob("*.json")))
    eligible = []
    for item in accepted:
        if isinstance(item["score"].get("probability_forecast"), dict):
            eligible.append(item)
        else:
            excluded.append({"target_draw": item["target_draw"], "file": item["file"], "reason": "missing_probability_forecast_vector"})
    report = analyze_v91_calibration(eligible, bootstrap_reps=bootstrap_reps, bins=bins)
    report["excluded_records"] = excluded
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"V9.1报告已存在，拒绝覆盖：{target}")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V9.1 prospective probability-vector calibration audit")
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    report = run_v91(args.scores_dir, args.output, bootstrap_reps=args.bootstrap_reps, bins=args.bins)
    print(json.dumps({"status": report["status"], "scored_draws": report["scored_draws"], "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
