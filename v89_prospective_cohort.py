# -*- coding: utf-8 -*-
"""V8.9: prospective, immutable multi-draw cohort audit (offline only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "8.9.0-prospective-cohort-audit"
ROLES = {
    "V5": ("V5_raw_probability", "V5_baseline", "V5"),
    "V8.1": ("V8.1_batch_quota", "V8.1", "V8_1"),
    "V8.5": ("V8.5_low_overlap", "V8.5", "V8_5"),
    "RANDOM": ("uniform_random_control", "Uniform_random_control", "uniform", "random_control"),
}
PRIMARY = "batch_at_least_3_hits"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _role_metric(score: dict[str, Any], role: str) -> dict[str, Any] | None:
    metrics = score.get("strategy_metrics", {})
    for key in ROLES[role]:
        if key in metrics:
            return metrics[key]
    return None


def _is_timing_verified(score: dict[str, Any]) -> bool:
    timing = score.get("commitment_timing", {})
    return timing.get("status") == "verified_commit_date_precedes_draw_date"


def load_score_cohort(paths: list[str | Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load immutable single-draw score reports; reject duplicates and malformed records."""
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in paths:
        path = Path(item)
        raw = path.read_bytes()
        score = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(score, dict) or not isinstance(score.get("target_draw"), int):
            raise ValueError(f"评分文件结构无效：{path}")
        draw = int(score["target_draw"])
        if draw in seen:
            raise ValueError(f"同一期出现多个评分文件，拒绝重复计数：{draw}")
        seen.add(draw)
        if not score.get("pre_draw_history_fingerprint_verified", score.get("pre_draw_history_endpoint_verified", False)):
            excluded.append({"target_draw": draw, "file": path.name, "reason": "pre_draw_history_not_verified"})
            continue
        if not _is_timing_verified(score):
            excluded.append({"target_draw": draw, "file": path.name, "reason": "commitment_timing_not_verified"})
            continue
        missing = [role for role in ROLES if _role_metric(score, role) is None]
        if missing:
            excluded.append({"target_draw": draw, "file": path.name, "reason": "missing_strategy_roles:" + ",".join(missing)})
            continue
        if not score.get("actual_main_numbers") or len(score["actual_main_numbers"]) != 6:
            raise ValueError(f"开奖号码字段无效：{path}")
        accepted.append({
            "target_draw": draw,
            "file": path.name,
            "score_sha256": _sha256(raw),
            "score": score,
        })
    accepted.sort(key=lambda x: x["target_draw"])
    if len({x["target_draw"] for x in accepted}) != len(accepted):
        raise ValueError("有效队列中存在重复期号。")
    return accepted, excluded


def _binomial_upper_tail(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n))


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=float))
    out = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[int(index)]))
        out[int(index)] = float(running)
    return out


def _wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def analyze_v89_cohort(records: list[dict[str, Any]], *, minimum_inferential_draws: int = 100) -> dict[str, Any]:
    if minimum_inferential_draws < 2:
        raise ValueError("minimum_inferential_draws must be >=2")
    if not records:
        raise ValueError("没有通过前瞻锁定验证的完整评分记录。")
    draws = [int(x["target_draw"]) for x in records]
    if len(draws) != len(set(draws)):
        raise ValueError("评分队列包含重复期号。")
    if draws != sorted(draws):
        raise ValueError("评分记录必须按期号递增。")

    comparisons = []
    p_values = []
    details = {}
    for role in ("V5", "V8.1", "V8.5"):
        wins = losses = ties = 0
        model_success = random_success = 0
        model_hits = random_hits = 0.0
        model_ticket3 = random_ticket3 = 0.0
        for record in records:
            score = record["score"]
            model = _role_metric(score, role)
            random = _role_metric(score, "RANDOM")
            m = bool(model[PRIMARY])
            r = bool(random[PRIMARY])
            wins += int(m and not r)
            losses += int(r and not m)
            ties += int(m == r)
            model_success += int(m)
            random_success += int(r)
            model_hits += float(model["mean_hits_per_ticket"])
            random_hits += float(random["mean_hits_per_ticket"])
            model_ticket3 += float(model["ticket_3plus_rate"])
            random_ticket3 += float(random["ticket_3plus_rate"])
        discordant = wins + losses
        p = _binomial_upper_tail(wins, discordant)
        p_values.append(p)
        n = len(records)
        comparisons.append({
            "strategy": role,
            "primary_endpoint": PRIMARY,
            "draws": n,
            "model_batch_3plus_rate": model_success / n,
            "random_batch_3plus_rate": random_success / n,
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": ties,
            "one_sided_exact_mcnemar_p": p,
            "model_batch_3plus_wilson_95ci": _wilson(model_success, n),
            "random_batch_3plus_wilson_95ci": _wilson(random_success, n),
            "mean_hits_per_ticket_difference": (model_hits - random_hits) / n,
            "ticket_3plus_rate_difference": (model_ticket3 - random_ticket3) / n,
        })
    adjusted = _holm(p_values)
    for row, value in zip(comparisons, adjusted):
        row["holm_adjusted_p"] = value

    inferential_ready = len(records) >= minimum_inferential_draws
    return {
        "research_version": VERSION,
        "status": "prospective_cohort_audit",
        "first_target_draw": min(draws),
        "last_target_draw": max(draws),
        "scored_draws": len(draws),
        "minimum_draws_for_inferential_readiness": minimum_inferential_draws,
        "inferentially_ready": inferential_ready,
        "primary_endpoint": PRIMARY,
        "comparisons": comparisons,
        "multiple_testing": {"method": "Holm", "family_size": 3},
        "design": {
            "unit_of_analysis": "one independently pre-locked target draw",
            "paired_comparison": "each strategy versus the uniform-random ten-ticket portfolio from the same locked draw",
            "primary_test": "one-sided exact McNemar/binomial test on discordant paired outcomes",
            "direction": "strategy batch-3+ success more often than same-draw uniform control",
            "no_retraining_or_parameter_selection_on_scored_outcomes": True,
            "minimum_draws_are_a_readiness_gate_not_a_claim_of_adequate_power": True,
        },
        "evidence_boundary": {
            "historical_replay_is_not_counted_as_prospective": True,
            "only_date_verified_locks_included": True,
            "one_draw_cannot_establish_superiority": True,
            "production_model_changed": False,
            "manual_history_only": True,
            "official_lottery_site_or_api_accessed": False,
        },
        "interpretation": (
            "该报告只汇总预测锁定时间早于开奖日期、且开奖前历史指纹核验通过的逐期评分。"
            "100期只是最低审计就绪门槛，不代表已具备足够统计功效；最终仍需结合效应量、置信区间和预先规定的检验结果。"
        ),
    }


def run_v89(scores_dir: str | Path, output_path: str | Path, *, minimum_inferential_draws: int = 100) -> dict[str, Any]:
    root = Path(scores_dir)
    paths = sorted(root.glob("*.json"))
    accepted, excluded = load_score_cohort(paths)
    report = analyze_v89_cohort(accepted, minimum_inferential_draws=minimum_inferential_draws)
    report["excluded_records"] = excluded
    report["included_score_files"] = [
        {"target_draw": item["target_draw"], "file": item["file"], "sha256": item["score_sha256"]}
        for item in accepted
    ]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"V8.9队列报告已存在，拒绝覆盖：{target}")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V8.9 prospective cohort audit; offline only")
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-draws", type=int, default=100)
    args = parser.parse_args()
    report = run_v89(args.scores_dir, args.output, minimum_inferential_draws=args.minimum_draws)
    print(json.dumps({
        "status": report["status"],
        "scored_draws": report["scored_draws"],
        "inferentially_ready": report["inferentially_ready"],
        "output": args.output,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
