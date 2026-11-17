# -*- coding: utf-8 -*-
"""V7.0 chronological signal-integrity and label-placebo audit.

Research only. Uses the repository's manually maintained CSV, never fetches
lottery results, never changes the production predictor, and never touches
pre-existing forward locks. All folds are chronological and end by draw 1049;
these are historical development screens, not prospective evidence.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lotto_core import NUMBER_COLS, validate_frame
from v50_backtest import run_v50_walk_forward
from scripts.v64_exact_null_holm_audit import (
    exact_null_baselines, holm_adjust, sign_flip_pvalue, bootstrap_ci,
)

FOLDS = ((850, 899), (900, 949), (950, 999), (1000, 1049))
PLACEBO_SEEDS = (1701, 2701, 3701)
TRAINING_WINDOW = 500
MIN_HISTORY = 180
RIDGE_ALPHA = 1.0
TICKETS_PER_DRAW = 10
PAIR_STRENGTH = 0.5
MASTER_SEED = 70026
BOOTSTRAP_REPS = 20000


def ranking_hits(probabilities: np.ndarray, actual: list[int], k: int) -> int:
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (45,) or not np.all(np.isfinite(p)) or not 1 <= int(k) <= 45:
        raise ValueError("概率向量必须是45个有限值，且k位于1—45。")
    chosen = set((np.argsort(-p, kind="mergesort")[:int(k)] + 1).tolist())
    return int(len(chosen.intersection(int(x) for x in actual)))


def per_draw_metrics(row: dict) -> dict[str, float]:
    probabilities = np.asarray(json.loads(row["V5_45号码概率_JSON"]), dtype=float)
    actual = [int(x) for x in json.loads(row["实际主号码"])]
    if probabilities.shape != (45,) or not np.isclose(probabilities.sum(), 6.0, atol=1e-5):
        raise ValueError("V7.0发现45维概率向量异常。")
    if len(actual) != 6 or len(set(actual)) != 6 or min(actual) < 1 or max(actual) > 45:
        raise ValueError("V7.0发现实际主号码异常。")
    return {
        "top6_hits": float(ranking_hits(probabilities, actual, 6)),
        "top10_hits": float(ranking_hits(probabilities, actual, 10)),
        "top15_hits": float(ranking_hits(probabilities, actual, 15)),
        "Brier": float(row["V5_Brier"]),
        "LogLoss": float(row["V5_LogLoss"]),
    }


def improvement_from_null(metrics: dict[str, float], nulls: dict[str, float]) -> dict[str, float]:
    """Positive values always mean improvement over the exact uniform null."""
    return {
        key: (nulls[key] - value if key in ("Brier", "LogLoss") else value - nulls[key])
        for key, value in metrics.items()
    }


def _run_fold(history: pd.DataFrame, start: int, end: int, placebo_seed: int | None) -> pd.DataFrame:
    prefix = history.loc[history["draw_no"] <= end].copy().reset_index(drop=True)
    if len(prefix) != end or int(prefix["draw_no"].iloc[-1]) != end:
        raise ValueError(f"V7.0折{start}—{end}缺少完整历史前缀。")
    result = run_v50_walk_forward(
        prefix,
        test_draws=end - start + 1,
        training_window=TRAINING_WINDOW,
        min_history=MIN_HISTORY,
        ridge_alpha=RIDGE_ALPHA,
        tickets_per_draw=TICKETS_PER_DRAW,
        seed=MASTER_SEED,
        bootstrap_reps=100,
        placebo_seed=placebo_seed,
        pair_sampling_strength=PAIR_STRENGTH,
    )
    detail = result["detail"].copy()
    expected = list(range(start, end + 1))
    if detail["目标期号"].astype(int).tolist() != expected:
        raise RuntimeError(f"V7.0折{start}—{end}目标期不匹配。")
    if not (detail["训练结束期"] < detail["目标期号"]).all():
        raise RuntimeError("V7.0检测到时间泄漏。")
    return detail


def run_experiment(frame: pd.DataFrame) -> dict:
    history = validate_frame(frame).sort_values("draw_no").reset_index(drop=True)
    if history.empty or np.any(np.diff(history["draw_no"].to_numpy(dtype=int)) != 1):
        raise ValueError("V7.0要求手动导入且期号连续的历史数据。")
    if int(history["draw_no"].iloc[0]) != 1 or int(history["draw_no"].iloc[-1]) < 1049:
        raise ValueError("V7.0开发审计要求数据至少连续覆盖第1—1049期。")
    # Hard pin the research screen to the already explored development period.
    history = history.loc[history["draw_no"] <= 1049].copy().reset_index(drop=True)
    all_nulls = exact_null_baselines()
    nulls = {key: all_nulls[key] for key in ("top6_hits", "top10_hits", "top15_hits", "Brier", "LogLoss")}
    records = []
    fold_reports = []

    for fold_index, (start, end) in enumerate(FOLDS, 1):
        real_detail = _run_fold(history, start, end, None)
        placebo_details = {
            seed: _run_fold(history, start, end, seed)
            for seed in PLACEBO_SEEDS
        }
        fold_improvements = []
        placebo_means = {seed: [] for seed in PLACEBO_SEEDS}
        for row in real_detail.to_dict(orient="records"):
            metrics = per_draw_metrics(row)
            improvement = improvement_from_null(metrics, nulls)
            item = {"fold": fold_index, "target_draw": int(row["目标期号"]), "V5_45号码概率_JSON": row["V5_45号码概率_JSON"], "实际主号码": row["实际主号码"], **metrics}
            item.update({f"improvement_{k}": v for k, v in improvement.items()})
            for seed, placebo_detail in placebo_details.items():
                p_row = placebo_detail.loc[placebo_detail["目标期号"] == row["目标期号"]]
                if len(p_row) != 1:
                    raise RuntimeError("Placebo目标期与真实模型未能一一配对。")
                p_metrics = per_draw_metrics(p_row.iloc[0].to_dict())
                p_improvement = improvement_from_null(p_metrics, nulls)
                for key, value in p_improvement.items():
                    item[f"placebo_{seed}_{key}"] = value
                    placebo_means[seed].append((key, value))
            records.append(item)
            fold_improvements.append(improvement)
        fold_reports.append({
            "fold": fold_index,
            "target_range": [start, end],
            "target_draws": len(real_detail),
            "placebo_seeds": list(PLACEBO_SEEDS),
            "mean_improvement_vs_exact_uniform": {
                key: float(np.mean([x[key] for x in fold_improvements]))
                for key in nulls
            },
            "placebo_mean_improvement_vs_exact_uniform": {
                str(seed): {
                    key: float(np.mean([
                        improvement_from_null(per_draw_metrics(row), nulls)[key]
                        for row in placebo_details[seed].to_dict(orient="records")
                    ]))
                    for key in nulls
                }
                for seed in PLACEBO_SEEDS
            },
        })

    detail = pd.DataFrame(records)
    if len(detail) != 200 or detail["target_draw"].nunique() != 200:
        raise RuntimeError("V7.0必须恰好评估200个唯一目标期。")
    if detail["target_draw"].min() != 850 or detail["target_draw"].max() != 1049:
        raise RuntimeError("V7.0目标期范围被意外改变。")

    summary = []
    raw_p = {}
    for index, metric in enumerate(nulls):
        values = detail[f"improvement_{metric}"].to_numpy(dtype=float)
        low, high = bootstrap_ci(values, repetitions=BOOTSTRAP_REPS, seed=MASTER_SEED + index * 101)
        p_value = sign_flip_pvalue(values, repetitions=BOOTSTRAP_REPS, seed=MASTER_SEED + index * 103)
        raw_p[metric] = p_value
        row = {
            "metric": metric,
            "direction": "higher_is_better" if metric.startswith("top") else "lower_is_better",
            "model_mean": float(detail[metric].mean()),
            "exact_uniform_null_mean": float(nulls[metric]),
            "improvement_positive_is_better": float(values.mean()),
            "improvement_95ci_low": low,
            "improvement_95ci_high": high,
            "p_value_two_sided_sign_flip": p_value,
        }
        for seed in PLACEBO_SEEDS:
            placebo_values = detail[f"placebo_{seed}_{metric}"].to_numpy(dtype=float)
            row[f"placebo_{seed}_mean_improvement"] = float(placebo_values.mean())
        summary.append(row)
    adjusted = holm_adjust(raw_p)
    for row in summary:
        row["holm_adjusted_p_value_5_tests"] = adjusted[row["metric"]]
        row["passes_holm_0_05"] = bool(adjusted[row["metric"]] < 0.05)

    out = ROOT / "artifacts" / "v70"
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "v70_signal_integrity_detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(out / "v70_signal_integrity_summary.csv", index=False, encoding="utf-8-sig")
    fingerprint = hashlib.sha256(history.to_csv(index=False).encode("utf-8")).hexdigest()
    report = {
        "experiment": "V7.0 chronological signal-integrity and label-placebo audit",
        "status": "historical_development_screen_not_independent_or_prospective_evidence",
        "data_range": [1, 1049],
        "target_folds": [{"start": a, "end": b} for a, b in FOLDS],
        "target_draws": int(detail["target_draw"].nunique()),
        "training_window": TRAINING_WINDOW,
        "min_history": MIN_HISTORY,
        "ridge_alpha": RIDGE_ALPHA,
        "pair_sampling_strength": PAIR_STRENGTH,
        "tickets_per_draw": TICKETS_PER_DRAW,
        "placebo_seeds": list(PLACEBO_SEEDS),
        "placebo_method": "permute complete historical 45-label vectors across feature rows within each rolling training window; target labels remain untouched",
        "null": "exact theoretical uniform 6-of-45 expectations",
        "multiple_testing": "Holm-Bonferroni across five predeclared metrics",
        "summary": summary,
        "fold_reports": fold_reports,
        "data_fingerprint_sha256": fingerprint,
        "limitations": [
            "目标期850—1049已用于先前V6.x历史研究，不是独立盲测或前瞻证据。",
            "标签置换仅为伪信号诊断；置换后模型并非可部署的预测策略。",
            "每期45个号码结果相关，主要统计单位为200个开奖期。",
            "Top-k命中衡量号码排序，不等同于六号码组合头奖概率。",
            "历史开发审计不得修改已锁定的1251期前瞻记录。",
            "任何生产模型变更仍须先冻结候选版本，再以未来开奖进行前瞻比较。",
        ],
    }
    (out / "v70_signal_integrity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    main_history = pd.read_csv(ROOT / "data" / "lotto645.csv")
    run_experiment(main_history)


if __name__ == "__main__":
    main()
