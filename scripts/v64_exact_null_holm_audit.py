# -*- coding: utf-8 -*-
"""V6.4 exact-null and multiplicity audit; research only, never changes production."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N_DRAWS = 200
SEED = 64026
REPS = 20000
TOP_KS = (6, 10, 15)
METRICS = tuple(f"top{k}_hits" for k in TOP_KS) + ("ndcg_at_6", "Brier", "LogLoss")


def exact_null_baselines() -> dict[str, float]:
    """Conditional expectations under a uniform 6-of-45 draw."""
    p = 6.0 / 45.0
    discounts = 1.0 / np.log2(np.arange(2, 8, dtype=float))
    return {
        "top6_hits": 6.0 * 6.0 / 45.0,
        "top10_hits": 6.0 * 10.0 / 45.0,
        "top15_hits": 6.0 * 15.0 / 45.0,
        "ndcg_at_6": 6.0 / 45.0,
        "Brier": (6.0 * (1.0 - p) ** 2 + 39.0 * p**2) / 45.0,
        "LogLoss": -(6.0 * np.log(p) + 39.0 * np.log(1.0 - p)) / 45.0,
    }


def sign_flip_pvalue(values, repetitions=REPS, seed=SEED) -> float:
    """Two-sided paired sign-flip test of zero mean; target draws are units."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("差值必须是一维有限数组，且至少包含两个目标期。")
    rng = np.random.default_rng(seed)
    observed = abs(float(x.mean()))
    exceed = 0
    batch = 1000
    done = 0
    while done < repetitions:
        count = min(batch, repetitions - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(x)))
        permuted = np.abs((signs * x).mean(axis=1))
        exceed += int(np.count_nonzero(permuted >= observed - 1e-15))
        done += count
    return float((exceed + 1) / (repetitions + 1))


def bootstrap_ci(values, repetitions=REPS, seed=SEED):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("Bootstrap差值必须是一维有限数组，且至少包含两个目标期。")
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    batch = 500
    done = 0
    while done < repetitions:
        count = min(batch, repetitions - done)
        idx = rng.integers(0, len(x), size=(count, len(x)))
        means[done:done + count] = x[idx].mean(axis=1)
        done += count
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, preserving original keys."""
    if not p_values or any(not 0.0 <= p <= 1.0 for p in p_values.values()):
        raise ValueError("p值必须非空且位于[0,1]。")
    ordered = sorted(p_values, key=p_values.get)
    m = len(ordered)
    adjusted = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def main() -> None:
    source = ROOT / "artifacts" / "v63" / "v63_probability_ranking_detail.csv"
    if not source.exists():
        raise FileNotFoundError("缺少V6.3逐期明细；必须先完成V6.3。")
    detail = pd.read_csv(source)
    if len(detail) != N_DRAWS or detail["目标期号"].nunique() != N_DRAWS:
        raise ValueError("V6.4要求V6.3提供200个唯一目标期。")
    if not (detail["训练结束期"] < detail["目标期号"]).all():
        raise ValueError("发现训练结束期不早于目标期，拒绝审计。")

    nulls = exact_null_baselines()
    tests = {}
    rows = []
    for model in ("V5", "V4"):
        for metric in METRICS:
            model_col = f"{model}_{metric}" if metric not in ("Brier", "LogLoss") else f"{model}_{metric}"
            if model_col not in detail:
                raise ValueError(f"V6.3明细缺少字段：{model_col}")
            # Positive means better: more ranking hits, or lower probability loss.
            if metric in ("Brier", "LogLoss"):
                improvement = nulls[metric] - detail[model_col].to_numpy(dtype=float)
            else:
                improvement = detail[model_col].to_numpy(dtype=float) - nulls[metric]
            key = f"{model}_{metric}"
            pval = sign_flip_pvalue(improvement, seed=SEED + len(rows) * 31)
            low, high = bootstrap_ci(improvement, seed=SEED + len(rows) * 37)
            tests[key] = pval
            rows.append({
                "model": model, "metric": metric,
                "model_mean": float(detail[model_col].mean()),
                "exact_uniform_null_mean": float(nulls[metric]),
                "improvement_positive_is_better": float(improvement.mean()),
                "improvement_95ci_low": low,
                "improvement_95ci_high": high,
                "p_value_two_sided_sign_flip": pval,
            })

    adjusted = holm_adjust(tests)
    for row in rows:
        row["holm_adjusted_p_value_12_tests"] = adjusted[f"{row['model']}_{row['metric']}"]
        row["passes_holm_0_05"] = bool(row["holm_adjusted_p_value_12_tests"] < 0.05)

    out = ROOT / "artifacts" / "v64"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "v64_exact_null_holm_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "experiment": "V6.4 exact uniform null and Holm multiplicity audit",
        "status": "historical_development_screen_not_independent_or_prospective_evidence",
        "target_draws": N_DRAWS,
        "target_unit": "one chronological target draw",
        "exact_uniform_null_expectations": nulls,
        "multiple_testing": {
            "method": "Holm-Bonferroni",
            "family": "12 predeclared model-by-metric tests: 2 models x 6 metrics",
            "alpha": 0.05,
        },
        "test": "two-sided paired target-level sign-flip test; 20000 Monte Carlo sign assignments",
        "summary": rows,
        "limitations": [
            "本分析沿用V6.3第850—1049期历史开发样本，不是独立盲测或前瞻证据。",
            "精确基准是公平6/45开奖下的理论期望；Top-k及NDCG按每期45号码排序计算。",
            "符号翻转检验依赖目标期差值在零假设下近似对称；Monte Carlo p值不是解析精确p值。",
            "Holm校正控制本次12项预先列明检验的家族错误率，不消除反复查看历史数据造成的研究者自由度。",
            "任何结果均不自动更改正式选号算法；未来开奖前锁定并逐期评分仍是必要验证。",
        ],
    }
    (out / "v64_exact_null_holm_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
