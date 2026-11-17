# -*- coding: utf-8 -*-
"""V6.3 chronological audit of number-probability ranking utility.

Historical development screen only. Uses repository history through draw 1049,
predicts each target from strictly earlier draws, and compares V5/V4 rankings
with a pre-seeded uniform random ranking. No network or external data access.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lotto_core import NUMBER_COLS, validate_frame
from v50_backtest import run_v50_walk_forward

DATA_FIRST_DRAW = 1
DATA_LAST_DRAW = 1049
FOLD_SIZE = 50
FOLD_STARTS = (850, 900, 950, 1000)
TRAINING_WINDOW = 500
MIN_HISTORY = 180
RIDGE_ALPHA = 1.0
TICKETS_PER_DRAW = 10
SEED = 63026
BOOTSTRAP_REPS = 5000
TOP_KS = (6, 10, 15)
METRICS = tuple(f"top{k}_hits" for k in TOP_KS) + ("ndcg_at_6",)


def chronological_folds() -> list[tuple[int, int]]:
    folds = [(int(start), int(start) + FOLD_SIZE - 1) for start in FOLD_STARTS]
    if tuple(sorted(FOLD_STARTS)) != FOLD_STARTS:
        raise ValueError("时间折必须按起始期升序排列。")
    for previous, current in zip(folds, folds[1:]):
        if current[0] <= previous[1]:
            raise ValueError("测试折必须互不重叠。")
    return folds


def rank_metrics(probabilities, actual_numbers, random_order) -> dict[str, float]:
    """Evaluate one 45-number ranking; ties use stable numeric order."""
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (45,) or not np.all(np.isfinite(p)):
        raise ValueError("每期必须提供45个有限号码概率。")
    if np.any(p < 0):
        raise ValueError("号码概率不能为负数。")
    actual = {int(x) for x in actual_numbers}
    if len(actual) != 6 or any(x < 1 or x > 45 for x in actual):
        raise ValueError("实际号码必须是1—45之间的6个不同号码。")
    order = np.argsort(-p, kind="stable") + 1
    hits = np.asarray([int(x in actual) for x in order], dtype=float)
    result = {f"top{k}_hits": float(hits[:k].sum()) for k in TOP_KS}
    discounts = 1.0 / np.log2(np.arange(2, 8, dtype=float))
    dcg = float(np.sum(hits[:6] * discounts))
    ideal = float(np.sum(discounts))
    result["ndcg_at_6"] = dcg / ideal
    random_order = np.asarray(random_order, dtype=int)
    if sorted(random_order.tolist()) != list(range(1, 46)):
        raise ValueError("随机基准排序必须恰好包含1—45。")
    random_hits = np.asarray([int(x in actual) for x in random_order], dtype=float)
    for k in TOP_KS:
        result[f"random_top{k}_hits"] = float(random_hits[:k].sum())
    random_dcg = float(np.sum(random_hits[:6] * discounts))
    result["random_ndcg_at_6"] = random_dcg / ideal
    return result


def bootstrap_ci(values, repetitions=BOOTSTRAP_REPS, seed=SEED):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("Bootstrap需要至少两个目标期的配对差值。")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(x), size=(int(repetitions), len(x)))
    means = x[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "lotto645.csv")
    if "draw_no" not in raw.columns:
        raise ValueError("数据文件缺少 draw_no 字段。")
    raw = raw.loc[pd.to_numeric(raw["draw_no"], errors="coerce") <= DATA_LAST_DRAW].copy()
    history = validate_frame(raw).reset_index(drop=True)
    draws = history["draw_no"].to_numpy(dtype=int)
    if len(history) != DATA_LAST_DRAW or int(draws[0]) != DATA_FIRST_DRAW or int(draws[-1]) != DATA_LAST_DRAW:
        raise ValueError("V6.3要求恰好包含第1—1049期。")
    if np.any(np.diff(draws) != 1):
        raise ValueError("历史期号不连续，拒绝运行。")

    parts = []
    fold_meta = []
    for fold_index, (start, end) in enumerate(chronological_folds(), 1):
        fold_history = history.loc[history["draw_no"] <= end].copy().reset_index(drop=True)
        result = run_v50_walk_forward(
            fold_history,
            test_draws=FOLD_SIZE,
            training_window=TRAINING_WINDOW,
            min_history=MIN_HISTORY,
            ridge_alpha=RIDGE_ALPHA,
            tickets_per_draw=TICKETS_PER_DRAW,
            seed=SEED + fold_index,
            bootstrap_reps=100,
            pair_sampling_strength=0.0,
        )
        if (result["test_start_draw"], result["test_end_draw"], result["test_draws"]) != (start, end, FOLD_SIZE):
            raise RuntimeError(f"第{fold_index}折测试区间不匹配。")
        detail = result["detail"].copy()
        records = []
        for row in detail.to_dict(orient="records"):
            target = int(row["目标期号"])
            if int(row["训练结束期"]) >= target:
                raise RuntimeError(f"目标期{target}发生时间泄漏。")
            actual = json.loads(row["实际主号码"])
            v5 = np.asarray(json.loads(row["V5_45号码概率_JSON"]), dtype=float)
            v4 = np.asarray(json.loads(row["V4_45号码概率_JSON"]), dtype=float)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, target, 6303]))
            random_order = rng.permutation(np.arange(1, 46))
            v5m = rank_metrics(v5, actual, random_order)
            v4m = rank_metrics(v4, actual, random_order)
            rec = {
                "fold": fold_index,
                "目标期号": target,
                "训练结束期": int(row["训练结束期"]),
                "训练期数": int(row["训练期数"]),
                "V5_Brier": float(row["V5_Brier"]),
                "V4_Brier": float(row["V4_Brier"]),
                "随机_Brier": float(row["随机_Brier"]),
                "V5_LogLoss": float(row["V5_LogLoss"]),
                "V4_LogLoss": float(row["V4_LogLoss"]),
                "随机_LogLoss": float(row["随机_LogLoss"]),
            }
            for key, value in v5m.items():
                rec[f"V5_{key}"] = value
            for key, value in v4m.items():
                rec[f"V4_{key}"] = value
            records.append(rec)
        parts.append(pd.DataFrame(records))
        fold_meta.append({
            "fold": fold_index, "start_draw": start, "end_draw": end,
            "data_fingerprint_sha256": result["data_fingerprint_sha256"],
            "elapsed_seconds": result["elapsed_seconds"],
        })

    detail = pd.concat(parts, ignore_index=True)
    if len(detail) != FOLD_SIZE * len(FOLD_STARTS) or detail["目标期号"].nunique() != FOLD_SIZE * len(FOLD_STARTS):
        raise RuntimeError("V6.3目标期数量或唯一性校验失败。")
    if not (detail["训练结束期"] < detail["目标期号"]).all():
        raise RuntimeError("发现训练结束期不早于目标期。")

    summary_rows = []
    comparisons = []
    for model in ("V5", "V4"):
        for metric in METRICS:
            model_col = f"{model}_{metric}"
            random_col = f"random_{metric}"
            delta = detail[model_col].to_numpy(dtype=float) - detail[random_col].to_numpy(dtype=float)
            low, high = bootstrap_ci(delta, seed=SEED + len(summary_rows) * 17)
            summary_rows.append({
                "model": model, "metric": metric,
                "model_mean": float(detail[model_col].mean()),
                "random_mean": float(detail[random_col].mean()),
                "delta_vs_random": float(delta.mean()),
                "delta_95ci_low": low, "delta_95ci_high": high,
            })
    for metric in METRICS:
        delta = detail[f"V5_{metric}"].to_numpy(dtype=float) - detail[f"V4_{metric}"].to_numpy(dtype=float)
        low, high = bootstrap_ci(delta, seed=SEED + 900 + len(comparisons) * 17)
        comparisons.append({
            "comparison": "V5_minus_V4", "metric": metric,
            "delta": float(delta.mean()), "delta_95ci_low": low, "delta_95ci_high": high,
        })
    # Proper probability scores are retained separately; lower is better.
    for metric in ("Brier", "LogLoss"):
        for model in ("V5", "V4"):
            delta = detail[f"{model}_{metric}"].to_numpy(dtype=float) - detail[f"随机_{metric}"].to_numpy(dtype=float)
            low, high = bootstrap_ci(-delta, seed=SEED + 1200 + len(summary_rows) * 13)
            summary_rows.append({
                "model": model, "metric": metric,
                "model_mean": float(detail[f"{model}_{metric}"].mean()),
                "random_mean": float(detail[f"随机_{metric}"].mean()),
                "delta_vs_random": float(-delta.mean()),
                "delta_95ci_low": low, "delta_95ci_high": high,
                "interpretation": "positive_is_improvement_lower_score",
            })

    out = ROOT / "artifacts" / "v63"
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "v63_probability_ranking_detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(out / "v63_probability_ranking_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparisons).to_csv(out / "v63_v5_vs_v4_comparison.csv", index=False, encoding="utf-8-sig")
    fingerprint = hashlib.sha256(history.to_csv(index=False).encode("utf-8")).hexdigest()
    report = {
        "experiment": "V6.3 chronological probability-ranking utility audit",
        "status": "historical_development_screen_not_independent_or_prospective_evidence",
        "data_range": [DATA_FIRST_DRAW, DATA_LAST_DRAW],
        "target_folds": chronological_folds(),
        "target_draws": int(detail["目标期号"].nunique()),
        "training_window": TRAINING_WINDOW,
        "min_history": MIN_HISTORY,
        "ridge_alpha": RIDGE_ALPHA,
        "tickets_per_draw": TICKETS_PER_DRAW,
        "random_ranking": "uniform permutation per target draw, seed derived only from fixed seed and target draw",
        "summary": summary_rows,
        "v5_vs_v4": comparisons,
        "fold_runs": fold_meta,
        "data_fingerprint_sha256": fingerprint,
        "limitations": [
            "仅使用第1—1049期开发数据，目标期为850—1049，不是未接触的独立盲测。",
            "每个目标期只使用此前历史；脚本检查训练结束期严格早于目标期。",
            "top-k命中和NDCG评估号码排序，不等同于单注六号码组合的中奖概率。",
            "200个目标期是主要样本单位；45个号码/期不是独立开奖样本。",
            "历史排名优势不能证明未来存在可利用信号；不自动修改生产选号模型。",
            "随机排序基准按目标期固定种子生成，比较为逐期配对。",
        ],
    }
    (out / "v63_probability_ranking_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "experiment": report["experiment"], "status": report["status"],
        "target_draws": report["target_draws"], "summary": summary_rows,
        "v5_vs_v4": comparisons, "data_fingerprint_sha256": fingerprint,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
