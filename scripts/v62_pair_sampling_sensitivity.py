# -*- coding: utf-8 -*-
"""V6.2 pair-sampling strength sensitivity on development draws 1-1049.

This is a predeclared historical development screen, not independent or prospective evidence.
All target draws are predicted strictly from earlier draws. No network access or external data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lotto_core import NUMBER_COLS, validate_frame
from v50_backtest import run_v50_walk_forward

DATA_LAST_DRAW = 1049
FOLD_SIZE = 50
FOLD_STARTS = (850, 900, 950, 1000)
TRAINING_WINDOW = 500
MIN_HISTORY = 180
RIDGE_ALPHA = 1.0
TICKETS_PER_DRAW = 10
PAIR_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (61026, 61027, 61028, 61029, 61030)


def validate_design() -> None:
    if len(set(PAIR_STRENGTHS)) != len(PAIR_STRENGTHS):
        raise ValueError("pair strength 网格存在重复值。")
    if any(not 0.0 <= x <= 2.0 for x in PAIR_STRENGTHS):
        raise ValueError("pair strength 超出允许范围。")
    if len(set(SEEDS)) != len(SEEDS):
        raise ValueError("随机种子必须互不相同。")
    folds = [(s, s + FOLD_SIZE - 1) for s in FOLD_STARTS]
    if any(b[0] <= a[1] for a, b in zip(folds, folds[1:])):
        raise ValueError("目标期折必须互不重叠。")


def main() -> None:
    validate_design()
    raw = pd.read_csv(ROOT / "data" / "lotto645.csv")
    if "draw_no" not in raw.columns:
        raise ValueError("数据文件缺少 draw_no 字段。")
    raw = raw.loc[pd.to_numeric(raw["draw_no"], errors="coerce") <= DATA_LAST_DRAW].copy()
    history = validate_frame(raw).reset_index(drop=True)
    draws = history["draw_no"].to_numpy(dtype=int)
    if len(history) != DATA_LAST_DRAW or int(draws[0]) != 1 or int(draws[-1]) != DATA_LAST_DRAW:
        raise ValueError("V6.2 要求恰好包含第1—1049期。")
    if np.any(np.diff(draws) != 1):
        raise ValueError("历史期号不连续，拒绝运行。")

    detail_parts = []
    fold_runs = []
    for fold_index, start in enumerate(FOLD_STARTS, 1):
        end = start + FOLD_SIZE - 1
        fold_history = history.loc[history["draw_no"] <= end].copy().reset_index(drop=True)
        for seed in SEEDS:
            for strength in PAIR_STRENGTHS:
                started = time.perf_counter()
                result = run_v50_walk_forward(
                    fold_history,
                    test_draws=FOLD_SIZE,
                    training_window=TRAINING_WINDOW,
                    min_history=MIN_HISTORY,
                    ridge_alpha=RIDGE_ALPHA,
                    tickets_per_draw=TICKETS_PER_DRAW,
                    seed=seed,
                    bootstrap_reps=0,
                    pair_sampling_strength=strength,
                )
                if (result["test_start_draw"], result["test_end_draw"], result["test_draws"]) != (start, end, FOLD_SIZE):
                    raise RuntimeError(f"seed={seed}, strength={strength} 测试区间不匹配。")
                part = result["detail"].copy()
                part["fold"] = fold_index
                part["seed"] = seed
                part["pair_sampling_strength"] = strength
                detail_parts.append(part)
                fold_runs.append({
                    "fold": fold_index, "start_draw": start, "end_draw": end,
                    "seed": seed, "pair_sampling_strength": strength,
                    "elapsed_seconds": time.perf_counter() - started,
                    "data_fingerprint_sha256": result["data_fingerprint_sha256"],
                })

    detail = pd.concat(detail_parts, ignore_index=True)
    expected = len(FOLD_STARTS) * len(SEEDS) * len(PAIR_STRENGTHS) * FOLD_SIZE
    if len(detail) != expected:
        raise RuntimeError(f"结果行数错误：{len(detail)} != {expected}")
    # Each seed/target has a common random baseline across pair strengths.
    random_check = detail.groupby(["fold", "seed", "目标期号"])[
        ["随机_mean_hits", "随机_batch_3plus", "随机_Brier", "随机_LogLoss"]
    ].nunique()
    if (random_check > 1).any().any():
        raise RuntimeError("共同随机基准不一致，配对比较无效。")

    rows = []
    for strength, group in detail.groupby("pair_sampling_strength", sort=True):
        rows.append({
            "pair_sampling_strength": float(strength),
            "runs": int(group[["fold", "seed"]].drop_duplicates().shape[0]),
            "target_predictions": int(len(group)),
            "mean_hits": float(group["V5_mean_hits"].mean()),
            "random_mean_hits": float(group["随机_mean_hits"].mean()),
            "mean_hits_delta_vs_random": float((group["V5_mean_hits"] - group["随机_mean_hits"]).mean()),
            "batch_3plus": float(group["V5_batch_3plus"].mean()),
            "random_batch_3plus": float(group["随机_batch_3plus"].mean()),
            "batch_3plus_delta_vs_random": float((group["V5_batch_3plus"] - group["随机_batch_3plus"]).mean()),
            "Brier": float(group["V5_Brier"].mean()),
            "random_Brier": float(group["随机_Brier"].mean()),
            "LogLoss": float(group["V5_LogLoss"].mean()),
            "random_LogLoss": float(group["随机_LogLoss"].mean()),
        })
    summary = pd.DataFrame(rows)
    # Paired, seed-matched comparison to strength=0.0; bootstrap target-round clusters.
    zero = detail.loc[detail["pair_sampling_strength"] == 0.0,
                      ["fold", "seed", "目标期号", "V5_mean_hits", "V5_batch_3plus"]]
    paired = detail.merge(zero, on=["fold", "seed", "目标期号"], suffixes=("", "_zero"), validate="many_to_one")
    rng = np.random.default_rng(62026)
    target_keys = paired[["fold", "目标期号"]].drop_duplicates().to_records(index=False)
    ci_rows = []
    for strength, group in paired.groupby("pair_sampling_strength", sort=True):
        if float(strength) == 0.0:
            hit_delta = np.zeros(len(target_keys))
            batch_delta = np.zeros(len(target_keys))
        else:
            per_target = group.assign(
                hit_delta=group["V5_mean_hits"] - group["V5_mean_hits_zero"],
                batch_delta=group["V5_batch_3plus"] - group["V5_batch_3plus_zero"],
            ).groupby(["fold", "目标期号"])[["hit_delta", "batch_delta"]].mean()
            hit_delta = per_target["hit_delta"].to_numpy()
            batch_delta = per_target["batch_delta"].to_numpy()
        samples = rng.integers(0, len(hit_delta), size=(4000, len(hit_delta)))
        hit_boot = hit_delta[samples].mean(axis=1)
        batch_boot = batch_delta[samples].mean(axis=1)
        ci_rows.append({
            "pair_sampling_strength": float(strength),
            "mean_hits_delta_vs_strength_0": float(hit_delta.mean()),
            "mean_hits_delta_95ci_low": float(np.quantile(hit_boot, 0.025)),
            "mean_hits_delta_95ci_high": float(np.quantile(hit_boot, 0.975)),
            "batch_3plus_delta_vs_strength_0": float(batch_delta.mean()),
            "batch_3plus_delta_95ci_low": float(np.quantile(batch_boot, 0.025)),
            "batch_3plus_delta_95ci_high": float(np.quantile(batch_boot, 0.975)),
        })
    summary = summary.merge(pd.DataFrame(ci_rows), on="pair_sampling_strength", validate="one_to_one")

    out = ROOT / "artifacts" / "v62"
    out.mkdir(parents=True, exist_ok=True)
    keep = [
        "目标期号", "训练结束期", "训练期数", "fold", "seed", "pair_sampling_strength",
        "V5_mean_hits", "随机_mean_hits", "V5_batch_3plus", "随机_batch_3plus",
        "V5_Brier", "随机_Brier", "V5_LogLoss", "随机_LogLoss",
    ]
    detail[keep].to_csv(out / "v62_pair_sampling_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "v62_pair_sampling_summary.csv", index=False, encoding="utf-8-sig")
    fingerprint = hashlib.sha256(history.to_csv(index=False).encode("utf-8")).hexdigest()
    report = {
        "experiment": "V6.2 pair-sampling strength sensitivity",
        "status": "historical_development_screen_not_independent_or_prospective_evidence",
        "data_range": [1, DATA_LAST_DRAW],
        "target_folds": [[s, s + FOLD_SIZE - 1] for s in FOLD_STARTS],
        "pair_sampling_strengths": list(PAIR_STRENGTHS),
        "seeds": list(SEEDS),
        "training_window": TRAINING_WINDOW,
        "min_history": MIN_HISTORY,
        "ridge_alpha": RIDGE_ALPHA,
        "tickets_per_draw": TICKETS_PER_DRAW,
        "total_predictions_per_strength": len(SEEDS) * len(FOLD_STARTS) * FOLD_SIZE,
        "data_fingerprint_sha256": fingerprint,
        "summary": summary.to_dict(orient="records"),
        "fold_runs": fold_runs,
        "limitations": [
            "只使用第1—1049期历史开发数据，不能视为独立盲测或前瞻证据。",
            "每个随机种子和目标期在不同强度间配对；随机基准保持一致。",
            "区间按目标期聚类重采样，仍属于探索性估计，不能替代未来开奖前锁定验证。",
            "号码概率模型未随 pair sampling strength 改变；该参数只影响候选组合构造。",
            "不得根据本实验结果直接宣称单注中奖概率提高，也不自动修改正式选号默认值。"
        ],
    }
    (out / "v62_pair_sampling_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"experiment": report["experiment"], "status": report["status"],
                      "summary": report["summary"], "data_fingerprint_sha256": fingerprint},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
