# -*- coding: utf-8 -*-
"""V6.1 nested chronological feature ablation on development draws 1-1049.

All folds are strictly chronological and disjoint in their target draws.
This is historical development evidence, not an untouched prospective test.
No external data or network access is used.
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
from v50_backtest import run_v50_walk_forward, _bootstrap_mean_ci

DATA_FIRST_DRAW = 1
DATA_LAST_DRAW = 1049
FOLD_SIZE = 50
FOLD_STARTS = (850, 900, 950, 1000)
TRAINING_WINDOW = 500
MIN_HISTORY = 180
RIDGE_ALPHA = 1.0
TICKETS_PER_DRAW = 10
MASTER_SEED = 61026
PAIR_STRENGTH = 0.5
BOOTSTRAP_REPS = 5000

# Fixed ablations; no parameter or variant is selected from the evaluation folds.
VARIANTS = {
    "full_model": {"feature_exclusions": (), "pair_sampling_strength": PAIR_STRENGTH},
    "frequency_only": {
        "feature_exclusions": (
            "trend_30_minus_120", "ewma_35_z", "gap_log_scaled",
            "pair_mean_log_lift", "pair_sd_log_lift",
        ),
        "pair_sampling_strength": 0.0,
    },
    "without_pair_features": {
        "feature_exclusions": ("pair_mean_log_lift", "pair_sd_log_lift"),
        "pair_sampling_strength": PAIR_STRENGTH,
    },
    "without_pair_sampling": {
        "feature_exclusions": (),
        "pair_sampling_strength": 0.0,
    },
    "without_gap": {
        "feature_exclusions": ("gap_log_scaled",),
        "pair_sampling_strength": PAIR_STRENGTH,
    },
    "without_ewma": {
        "feature_exclusions": ("ewma_35_z",),
        "pair_sampling_strength": PAIR_STRENGTH,
    },
    "without_trend": {
        "feature_exclusions": ("trend_30_minus_120",),
        "pair_sampling_strength": PAIR_STRENGTH,
    },
}

METRICS = (
    ("mean_hits", "V5_mean_hits", "随机_mean_hits", False),
    ("batch_3plus", "V5_batch_3plus", "随机_batch_3plus", False),
    ("Brier", "V5_Brier", "随机_Brier", True),
    ("LogLoss", "V5_LogLoss", "随机_LogLoss", True),
)


def chronological_folds(starts=FOLD_STARTS, size=FOLD_SIZE):
    """Return non-overlapping inclusive target intervals."""
    starts = tuple(int(x) for x in starts)
    size = int(size)
    if size < 1 or not starts:
        raise ValueError("折数和每折期数必须为正数。")
    if tuple(sorted(starts)) != starts:
        raise ValueError("时间折必须按起始期升序排列。")
    folds = [(start, start + size - 1) for start in starts]
    for previous, current in zip(folds, folds[1:]):
        if current[0] <= previous[1]:
            raise ValueError("各折测试期必须互不重叠。")
    return folds


def paired_bootstrap_ci(values, repetitions=BOOTSTRAP_REPS, seed=MASTER_SEED):
    values = np.asarray(values, dtype=float)
    return _bootstrap_mean_ci(values, int(repetitions), int(seed))


def summarize_variant(detail: pd.DataFrame, variant: str) -> dict:
    rows = {"variant": variant, "test_draws": int(len(detail))}
    for offset, (label, model_col, random_col, lower_is_better) in enumerate(METRICS):
        model = detail[model_col].to_numpy(dtype=float)
        random = detail[random_col].to_numpy(dtype=float)
        delta = random - model if lower_is_better else model - random
        low, high = paired_bootstrap_ci(delta, seed=MASTER_SEED + offset)
        rows[label] = float(model.mean())
        rows[f"{label}_random"] = float(random.mean())
        rows[f"{label}_improvement_vs_random"] = float(delta.mean())
        rows[f"{label}_improvement_95ci_low"] = low
        rows[f"{label}_improvement_95ci_high"] = high
        rows[f"{label}_fold_direction_count"] = int(
            sum(
                (random_fold - model_fold if lower_is_better else model_fold - random_fold) > 0
                for _, fold in detail.groupby("fold", sort=True)
                for model_fold, random_fold in [(fold[model_col].mean(), fold[random_col].mean())]
            )
        )
    return rows


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "lotto645.csv")
    history = validate_frame(raw).reset_index(drop=True)
    draws = history["draw_no"].to_numpy(dtype=int)
    if len(history) != DATA_LAST_DRAW - DATA_FIRST_DRAW + 1:
        raise ValueError("V6.1 锁定数据必须恰好包含第1—1049期。")
    if int(draws[0]) != DATA_FIRST_DRAW or int(draws[-1]) != DATA_LAST_DRAW:
        raise ValueError("V6.1 数据起止期与预注册范围不一致。")
    if np.any(np.diff(draws) != 1):
        raise ValueError("V6.1 要求历史期号连续，拒绝执行。")

    detail_parts = []
    fold_rows = []
    for fold_index, (start_draw, end_draw) in enumerate(chronological_folds(), 1):
        fold_history = history.loc[history["draw_no"] <= end_draw].copy().reset_index(drop=True)
        if int(fold_history["draw_no"].iloc[-1]) != end_draw:
            raise RuntimeError("折末期数据缺失。")
        for variant_index, (variant, config) in enumerate(VARIANTS.items()):
            result = run_v50_walk_forward(
                fold_history,
                test_draws=FOLD_SIZE,
                training_window=TRAINING_WINDOW,
                min_history=MIN_HISTORY,
                ridge_alpha=RIDGE_ALPHA,
                tickets_per_draw=TICKETS_PER_DRAW,
                seed=MASTER_SEED + fold_index * 100,  # common random numbers across ablations
                bootstrap_reps=1000,
                feature_exclusions=config["feature_exclusions"],
                pair_sampling_strength=config["pair_sampling_strength"],
            )
            if (result["test_start_draw"], result["test_end_draw"], result["test_draws"]) != (
                start_draw, end_draw, FOLD_SIZE
            ):
                raise RuntimeError(f"{variant} 第{fold_index}折测试区间不匹配。")
            part = result["detail"].copy()
            part["fold"] = fold_index
            part["variant"] = variant
            detail_parts.append(part)
            fold_rows.append({
                "fold": fold_index,
                "start_draw": start_draw,
                "end_draw": end_draw,
                "variant": variant,
                "data_fingerprint_sha256": result["data_fingerprint_sha256"],
                "elapsed_seconds": result["elapsed_seconds"],
            })

    detail = pd.concat(detail_parts, ignore_index=True)
    summaries = [
        summarize_variant(detail.loc[detail["variant"] == variant].copy(), variant)
        for variant in VARIANTS
    ]
    summary = pd.DataFrame(summaries)
    output_dir = ROOT / "artifacts" / "v61"
    output_dir.mkdir(parents=True, exist_ok=True)
    keep = [
        "目标期号", "训练结束期", "训练期数", "fold", "variant",
        "V5_mean_hits", "随机_mean_hits", "V5_batch_3plus", "随机_batch_3plus",
        "V5_Brier", "随机_Brier", "V5_LogLoss", "随机_LogLoss",
        "V5_45号码概率_JSON", "实际主号码",
    ]
    detail[keep].to_csv(output_dir / "v61_nested_ablation_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "v61_nested_ablation_summary.csv", index=False, encoding="utf-8-sig")
    fingerprint = hashlib.sha256(history.to_csv(index=False).encode("utf-8")).hexdigest()
    report = {
        "experiment": "V6.1 nested chronological feature ablation",
        "status": "historical_development_screen_not_independent_or_prospective_evidence",
        "data_range": [DATA_FIRST_DRAW, DATA_LAST_DRAW],
        "target_folds": chronological_folds(),
        "total_target_draws_per_variant": FOLD_SIZE * len(FOLD_STARTS),
        "training_window": TRAINING_WINDOW,
        "min_history": MIN_HISTORY,
        "ridge_alpha": RIDGE_ALPHA,
        "tickets_per_draw": TICKETS_PER_DRAW,
        "master_seed": MASTER_SEED,
        "pair_sampling_strength_full": PAIR_STRENGTH,
        "variants": VARIANTS,
        "data_fingerprint_sha256": fingerprint,
        "summary": summaries,
        "fold_runs": fold_rows,
        "limitations": [
            "第850—1049期仅用于历史开发区间的滚动筛查，不是未接触的独立盲测。",
            "四个50期测试折按时间先后排列且目标期不重叠；每个目标期只使用此前历史。",
            "训练窗口、Ridge参数和候选消融组均预先固定；本实验不自动挑选最佳模型或参数。",
            "各折样本量有限，号码结果存在同期开奖内依赖；区间估计仅作探索性描述。",
            "任何历史消融优势都不能证明未来开奖具有可利用的预测信号。",
            "V6.0的1050—1249期结果不用于本轮特征选择；未来独立验证仍需开奖前锁定预测。"
        ],
    }
    (output_dir / "v61_nested_ablation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "experiment": report["experiment"],
        "status": report["status"],
        "target_folds": report["target_folds"],
        "summary": summaries,
        "data_fingerprint_sha256": fingerprint,
        "limitations": report["limitations"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
