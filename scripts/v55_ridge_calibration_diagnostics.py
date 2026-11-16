# -*- coding: utf-8 -*-
"""V5.5 Ridge sensitivity and probability-reliability diagnostics.

This is a retrospective diagnostic on the latest 50 draws in the manually
maintained repository CSV. Those draws may have appeared in earlier analyses;
the output is exploratory and is not an independent confirmation or a tuning
license. No external data is fetched.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v50_backtest import run_v50_walk_forward

RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
TEST_DRAWS = 50
TRAINING_WINDOW = 500
MIN_HISTORY = 180
TICKETS_PER_DRAW = 10
MASTER_SEED = 5526
PAIR_SAMPLING_STRENGTH = 0.5
CALIBRATION_BINS = 10


def probability_reliability(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> dict:
    """Equal-count reliability bins; descriptive only, with draw clustering retained in caveats."""
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    y = np.asarray(outcomes, dtype=float).reshape(-1)
    if p.shape != y.shape or p.size == 0 or not np.all(np.isfinite(p)) or not np.all(np.isfinite(y)):
        raise ValueError("概率和结果必须是同长度、非空、有限数组。")
    if not 1 <= int(bins) <= p.size:
        raise ValueError("分箱数量必须在1至观测数量之间。")
    if np.any((p <= 0) | (p >= 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError("概率必须位于(0,1)，结果必须为0/1。")
    bin_indices = np.array_split(np.argsort(p, kind="mergesort"), int(bins))
    rows = []
    weighted_abs_gap = 0.0
    max_abs_gap = 0.0
    for index, members in enumerate(bin_indices, 1):
        predicted = float(p[members].mean())
        observed = float(y[members].mean())
        gap = observed - predicted
        weighted_abs_gap += len(members) / p.size * abs(gap)
        max_abs_gap = max(max_abs_gap, abs(gap))
        rows.append({
            "bin": index,
            "count": int(len(members)),
            "mean_predicted_probability": predicted,
            "observed_frequency": observed,
            "observed_minus_predicted": gap,
        })
    return {
        "observations": int(p.size),
        "bins": rows,
        "equal_count_ECE": float(weighted_abs_gap),
        "maximum_absolute_bin_gap": float(max_abs_gap),
        "caveat": (
            "Descriptive equal-count bins over number-draw observations. "
            "The 45 outcomes within a draw are dependent; bins are not independent samples."
        ),
    }


def _extract_probability_arrays(detail: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probability_rows = []
    outcome_rows = []
    for _, row in detail.iterrows():
        probabilities = np.asarray(json.loads(row["V5_45号码概率_JSON"]), dtype=float)
        actual = np.asarray(json.loads(row["实际主号码"]), dtype=int)
        if probabilities.shape != (45,) or not np.isclose(probabilities.sum(), 6.0, atol=1e-5):
            raise ValueError(f"第{row['目标期号']}期概率向量异常。")
        outcome = np.isin(np.arange(1, 46), actual).astype(float)
        if int(outcome.sum()) != 6:
            raise ValueError(f"第{row['目标期号']}期实际号码数量异常。")
        probability_rows.append(probabilities)
        outcome_rows.append(outcome)
    return np.vstack(probability_rows), np.vstack(outcome_rows)


def run_diagnostics(history: pd.DataFrame) -> dict:
    history = history.sort_values("draw_no").reset_index(drop=True)
    if len(history) < MIN_HISTORY + TEST_DRAWS:
        raise ValueError("手动导入的历史数据不足以运行本次诊断。")
    if np.any(np.diff(history["draw_no"].to_numpy(dtype=int)) != 1):
        raise ValueError("历史期号不连续，拒绝运行时间顺序诊断。")

    cases = []
    for alpha in RIDGE_GRID:
        result = run_v50_walk_forward(
            history,
            test_draws=TEST_DRAWS,
            training_window=TRAINING_WINDOW,
            min_history=MIN_HISTORY,
            ridge_alpha=alpha,
            tickets_per_draw=TICKETS_PER_DRAW,
            seed=MASTER_SEED,
            bootstrap_reps=1000,
            pair_sampling_strength=PAIR_SAMPLING_STRENGTH,
        )
        probabilities, outcomes = _extract_probability_arrays(result["detail"])
        reliability = probability_reliability(
            probabilities.reshape(-1), outcomes.reshape(-1), CALIBRATION_BINS
        )
        summary = result["summary"].set_index("指标")
        mean_training_draws = float(result["detail"]["训练期数"].mean())
        cases.append({
            "ridge_alpha": alpha,
            "ridge_alpha_mean_loss_equivalent_mean": float(alpha / (mean_training_draws * 45.0)),
            "test_start_draw": result["test_start_draw"],
            "test_end_draw": result["test_end_draw"],
            "test_draws": result["test_draws"],
            "training_window": result["training_window"],
            "pair_sampling_strength": result["pair_sampling_strength"],
            "mean_hits": float(summary.loc["平均命中数", "V5平均"]),
            "batch_3plus_rate": float(summary.loc["整批至少一注3+率", "V5平均"]),
            "brier": float(summary.loc["概率Brier分数", "V5平均"]),
            "log_loss": float(summary.loc["概率Log Loss", "V5平均"]),
            "brier_vs_v4_delta": float(summary.loc["概率Brier分数", "V5_vs_V4差值"]),
            "log_loss_vs_v4_delta": float(summary.loc["概率Log Loss", "V5_vs_V4差值"]),
            "brier_vs_uniform_delta": float(
                summary.loc["概率Brier分数", "V5平均"] - summary.loc["概率Brier分数", "随机基准平均"]
            ),
            "log_loss_vs_uniform_delta": float(
                summary.loc["概率Log Loss", "V5平均"] - summary.loc["概率Log Loss", "随机基准平均"]
            ),
            "reliability": reliability,
            "data_fingerprint_sha256": result["data_fingerprint_sha256"],
            "evidence_status": result["evidence_status"],
        })

    return {
        "diagnostic_version": "v5.5-ridge-calibration-1",
        "purpose": "ridge sensitivity and descriptive probability reliability",
        "data_first_draw": int(history["draw_no"].iloc[0]),
        "data_last_draw": int(history["draw_no"].iloc[-1]),
        "test_draws": TEST_DRAWS,
        "training_window": TRAINING_WINDOW,
        "ridge_grid": list(RIDGE_GRID),
        "master_seed": MASTER_SEED,
        "tickets_per_draw": TICKETS_PER_DRAW,
        "pair_sampling_strength": PAIR_SAMPLING_STRENGTH,
        "cases": cases,
        "selection_policy": (
            "No best alpha is selected from this report. All candidate settings are compared "
            "on the same fixed retrospective interval; this interval has prior project use."
        ),
        "limitations": [
            "Retrospective interval may overlap earlier V5.2/V5.3 analyses; results are not independent confirmation.",
            "Equal-count reliability bins are descriptive and number outcomes within each draw are dependent.",
            "The six-inclusion total is constrained to exactly six in expectation, so aggregate mean predicted probability is fixed at 6/45.",
            "No prospective predictive advantage is established by this diagnostic.",
        ],
    }


def main() -> None:
    history = pd.read_csv(ROOT / "data" / "lotto645.csv")
    report = run_diagnostics(history)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
