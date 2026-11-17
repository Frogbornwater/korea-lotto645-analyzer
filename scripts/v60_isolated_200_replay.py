# -*- coding: utf-8 -*-
"""V6.0 isolated chronological replay for the latest 200 historical draws.

This is a deterministic historical simulation, not fresh prospective evidence.
Every target prediction is generated from draws strictly earlier than its target.
No network access or official lottery integration is used.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lotto_core import validate_frame
from v50_backtest import run_v50_walk_forward, _bootstrap_mean_ci

# Frozen before this replay. Do not tune these values against the 200-draw results.
HOLDOUT_DRAWS = 200
SENSITIVITY_WINDOWS = (100, 150, 200)
TRAINING_WINDOW = 500
MIN_HISTORY = 180
RIDGE_ALPHA = 1.0
TICKETS_PER_DRAW = 10
SEED = 6025
PAIR_SAMPLING_STRENGTH = 0.5
EXPECTED_FIRST_DRAW = 1
EXPECTED_LAST_DRAW = 1249


def isolated_window_bounds(last_draw: int, test_draws: int = HOLDOUT_DRAWS) -> tuple[int, int]:
    """Return inclusive target draw bounds for a chronological tail replay."""
    last_draw = int(last_draw)
    test_draws = int(test_draws)
    if test_draws < 1 or last_draw < test_draws:
        raise ValueError("测试期数必须为正数且不能超过历史末期号。")
    return last_draw - test_draws + 1, last_draw


def build_window_sensitivity(detail: pd.DataFrame, windows=SENSITIVITY_WINDOWS) -> pd.DataFrame:
    """Summarize nested tail windows; shorter windows are diagnostics, not independent tests."""
    if detail.empty or "目标期号" not in detail:
        raise ValueError("逐期回放明细为空或缺少目标期号。")
    last_draw = int(detail["目标期号"].max())
    metrics = (
        ("mean_hits", "每注平均命中数", False),
        ("ticket_3plus", "单注3+命中率", False),
        ("ticket_4plus", "单注4+命中率", False),
        ("batch_3plus", "每期10注至少一注3+率", False),
        ("batch_4plus", "每期10注至少一注4+率", False),
        ("Brier", "Brier分数", True),
        ("LogLoss", "Log Loss", True),
    )
    rows = []
    for window in windows:
        window = int(window)
        start_draw, end_draw = isolated_window_bounds(last_draw, window)
        subset = detail.loc[detail["目标期号"].between(start_draw, end_draw)].copy()
        if len(subset) != window:
            raise ValueError(f"最近{window}期数据不完整：实际{len(subset)}期。")
        for key, label, lower_is_better in metrics:
            v5 = float(subset[f"V5_{key}"].mean())
            v4 = float(subset[f"V4_{key}"].mean())
            random = float(subset[f"随机_{key}"].mean())
            # Positive means V5 is better for all reported metrics.
            delta_random = (random - v5) if lower_is_better else (v5 - random)
            delta_v4 = (v4 - v5) if lower_is_better else (v5 - v4)
            source = (random - subset[f"V5_{key}"].to_numpy()) if lower_is_better else (subset[f"V5_{key}"].to_numpy() - subset[f"随机_{key}"].to_numpy())
            ci_low, ci_high = _bootstrap_mean_ci(source, 3000, SEED + window + len(rows))
            rows.append({
                "窗口期数": window,
                "起始期": start_draw,
                "结束期": end_draw,
                "指标": label,
                "V5": v5,
                "V4冻结模型": v4,
                "随机基准": random,
                "V5相对随机优势_正值较好": delta_random,
                "V5相对随机优势95%CI下限": ci_low,
                "V5相对随机优势95%CI上限": ci_high,
                "V5相对V4优势_正值较好": delta_v4,
            })
    return pd.DataFrame(rows)


def main() -> None:
    data_path = ROOT / "data" / "lotto645.csv"
    output_dir = ROOT / "artifacts" / "v60"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(data_path)
    history = validate_frame(raw).reset_index(drop=True)
    draws = history["draw_no"].astype(int)
    if int(draws.iloc[0]) != EXPECTED_FIRST_DRAW or int(draws.iloc[-1]) != EXPECTED_LAST_DRAW:
        raise ValueError(
            f"本实验锁定的数据范围为第{EXPECTED_FIRST_DRAW}—{EXPECTED_LAST_DRAW}期；"
            f"当前文件为第{int(draws.iloc[0])}—{int(draws.iloc[-1])}期。"
            "请先人工核验并更新实验版本，不要静默改变测试窗口。"
        )
    if len(history) != EXPECTED_LAST_DRAW - EXPECTED_FIRST_DRAW + 1:
        raise ValueError("历史数据期数与期号范围不一致，拒绝执行。")

    result = run_v50_walk_forward(
        history,
        test_draws=HOLDOUT_DRAWS,
        training_window=TRAINING_WINDOW,
        min_history=MIN_HISTORY,
        ridge_alpha=RIDGE_ALPHA,
        tickets_per_draw=TICKETS_PER_DRAW,
        seed=SEED,
        bootstrap_reps=3000,
        pair_sampling_strength=PAIR_SAMPLING_STRENGTH,
    )
    expected_start, expected_end = isolated_window_bounds(EXPECTED_LAST_DRAW)
    if (result["test_start_draw"], result["test_end_draw"], result["test_draws"]) != (
        expected_start, expected_end, HOLDOUT_DRAWS
    ):
        raise RuntimeError("回放引擎返回的测试区间与预注册区间不一致。")

    detail = result["detail"]
    sensitivity = build_window_sensitivity(detail)
    detail.to_csv(output_dir / "v60_200_draw_detail.csv", index=False, encoding="utf-8-sig")
    result["summary"].to_csv(output_dir / "v60_200_draw_summary.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(output_dir / "v60_window_sensitivity.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "experiment": "V6.0 isolated chronological 200-draw replay",
        "status": "historical_replay_not_fresh_prospective_evidence",
        "data_first_draw": EXPECTED_FIRST_DRAW,
        "data_last_draw": EXPECTED_LAST_DRAW,
        "test_start_draw": expected_start,
        "test_end_draw": expected_end,
        "test_draws": HOLDOUT_DRAWS,
        "sensitivity_windows": list(SENSITIVITY_WINDOWS),
        "training_window": TRAINING_WINDOW,
        "min_history": MIN_HISTORY,
        "ridge_alpha": RIDGE_ALPHA,
        "pair_sampling_strength": PAIR_SAMPLING_STRENGTH,
        "tickets_per_draw_per_strategy": TICKETS_PER_DRAW,
        "seed": SEED,
        "engine_version": result["engine_version"],
        "data_fingerprint_sha256": result["data_fingerprint_sha256"],
        "elapsed_seconds": result["elapsed_seconds"],
        "limitations": [
            "每期预测仅使用该期之前的历史数据；目标期及之后的开奖结果不进入该期训练。",
            "参数在本轮回放中固定，不依据测试区间结果调参。",
            "1050—1249期曾在此前项目研究中被查看或分析，因此本结果不是从未接触过的独立盲测。",
            "100期和150期是200期主窗口的嵌套尾部诊断，不是独立样本。",
            "历史模拟不能证明未来开奖存在可利用的预测优势。"
        ],
        "summary": result["summary"].to_dict(orient="records"),
        "window_sensitivity": sensitivity.to_dict(orient="records"),
    }
    (output_dir / "v60_200_draw_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "experiment": metadata["experiment"],
        "test_start_draw": expected_start,
        "test_end_draw": expected_end,
        "test_draws": HOLDOUT_DRAWS,
        "elapsed_seconds": round(result["elapsed_seconds"], 2),
        "data_fingerprint_sha256": result["data_fingerprint_sha256"],
        "summary": metadata["summary"],
        "window_sensitivity": metadata["window_sensitivity"],
        "limitations": metadata["limitations"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
