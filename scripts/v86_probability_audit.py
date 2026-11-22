# -*- coding: utf-8 -*-
"""Run V8.6 probability calibration and ranking audit on manual history only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lotto_core import quality_report, validate_frame
from v86_probability_audit import run_v86_probability_audit, write_v86_artifacts

HISTORY_PATH = ROOT / "data" / "lotto645.csv"
OUTPUT_DIR = ROOT / "artifacts" / "v86"


def main() -> None:
    if not HISTORY_PATH.exists():
        raise FileNotFoundError("缺少手动维护的data/lotto645.csv；不会联网获取开奖数据。")
    raw = pd.read_csv(HISTORY_PATH)
    history = validate_frame(raw).reset_index(drop=True)
    quality = quality_report(raw)
    if history.empty or int(history["draw_no"].iloc[0]) != 1:
        raise ValueError("V8.6要求从第1期开始的完整手动历史数据。")
    if not quality.get("sequence_complete", False):
        raise ValueError("V8.6拒绝期号缺失的手动历史数据。")
    if int(history["draw_no"].iloc[-1]) < 752:
        raise ValueError("V8.6需要至少752期连续手动历史数据。")

    result = run_v86_probability_audit(
        history,
        test_draws=500,
        warmup_draws=200,
        calibration_window=200,
        training_window=500,
        min_history=180,
        ridge_alpha=1.0,
        bootstrap_repetitions=10_000,
        block_length=10,
        seed=86_2026,
    )
    paths = write_v86_artifacts(result, str(OUTPUT_DIR))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print(f"V8.6 report: {paths[0]}")
    print(f"V8.6 detail: {paths[1]}")
    print(f"V8.6 reliability: {paths[2]}")


if __name__ == "__main__":
    main()
