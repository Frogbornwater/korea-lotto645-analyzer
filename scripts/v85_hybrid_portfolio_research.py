# -*- coding: utf-8 -*-
"""Run the V8.5 500-draw prequential comparison on manually maintained history."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lotto_core import quality_report, validate_frame
from v85_hybrid_backtest import run_v85_walk_forward, write_v85_artifacts

HISTORY_PATH = ROOT / "data" / "lotto645.csv"
OUTPUT_DIR = ROOT / "artifacts" / "v85"


def main() -> None:
    if not HISTORY_PATH.exists():
        raise FileNotFoundError("缺少手动维护的data/lotto645.csv；不会联网获取开奖数据。")
    raw = pd.read_csv(HISTORY_PATH)
    history = validate_frame(raw).reset_index(drop=True)
    quality = quality_report(raw)
    if history.empty or int(history["draw_no"].iloc[0]) != 1:
        raise ValueError("V8.5要求从第1期开始的完整手动历史数据。")
    if not quality.get("sequence_complete", False):
        raise ValueError("V8.5拒绝期号缺失的手动历史数据。")
    if int(history["draw_no"].iloc[-1]) < 680:
        raise ValueError("V8.5至少需要680期连续历史数据，才能进行500期逐期回放。")

    result = run_v85_walk_forward(
        history,
        test_draws=500,
        training_window=500,
        min_history=180,
        ridge_alpha=1.0,
        tickets_per_draw=10,
        seed=85_2026,
        bootstrap_repetitions=10_000,
        block_length=10,
    )
    report_path, detail_path = write_v85_artifacts(result, str(OUTPUT_DIR))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print(f"V8.5 report: {report_path}")
    print(f"V8.5 detail: {detail_path}")
    print(
        f"Manual history: {history['draw_no'].iloc[0]}-{history['draw_no'].iloc[-1]}, "
        f"targets: {result['report']['target_start_draw']}-{result['report']['target_end_draw']}"
    )


if __name__ == "__main__":
    main()
