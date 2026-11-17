# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lotto_core import quality_report, validate_frame
from v83_feature_audit import evaluate_v83_feature_audit

HISTORY_PATH = ROOT / "data" / "lotto645.csv"
OUTPUT_DIR = ROOT / "artifacts" / "v83"


def main() -> None:
    raw = pd.read_csv(HISTORY_PATH)
    history = validate_frame(raw).reset_index(drop=True)
    quality = quality_report(raw)
    if history.empty or int(history["draw_no"].iloc[0]) != 1:
        raise ValueError("V8.3研究要求从第1期开始的完整手动历史数据。")
    if not quality.get("sequence_complete", False):
        raise ValueError("V8.3研究拒绝存在缺期的手动历史数据。")
    if int(history["draw_no"].iloc[-1]) < 500:
        raise ValueError("V8.3研究至少需要500期历史数据。")

    report, detail = evaluate_v83_feature_audit(
        history,
        test_draws=500,
        training_window=500,
        min_history=180,
        ridge_alpha=1.0,
        bootstrap_repetitions=10000,
        block_length=10,
        seed=8302028,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "v83_feature_audit_report.json"
    detail_path = OUTPUT_DIR / "v83_feature_audit_detail.csv"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"V8.3 report: {report_path}")
    print(f"V8.3 detail: {detail_path}")
    print(f"Manual history: {history['draw_no'].iloc[0]}-{history['draw_no'].iloc[-1]}")


if __name__ == "__main__":
    main()
