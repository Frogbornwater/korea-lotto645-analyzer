# -*- coding: utf-8 -*-
"""Reproducible V5.0 real-history screening benchmark (retrospective only)."""
from pathlib import Path
import json

import pandas as pd

from v50_backtest import run_v50_walk_forward

ROOT = Path(__file__).resolve().parents[1]
history = pd.read_csv(ROOT / "data" / "lotto645.csv")
history = history.loc[history["draw_no"] <= 1088].copy().reset_index(drop=True)
result = run_v50_walk_forward(
    history,
    test_draws=50,
    training_window=500,
    min_history=180,
    ridge_alpha=1.0,
    tickets_per_draw=10,
    seed=5026,
    bootstrap_reps=1000,
)
print(json.dumps({
    "engine_version": result["engine_version"],
    "test_start_draw": result["test_start_draw"],
    "test_end_draw": result["test_end_draw"],
    "test_draws": result["test_draws"],
    "elapsed_seconds": round(result["elapsed_seconds"], 3),
    "data_fingerprint_sha256": result["data_fingerprint_sha256"],
    "evidence_status": result["evidence_status"],
    "summary": result["summary"].to_dict(orient="records"),
}, ensure_ascii=False, indent=2))
