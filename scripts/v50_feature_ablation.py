# -*- coding: utf-8 -*-
"""V5 feature ablation screen on the protected retrospective interval."""
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v50_backtest import run_v50_walk_forward

history = pd.read_csv(ROOT / "data" / "lotto645.csv")
history = history.loc[history["draw_no"] <= 1088].copy().reset_index(drop=True)
ablations = {
    "full_model": (),
    "frequency_only": (
        "trend_30_minus_120", "ewma_35_z", "gap_log_scaled",
        "pair_mean_log_lift", "pair_sd_log_lift",
    ),
    "without_pair": ("pair_mean_log_lift", "pair_sd_log_lift"),
    "without_gap": ("gap_log_scaled",),
    "without_ewma": ("ewma_35_z",),
}
reports = []
for name, exclusions in ablations.items():
    result = run_v50_walk_forward(
        history,
        test_draws=50,
        training_window=500,
        min_history=180,
        ridge_alpha=1.0,
        tickets_per_draw=10,
        seed=5026,
        bootstrap_reps=1000,
        feature_exclusions=exclusions,
    )
    summary = result["summary"].set_index("指标")
    reports.append({
        "variant": name,
        "excluded_features": list(exclusions),
        "mean_hits_v5": float(summary.loc["平均命中数", "V5平均"]),
        "mean_hits_v4": float(summary.loc["平均命中数", "V4冻结模型平均"]),
        "mean_hits_random": float(summary.loc["平均命中数", "随机基准平均"]),
        "batch_3plus_v5": float(summary.loc["整批至少一注3+率", "V5平均"]),
        "batch_3plus_v4": float(summary.loc["整批至少一注3+率", "V4冻结模型平均"]),
        "batch_3plus_random": float(summary.loc["整批至少一注3+率", "随机基准平均"]),
        "brier_v5": float(summary.loc["概率Brier分数", "V5平均"]),
        "brier_random": float(summary.loc["概率Brier分数", "随机基准平均"]),
    })
print(json.dumps({
    "warning": "Retrospective ablation only; not independent or prospective evidence.",
    "test_range": [1039, 1088],
    "variants": reports,
}, ensure_ascii=False, indent=2))
