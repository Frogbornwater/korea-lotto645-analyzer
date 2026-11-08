# -*- coding: utf-8 -*-
"""V5.0 temporal robustness and label-permutation placebo diagnostics.

All runs are retrospective diagnostics on the already protected 1039-1088 window.
They must never be described as unseen or prospective evidence.
"""
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v50_backtest import run_v50_walk_forward

history = pd.read_csv(ROOT / "data" / "lotto645.csv")
history = history.loc[history["draw_no"] <= 1088].copy().reset_index(drop=True)

def run_case(name, window, alpha, placebo=None):
    result = run_v50_walk_forward(
        history,
        test_draws=50,
        training_window=window,
        min_history=180,
        ridge_alpha=alpha,
        tickets_per_draw=10,
        seed=5026,
        bootstrap_reps=1000,
        placebo_seed=placebo,
    )
    summary = result["summary"].set_index("指标")
    mean = summary.loc["平均命中数"]
    batch = summary.loc["整批至少一注3+率"]
    brier = summary.loc["概率Brier分数"]
    return {
        "case": name,
        "training_window": window,
        "ridge_alpha": alpha,
        "placebo_seed": placebo,
        "mean_hits_v5": float(mean["V5平均"]),
        "mean_hits_v4": float(mean["V4冻结模型平均"]),
        "mean_hits_random": float(mean["随机基准平均"]),
        "mean_hits_v5_minus_v4": float(mean["V5_vs_V4差值"]),
        "mean_hits_v5_minus_v4_ci": [
            float(mean["V5_vs_V4差值95%CI下限"]),
            float(mean["V5_vs_V4差值95%CI上限"]),
        ],
        "batch_3plus_v5": float(batch["V5平均"]),
        "batch_3plus_v4": float(batch["V4冻结模型平均"]),
        "batch_3plus_random": float(batch["随机基准平均"]),
        "brier_v5": float(brier["V5平均"]),
        "brier_v4": float(brier["V4冻结模型平均"]),
        "brier_random": float(brier["随机基准平均"]),
    }

# Sensitivity grid: one-factor comparisons around the baseline.
cases = []
for window in (250, 500, 800):
    cases.append(run_case(f"window_{window}", window, 1.0))
for alpha in (0.1, 1.0, 10.0):
    if alpha != 1.0:
        cases.append(run_case(f"ridge_{alpha}", 500, alpha))

# Three independently seeded label permutations; the held-out outcomes stay
# untouched. A real predictive signal should not be reproducible from shuffled
# training labels.
placebos = [
    run_case(f"label_placebo_{seed}", 500, 1.0, placebo=seed)
    for seed in (1701, 2701, 3701)
]

print(json.dumps({
    "warning": "All results are retrospective diagnostics on previously explored draws 1039-1088; not independent or prospective evidence.",
    "test_range": [1039, 1088],
    "sensitivity": cases,
    "label_permutation_placebos": placebos,
}, ensure_ascii=False, indent=2))
