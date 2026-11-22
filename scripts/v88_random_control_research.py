# -*- coding: utf-8 -*-
"""Run V8.8 multi-random-control replay using manually supplied CSV history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lotto_core import validate_frame
from v85_hybrid_backtest import run_v85_walk_forward
from v88_random_control import run_v88_random_control, write_v88_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="data/lotto645.csv")
    parser.add_argument("--test-draws", type=int, default=500)
    parser.add_argument("--random-portfolios", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=882026)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=10)
    parser.add_argument("--output-dir", default="artifacts/v88")
    args = parser.parse_args()

    frame = validate_frame(pd.read_csv(args.history))
    base = run_v85_walk_forward(
        frame, test_draws=args.test_draws, training_window=500, min_history=180,
        ridge_alpha=1.0, seed=args.seed, bootstrap_repetitions=1000,
    )
    result = run_v88_random_control(
        frame, base["detail"], random_portfolios_per_draw=args.random_portfolios,
        seed=args.seed, bootstrap_repetitions=args.bootstrap_repetitions,
        block_length=args.block_length,
    )
    report_path, detail_path = write_v88_artifacts(result, args.output_dir)
    print(json.dumps({
        "report": report_path, "per_draw": detail_path,
        "target_start_draw": result["report"]["target_start_draw"],
        "target_end_draw": result["report"]["target_end_draw"],
        "test_draws": result["report"]["test_draws"],
        "random_portfolios_per_draw": result["report"]["random_portfolios_per_draw"],
        "primary_comparisons": result["report"]["primary_comparisons"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
