# -*- coding: utf-8 -*-
"""CLI for preparing or scoring immutable V8.7 forward locks."""
from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v87_forward_protocol import (
    prepare_v87_lock, score_legacy_v8x_lock, score_v87_lock,
    write_score_report,
)


def _read_history(path: str) -> tuple[pd.DataFrame, bytes]:
    raw = Path(path).read_bytes()
    frame = pd.read_csv(BytesIO(raw))
    return frame, raw


def _latest_draw(history: pd.DataFrame) -> int:
    if history.empty:
        raise ValueError("手动CSV没有可用开奖记录。")
    return int(history["draw_no"].max())


def command_prepare(args: argparse.Namespace) -> None:
    history, raw = _read_history(args.history)
    bundle, manifest, paths = prepare_v87_lock(
        history,
        commitment_dir=args.commitment_dir,
        output_dir=args.output_dir,
        raw_csv_bytes=raw,
        training_window=args.training_window,
        min_history=args.min_history,
        ridge_alpha=args.ridge_alpha,
        master_seed=args.master_seed,
    )
    target = int(bundle["target_draw"])
    if args.github_output:
        with Path(args.github_output).open("a", encoding="utf-8") as stream:
            stream.write(f"target_draw={target}\n")
            stream.write(f"candidate_path={paths[0]}\n")
            stream.write(f"manifest_path={paths[1]}\n")
    print(json.dumps({
        "target_draw": target,
        "commitment_id": manifest["commitment_id"],
        "candidate_path": paths[0],
        "manifest_path": paths[1],
        "candidate_file_sha256": manifest["candidate_file_sha256"],
        "canonical_payload_sha256": manifest["canonical_payload_sha256"],
        "note": "尚未发布锁定分支；候选包需与清单一起提交到唯一不可覆盖分支。",
    }, ensure_ascii=False, indent=2))


def command_score(args: argparse.Namespace) -> None:
    history, _ = _read_history(args.history)
    report = score_v87_lock(
        history, args.candidate, args.manifest,
        commitment_timestamp_utc=args.commit_timestamp_utc,
    )
    path = write_score_report(report, args.output)
    print(json.dumps({
        "score_path": path,
        "target_draw": report["target_draw"],
        "commitment_timing": report["commitment_timing"],
        "strategy_metrics": report["strategy_metrics"],
        "inference_boundary": report["inference_boundary"],
    }, ensure_ascii=False, indent=2))


def command_score_legacy(args: argparse.Namespace) -> None:
    history, _ = _read_history(args.history)
    report = score_legacy_v8x_lock(
        history, args.manifest, args.archive,
        commitment_timestamp_utc=args.commit_timestamp_utc,
    )
    path = write_score_report(report, args.output)
    print(json.dumps({
        "score_path": path,
        "target_draw": report["target_draw"],
        "commitment_timing": report["commitment_timing"],
        "strategy_metrics": report["strategy_metrics"],
        "inference_boundary": report["inference_boundary"],
    }, ensure_ascii=False, indent=2))


def command_latest_draw(args: argparse.Namespace) -> None:
    history, _ = _read_history(args.history)
    print(_latest_draw(history))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V8.7 immutable forward validation protocol")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="generate an immutable next-draw candidate package")
    prepare.add_argument("--history", default="data/lotto645.csv")
    prepare.add_argument("--commitment-dir", default="data/forward_lock_commitments")
    prepare.add_argument("--output-dir", default="artifacts/v87")
    prepare.add_argument("--training-window", type=int, default=500)
    prepare.add_argument("--min-history", type=int, default=180)
    prepare.add_argument("--ridge-alpha", type=float, default=1.0)
    prepare.add_argument("--master-seed", type=int, default=87_2026)
    prepare.add_argument("--github-output", default="")
    prepare.set_defaults(func=command_prepare)

    score = sub.add_parser("score", help="score a V8.7 lock after the result is manually imported")
    score.add_argument("--history", default="data/lotto645.csv")
    score.add_argument("--candidate", required=True)
    score.add_argument("--manifest", required=True)
    score.add_argument("--commit-timestamp-utc", default="")
    score.add_argument("--output", required=True)
    score.set_defaults(func=command_score)

    legacy = sub.add_parser("score-legacy", help="score the existing V8.x 1252 archive without replacing it")
    legacy.add_argument("--history", default="data/lotto645.csv")
    legacy.add_argument("--manifest", required=True)
    legacy.add_argument("--archive", required=True)
    legacy.add_argument("--commit-timestamp-utc", default="")
    legacy.add_argument("--output", required=True)
    legacy.set_defaults(func=command_score_legacy)

    latest = sub.add_parser("latest-draw", help="print the latest manually imported draw number")
    latest.add_argument("--history", default="data/lotto645.csv")
    latest.set_defaults(func=command_latest_draw)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
