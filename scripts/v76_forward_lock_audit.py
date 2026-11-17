# -*- coding: utf-8 -*-
"""Reproducible integrity audit for the hash-committed 1251 pre-draw lock.

Only the user-maintained CSV is read. An optional V5.1 archive JSON can be
supplied to verify its committed digest and, when the target draw is present,
score the original locked tickets with the existing evaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from forward_lock_commitment import (
    audit_1251_history_prefix,
    load_1251_commitment,
    verify_1251_archive_bytes,
)
from lotto_core import validate_frame
from v40_store import import_event_archive
from v52_rolling_validation import evaluate_v51_forward_lock_events

ROOT = Path(__file__).resolve().parents[1]


def run_audit(
    history: pd.DataFrame,
    *,
    archive_bytes: bytes | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    commitment = manifest or load_1251_commitment()
    clean = validate_frame(history)
    prefix_report = audit_1251_history_prefix(clean, commitment)
    report: dict[str, Any] = {
        "experiment": "V7.6 frozen 1251 forward-lock integrity audit",
        "status": prefix_report["status"],
        "prefix_audit": prefix_report,
        "archive_commitment": {
            "present": archive_bytes is not None,
            "matched": None,
            "details": None,
        },
        "scoring": {
            "status": "not_run_archive_not_supplied",
            "summary": [],
            "ticket_detail_rows": 0,
        },
        "limitations": [
            "开奖数据只从用户手动维护的CSV读取；本脚本不访问任何开奖网站或API。",
            "SHA-256承诺只锚定锁定档案内容；是否早于实际开奖，仍须对照GitHub承诺提交时间和手动导入的目标期开奖日期。",
            "单期命中结果只能作描述性复盘，不能单独证明模型存在统计显著的预测优势。",
            "本脚本不会生成、修改或覆盖前瞻锁定号码及原始开奖数据。",
        ],
    }

    if archive_bytes is not None:
        verification = verify_1251_archive_bytes(archive_bytes, commitment)
        report["archive_commitment"] = {
            "present": True,
            "matched": bool(verification["matched"]),
            "details": verification,
        }
        if verification["matched"]:
            events = import_event_archive(archive_bytes)
            detail, summary, status = evaluate_v51_forward_lock_events(clean, events)
            report["scoring"] = {
                "status": (
                    str(status.iloc[0]["状态"])
                    if not status.empty else "no_matching_v51_lock_status"
                ),
                "summary": summary.to_dict(orient="records"),
                "ticket_detail_rows": int(len(detail)),
                "status_rows": status.to_dict(orient="records"),
            }
        else:
            report["scoring"] = {
                "status": "not_run_archive_does_not_match_commitment",
                "summary": [],
                "ticket_detail_rows": 0,
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        default=str(ROOT / "data" / "lotto645.csv"),
        help="用户手动维护的开奖CSV（默认：data/lotto645.csv）",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help="可选：原始V5.1前瞻锁定JSON档案路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "v76"),
        help="报告输出目录",
    )
    args = parser.parse_args()

    history = pd.read_csv(args.history)
    archive_bytes = Path(args.archive).read_bytes() if args.archive else None
    report = run_audit(history, archive_bytes=archive_bytes)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "forward_lock_1251_integrity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
