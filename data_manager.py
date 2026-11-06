# -*- coding: utf-8 -*-
"""Manual-only history data loading and validation helpers.

This module intentionally contains no HTTP client, URL, API, scraping, or sync code.
History must be supplied by the user as a CSV or Excel file.
"""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

from lotto_core import COLUMNS, quality_report, validate_frame

SUPPORTED_EXTENSIONS = {"csv", "xlsx", "xls"}


def read_manual_upload(file_obj: BinaryIO, filename: str) -> tuple[pd.DataFrame, dict]:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("仅支持 CSV、XLSX 或 XLS 格式的手动历史数据文件。")
    file_obj.seek(0)
    if suffix == "csv":
        try:
            raw = pd.read_csv(file_obj)
        except UnicodeDecodeError:
            file_obj.seek(0)
            raw = pd.read_csv(file_obj, encoding="cp949")
    else:
        raw = pd.read_excel(file_obj)
    if raw.empty:
        raise ValueError("上传文件没有数据行。")
    clean = validate_frame(raw)
    if clean.empty:
        raise ValueError("没有找到有效开奖记录。请检查期号、六个主号码及奖金号码列。")
    report = quality_report(raw)
    report["source_mode"] = "manual_upload"
    report["source_filename"] = Path(filename).name
    report["source_rows"] = int(len(raw))
    return clean, report


def load_repository_history(path: str | Path) -> tuple[pd.DataFrame, dict]:
    target = Path(path)
    if not target.exists():
        return pd.DataFrame(columns=COLUMNS), {"status": "missing", "path": str(target)}
    raw = pd.read_csv(target)
    clean = validate_frame(raw)
    report = quality_report(raw)
    report["source_mode"] = "manual_repository_file"
    report["source_filename"] = target.name
    report["source_rows"] = int(len(raw))
    return clean, report
