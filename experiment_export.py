# -*- coding: utf-8 -*-
"""Build a portable ZIP bundle of the current offline Lotto 6/45 research session.

Only explicitly allow-listed artifacts are exported. No environment variables,
secrets, arbitrary session-state values, or network data are collected.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import zipfile
from typing import Any, Mapping

import pandas as pd

EXPORT_SCHEMA_VERSION = 1
ENGINE_VERSION = "3.6.2"

# Session-state key -> folder / file stem / artifact type.
_ARTIFACTS = {
    "v36_bt_raw": ("backtest", "rolling_backtest_detail.csv", "dataframe"),
    "v36_bt_summary": ("backtest", "rolling_backtest_summary.csv", "dataframe"),
    "v36_bt_metadata": ("backtest", "rolling_backtest_metadata.json", "json"),
    "v361_param_detail": ("parameter_search", "parameter_search_detail.csv", "dataframe"),
    "v361_param_summary": ("parameter_search", "parameter_search_ranking.csv", "dataframe"),
    "v361_param_meta": ("parameter_search", "parameter_search_metadata.json", "json"),
    "v361_param_context": ("parameter_search", "parameter_search_context.json", "json"),
    "v361_holdout_raw": ("holdout", "reserved_holdout_detail.csv", "dataframe"),
    "v361_holdout_summary": ("holdout", "reserved_holdout_summary.csv", "dataframe"),
    "v361_holdout_meta": ("holdout", "reserved_holdout_metadata.json", "json"),
    "v361_ensemble_raw": ("ensemble", "ensemble_detail.csv", "dataframe"),
    "v361_ensemble_summary": ("ensemble", "ensemble_summary.csv", "dataframe"),
    "v361_ensemble_meta": ("ensemble", "ensemble_metadata.json", "json"),
    "v361_portfolio": ("portfolio", "optimized_10_tickets.csv", "dataframe"),
    "v361_portfolio_report": ("portfolio", "portfolio_quality_report.json", "json"),
    "v36_candidates": ("candidate_generation", "generated_candidates.csv", "dataframe"),
}


def _json_safe(value: Any) -> Any:
    """Convert common NumPy/pandas values to strict JSON-compatible values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if pd.notna(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    # NumPy scalar support without importing NumPy as a hard dependency here.
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass
    if pd.isna(value):
        return None
    return str(value)


def build_research_bundle(
    history: pd.DataFrame,
    quality: Mapping[str, Any],
    data_source: str,
    session_state: Mapping[str, Any],
    project_state: Mapping[str, Any] | None = None,
    include_history: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """Return (zip bytes, manifest). Only known session keys are considered."""
    if not isinstance(history, pd.DataFrame) or history.empty:
        raise ValueError("没有可导出的有效历史数据。")

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    history_csv = history.to_csv(index=False).encode("utf-8-sig")
    history_hash = hashlib.sha256(history_csv).hexdigest()

    output = io.BytesIO()
    included: list[str] = []
    missing: list[str] = []
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        if include_history:
            path = "history/cleaned_manual_history.csv"
            zf.writestr(path, history_csv)
            included.append(path)

        if project_state is not None:
            try:
                from model_management import export_project_state
                project_json = export_project_state(dict(project_state))
            except Exception:
                project_json = json.dumps(_json_safe(dict(project_state)), ensure_ascii=False, indent=2)
            path = "model_management/model_and_experiment_archive.json"
            zf.writestr(path, project_json.encode("utf-8"))
            included.append(path)

        candidate_meta_keys = ("v36_candidate_model", "v36_candidate_window", "v36_candidate_relaxed")
        candidate_metadata = {key: session_state.get(key) for key in candidate_meta_keys if session_state.get(key) is not None}
        if candidate_metadata:
            candidate_path = "candidate_generation/candidate_metadata.json"
            zf.writestr(candidate_path, json.dumps(_json_safe(candidate_metadata), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
            included.append(candidate_path)

        for key, (folder, filename, kind) in _ARTIFACTS.items():
            value = session_state.get(key)
            if value is None:
                missing.append(key)
                continue
            path = f"{folder}/{filename}"
            if kind == "dataframe":
                if not isinstance(value, pd.DataFrame) or value.empty:
                    missing.append(key)
                    continue
                payload = value.to_csv(index=False).encode("utf-8-sig")
            else:
                try:
                    payload = json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
                except (TypeError, ValueError) as exc:
                    missing.append(f"{key} (JSON serialization failed: {exc})")
                    continue
            zf.writestr(path, payload)
            included.append(path)

        manifest = {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "export_version": ENGINE_VERSION,
            "created_at_utc": now.isoformat(timespec="seconds"),
            "data_policy": "offline_manual_import_only",
            "network_policy": "no_lottery_official_site_api_scraping_or_external_draw_fetching",
            "data_source_label": str(data_source),
            "history_rows": int(len(history)),
            "history_first_draw": int(history["draw_no"].min()) if "draw_no" in history.columns else None,
            "history_last_draw": int(history["draw_no"].max()) if "draw_no" in history.columns else None,
            "history_sha256_utf8_sig_csv": history_hash,
            "history_included": bool(include_history),
            "quality_report": _json_safe(dict(quality)),
            "included_artifacts": included,
            "unavailable_or_empty_artifacts": missing,
            "notes": [
                "This bundle contains only allow-listed current-session research artifacts.",
                "A missing artifact means it was not generated or is not present in this Streamlit session.",
                "Parameter-search rankings are exploratory and are not proof of predictive advantage.",
                "The historical draws are manually supplied by the user; no official lottery website was accessed.",
            ],
        }
        # The manifest lists itself and the explanatory README as bundle members.
        included.extend(["manifest.json", "README_CN.txt"])
        manifest["included_artifacts"] = included
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
        zf.writestr(
            "README_CN.txt",
            (
                "Lotto 6/45 研究实验包\n\n"
                "本压缩包用于审查当前 Streamlit 会话中的回测、参数搜索、保留区间、集成和组合优化结果。\n"
                "manifest.json 列出实际包含的文件及未生成的文件。\n\n"
                "数据规则：仅使用用户手动导入的 CSV/Excel；不连接韩国 Lotto 官网、API 或抓取服务。\n"
                "请注意：参数搜索第一名属于探索性结果，不代表统计显著或未来预测能力。\n"
            ).encode("utf-8"),
        )

    return output.getvalue(), manifest
