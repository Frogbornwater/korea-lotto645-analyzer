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

EXPORT_SCHEMA_VERSION = 2
ENGINE_VERSION = "4.0.0-governed"

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
    # V5.2 rolling replay: full row-level detail, summary and reproducibility metadata.
    "v52_replay_detail": ("v52_rolling_replay", "rolling_replay_detail.csv", "dataframe"),
    "v52_replay_summary": ("v52_rolling_replay", "rolling_replay_summary.csv", "dataframe"),
    "v52_replay_metadata": ("v52_rolling_replay", "rolling_replay_metadata.json", "json"),
    # V5.3 Monte Carlo audit: retain every replicate, per-draw RNG stream and manifest.
    "v53_audit_summary": ("v53_random_audit", "audit_summary.csv", "dataframe"),
    "v53_random_distribution": ("v53_random_audit", "random_experiment_distribution.csv", "dataframe"),
    "v53_rng_seed_records": ("v53_random_audit", "per_draw_rng_seeds.csv", "dataframe"),
    "v53_audit_manifest": ("v53_random_audit", "audit_manifest.json", "json"),
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

        # V3.7's joint optimizer stores multiple DataFrames inside one known
        # session-state result. Export only these explicit keys; never walk the
        # arbitrary session state or serialize secrets/widget values.
        joint_result_exports = {
            "v37_joint_result": {
                "prefix": "joint_optimization_v37",
                "dataframes": {
                    "search_summary": "search_ranking.csv",
                    "pareto_front": "pareto_front.csv",
                    "inner_period_detail": "inner_period_detail.csv",
                    "inner_fold_detail": "inner_fold_detail.csv",
                    "outer_detail": "outer_validation.csv",
                    "outer_selections": "outer_selections.csv",
                    "outer_summary": "outer_summary.csv",
                    "holdout_detail": "holdout_period_detail.csv",
                    "holdout_ticket_detail": "holdout_tickets.csv",
                    "holdout_summary": "holdout_statistics.csv",
                },
                "json": ["selected_config", "metadata"],
            },
            "v38_robust_result": {
                "prefix": "robust_research_v38",
                "dataframes": {
                    "screen_summary": "500_config_screen_ranking.csv",
                    "screen_period_detail": "screen_period_detail.csv",
                    "screen_fold_detail": "screen_fold_detail.csv",
                    "finalist_summary": "finalist_refinement_ranking.csv",
                    "finalist_period_detail": "finalist_period_detail.csv",
                    "finalist_fold_detail": "finalist_fold_detail.csv",
                    "predictive_pareto_front": "predictive_pareto_front.csv",
                    "structure_pareto_front": "structure_pareto_front.csv",
                    "outer_detail": "outer_rolling_validation.csv",
                    "outer_selections": "outer_selections.csv",
                    "outer_summary": "outer_summary.csv",
                },
                "json": ["selected_config", "metadata"],
            },
        }
        result_exports: dict[str, list[str]] = {}
        for state_key, spec in joint_result_exports.items():
            result = session_state.get(state_key)
            if not isinstance(result, dict):
                missing.append(state_key)
                continue
            paths: list[str] = []
            prefix = str(spec["prefix"])
            for field, filename in spec["dataframes"].items():
                value = result.get(field)
                if not isinstance(value, pd.DataFrame) or value.empty:
                    missing.append(f"{state_key}.{field}")
                    continue
                path = f"{prefix}/{filename}"
                zf.writestr(path, value.to_csv(index=False).encode("utf-8-sig"))
                included.append(path)
                paths.append(path)
            for field in spec["json"]:
                value = result.get(field)
                if value is None:
                    missing.append(f"{state_key}.{field}")
                    continue
                try:
                    payload = json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
                except (TypeError, ValueError) as exc:
                    missing.append(f"{state_key}.{field} (JSON serialization failed: {exc})")
                    continue
                path = f"{prefix}/{field}.json"
                zf.writestr(path, payload)
                included.append(path)
                paths.append(path)
            result_exports[state_key] = paths

        # Forward locks are also an explicit allow-list: each value must contain
        # exactly the known {detail: DataFrame, metadata: dict} structure.
        forward_locks = session_state.get("v38_forward_locks")
        if isinstance(forward_locks, dict):
            for target_draw, lock in sorted(forward_locks.items(), key=lambda item: int(item[0])):
                if not isinstance(lock, dict):
                    continue
                detail = lock.get("detail")
                metadata = lock.get("metadata")
                if not isinstance(detail, pd.DataFrame) or detail.empty or not isinstance(metadata, dict):
                    missing.append(f"v38_forward_locks.{target_draw}")
                    continue
                folder = f"forward_locks/draw_{int(target_draw)}"
                csv_path = f"{folder}/locked_tickets.csv"
                json_path = f"{folder}/lock_metadata.json"
                zf.writestr(csv_path, detail.to_csv(index=False).encode("utf-8-sig"))
                zf.writestr(json_path, json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
                included.extend([csv_path, json_path])
        else:
            missing.append("v38_forward_locks")

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
            "research_result_exports": result_exports,
            "unavailable_or_empty_artifacts": missing,
            "notes": [
                "This bundle contains only allow-listed current-session research artifacts.",
                "V5.2 rolling replay and V5.3 randomized audit detail, seed records and manifests are included when present.",
                "V3.7 joint-result DataFrames and metadata are explicitly exported when present.",
                "V3.8 predictive ranking and portfolio-structure ranking are separate objectives.",
                "A missing artifact means it was not generated or is not present in this Streamlit session.",
                "Parameter-search rankings are exploratory and are not proof of predictive advantage.",
                "The historical draws are manually supplied by the user; no official lottery website was accessed.",
            ],
        }
        # Hash every payload member so an imported archive can be integrity-checked.
        manifest["artifact_sha256"] = {
            name: hashlib.sha256(zf.read(name)).hexdigest()
            for name in included
        }
        # The manifest lists itself and the explanatory README as bundle members.
        included.extend(["manifest.json", "README_CN.txt"])
        manifest["included_artifacts"] = included
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
        zf.writestr(
            "README_CN.txt",
            (
                "Lotto 6/45 研究实验包\n\n"
                "本压缩包用于审查当前 Streamlit 会话中的回测、V3.7/V3.8研究、V5.2逐期回放、V5.3随机审计、前瞻锁定和模型档案。\n"
                "manifest.json 列出实际包含的文件及未生成的文件。\n\n"
                "数据规则：仅使用用户手动导入的 CSV/Excel；不连接韩国 Lotto 官网、API 或抓取服务。\n"
                "请注意：参数搜索第一名属于探索性结果，不代表统计显著或未来预测能力。\n"
            ).encode("utf-8"),
        )

    return output.getvalue(), manifest


def restore_v54_research_bundle(payload: bytes) -> dict[str, Any]:
    """Validate a V5.4 ZIP and restore only the allow-listed V5.2/V5.3 research artifacts."""
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise ValueError("归档文件为空。")
    if len(payload) > 100 * 1024 * 1024:
        raise ValueError("归档文件超过100MB安全上限。")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload), "r")
    except zipfile.BadZipFile as exc:
        raise ValueError("文件不是有效的ZIP实验归档。") from exc
    with archive:
        names = set(archive.namelist())
        if "manifest.json" not in names:
            raise ValueError("归档缺少manifest.json。")
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        if int(manifest.get("export_schema_version", 0)) < 2:
            raise ValueError("归档版本过旧或不支持；请使用V5.4导出的完整ZIP。")
        if manifest.get("network_policy") != "no_lottery_official_site_api_scraping_or_external_draw_fetching":
            raise ValueError("归档数据政策标识不匹配，已拒绝导入。")
        hashes = manifest.get("artifact_sha256")
        if not isinstance(hashes, dict):
            raise ValueError("归档缺少文件完整性校验清单。")
        for name, expected in hashes.items():
            if name not in names or name.startswith("/") or ".." in name.split("/"):
                raise ValueError(f"归档文件路径无效或缺失：{name}")
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            if actual != expected:
                raise ValueError(f"文件完整性校验失败：{name}")
        restored: dict[str, Any] = {"manifest": manifest}
        csv_map = {
            "v52_replay_detail": "v52_rolling_replay/rolling_replay_detail.csv",
            "v52_replay_summary": "v52_rolling_replay/rolling_replay_summary.csv",
            "v53_audit_summary": "v53_random_audit/audit_summary.csv",
            "v53_random_distribution": "v53_random_audit/random_experiment_distribution.csv",
            "v53_rng_seed_records": "v53_random_audit/per_draw_rng_seeds.csv",
        }
        json_map = {
            "v52_replay_metadata": "v52_rolling_replay/rolling_replay_metadata.json",
            "v53_audit_manifest": "v53_random_audit/audit_manifest.json",
        }
        for key, path in csv_map.items():
            if path in names:
                restored[key] = pd.read_csv(io.BytesIO(archive.read(path)))
        for key, path in json_map.items():
            if path in names:
                restored[key] = json.loads(archive.read(path).decode("utf-8"))
        required = {"v52_replay_detail", "v52_replay_summary", "v52_replay_metadata"}
        if not required.issubset(restored):
            raise ValueError("归档未包含完整的V5.2逐期回放明细、摘要和元数据。")
        metadata = restored["v52_replay_metadata"]
        restored["v52_rolling_result"] = {
            **metadata,
            "detail": restored["v52_replay_detail"],
            "summary": restored["v52_replay_summary"],
        }
        if {"v53_audit_summary", "v53_random_distribution", "v53_rng_seed_records", "v53_audit_manifest"}.issubset(restored):
            restored["v53_random_audit"] = {
                "summary": restored["v53_audit_summary"],
                "random_distribution": restored["v53_random_distribution"],
                "rng_seed_records": restored["v53_rng_seed_records"],
                "manifest": restored["v53_audit_manifest"],
            }
        return restored
