# -*- coding: utf-8 -*-
"""Session-exportable model lifecycle and experiment log helpers."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from model_registry import ENGINE_VERSION, get_model_registry

SCHEMA_VERSION = 1
MODEL_STATUSES = ["实验", "候选", "验证通过"]


def new_project_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": {
            item["name"]: {
                "key": item["key"],
                "version": item["version"],
                "family": item["family"],
                "description": item["description"],
                "status": "基准" if item["status"] == "基准" else "实验",
                "status_note": "系统初始状态；尚未自动认定任何统计模型具有预测优势。",
                "evidence_experiment_id": None,
            }
            for item in get_model_registry()
        },
        "experiments": [],
    }


def validate_project_state(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("模型档案 JSON 版本不受支持。")
    if not isinstance(payload.get("models"), dict) or not isinstance(payload.get("experiments"), list):
        raise ValueError("模型档案缺少 models 或 experiments 结构。")
    base = new_project_state()
    for name, record in payload["models"].items():
        if name not in base["models"]:
            continue
        if not isinstance(record, dict) or record.get("status") not in MODEL_STATUSES + ["基准"]:
            raise ValueError(f"模型 {name} 的状态字段无效。")
        base["models"][name].update(record)
    experiments = payload["experiments"]
    if len(experiments) > 2000:
        raise ValueError("实验档案超过2000条，请先归档旧记录。")
    base["experiments"] = [item for item in experiments if isinstance(item, dict)][-2000:]
    evidence_by_id = {item.get("experiment_id"): item for item in base["experiments"]}
    for model_name, record in base["models"].items():
        if record.get("status") == "验证通过":
            evidence = evidence_by_id.get(record.get("evidence_experiment_id"))
            if (not evidence or not evidence.get("holdout_evaluated")
                    or evidence.get("model_name") != model_name):
                raise ValueError(f"模型 {model_name} 的‘验证通过’状态缺少匹配的保留区间验证证据。")
    base["created_at_utc"] = payload.get("created_at_utc", base["created_at_utc"])
    return base


def add_experiment(
    state: dict[str, Any],
    experiment_type: str,
    metadata: dict[str, Any],
    summary_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Append a compact experiment summary; raw ticket detail stays in CSV exports."""
    valid = validate_project_state(state)
    generated_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    experiment_id = str(metadata.get("experiment_id") or f"{experiment_type}-{generated_id}-{len(valid['experiments']) + 1}")
    record = {
        "experiment_id": experiment_id,
        "experiment_type": str(experiment_type),
        "engine_version": str(metadata.get("engine_version", ENGINE_VERSION)),
        "created_at_utc": str(metadata.get("created_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))),
        "data_first_draw": metadata.get("data_first_draw"),
        "data_last_draw": metadata.get("data_last_draw"),
        "test_start_draw": metadata.get("test_start_draw", metadata.get("validation_start_draw")),
        "test_end_draw": metadata.get("test_end_draw", metadata.get("validation_end_draw")),
        "model_name": metadata.get("model_name", metadata.get("evaluated_model")),
        "best_config": metadata.get("best_config"),
        "holdout_evaluated": bool(metadata.get("holdout_evaluated", False)),
        "data_fingerprint": metadata.get("data_fingerprint"),
        "reserved_holdout_start_draw": metadata.get("reserved_holdout_start_draw"),
        "reserved_holdout_end_draw": metadata.get("reserved_holdout_end_draw"),
        "summary": (summary_records or [])[:30],
        "metadata_excerpt": {
            key: metadata.get(key) for key in (
                "lookback_minimum", "recent_window", "tickets_per_draw_per_seed",
                "seed_groups", "base_seed", "grid_size", "reserved_holdout_draws",
                "holdout_consumed", "network_policy", "data_fingerprint",
                "reserved_holdout_start_draw", "reserved_holdout_end_draw",
            ) if key in metadata
        },
    }
    if not any(x.get("experiment_id") == experiment_id for x in valid["experiments"]):
        valid["experiments"].append(record)
    return valid


def change_model_status(
    state: dict[str, Any],
    model_name: str,
    new_status: str,
    note: str,
    evidence_experiment_id: str | None = None,
) -> dict[str, Any]:
    valid = validate_project_state(state)
    if model_name not in valid["models"]:
        raise ValueError(f"模型未登记：{model_name}")
    if new_status not in MODEL_STATUSES:
        raise ValueError("模型状态必须为实验、候选或验证通过。")
    if not str(note).strip():
        raise ValueError("调整模型状态必须填写原因。")
    if new_status == "验证通过":
        matching = [x for x in valid["experiments"] if x.get("experiment_id") == evidence_experiment_id]
        if (not matching or not matching[0].get("holdout_evaluated")
                or matching[0].get("model_name") != model_name):
            raise ValueError("模型标记为验证通过必须关联该模型已执行的保留区间验证记录。")
    valid["models"][model_name].update({
        "status": new_status,
        "status_note": str(note).strip(),
        "evidence_experiment_id": evidence_experiment_id,
        "status_updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return valid


def export_project_state(state: dict[str, Any]) -> str:
    return json.dumps(validate_project_state(state), ensure_ascii=False, indent=2)


def import_project_state(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 文件无法解析：{exc}") from exc
    return validate_project_state(payload)
