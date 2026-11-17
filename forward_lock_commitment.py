# -*- coding: utf-8 -*-
"""Hash-only commitment and integrity checks for the frozen 1251 forward lock.

The manifest contains no ticket numbers. It commits only to the exact archive
content and the historical prefix used to create the lock. All draw data is
provided manually; this module performs no network access.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from lotto_core import validate_frame
from v40_governance import data_fingerprint
from v40_store import import_event_archive

ROOT = Path(__file__).resolve().parent
DEFAULT_COMMITMENT_PATH = ROOT / "data" / "forward_lock_commitments" / "draw_1251.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256(canonical)


def load_1251_commitment(path: str | Path = DEFAULT_COMMITMENT_PATH) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("manifest_schema_version") != 1:
        raise ValueError("不支持的1251期锁定承诺清单版本。")
    if int(raw.get("target_draw", -1)) != 1251:
        raise ValueError("锁定承诺清单目标期号不是1251。")
    return raw


def _is_1251_lock_candidate(
    event: dict[str, Any],
    commitment: dict[str, Any],
) -> bool:
    """Recognize the committed event even if its event_id was altered."""
    expected_id = str(commitment["event_id"])
    if str(event.get("event_id")) == expected_id:
        return True
    payload = event.get("payload", {})
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    configuration = payload.get("configuration", {}) if isinstance(payload, dict) else {}
    try:
        targets_1251 = int(metadata.get("target_draw", -1)) == int(commitment.get("target_draw", -1))
    except (TypeError, ValueError):
        targets_1251 = False
    expected_engine = commitment.get("configuration", {}).get("engine_version")
    engine_matches = (
        metadata.get("engine_version") == expected_engine
        or configuration.get("engine_version") == expected_engine
    )
    return targets_1251 and (
        event.get("event_type") == "v51_forward_lock" or engine_matches
    )


def verify_1251_archive_bytes(
    archive_bytes: bytes,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a supplied V5.1 archive against the committed 1251 event digest."""
    commitment = manifest or load_1251_commitment()
    result: dict[str, Any] = {
        "applicable": False,
        "matched": False,
        "raw_file_sha256_matches": False,
        "canonical_archive_sha256_matches": False,
        "canonical_event_sha256_matches": False,
        "event_id_matches": False,
        "target_draw_matches": False,
        "target_event_count_matches": False,
        "history_fingerprint_matches": False,
        "configuration_matches": False,
        "model_ticket_count_matches": False,
        "random_ticket_count_matches": False,
        "probability_count_matches": False,
        "error": None,
    }
    try:
        events = import_event_archive(archive_bytes)
    except (ValueError, UnicodeDecodeError) as exc:
        result["error"] = f"archive_invalid:{type(exc).__name__}:{exc}"
        return result

    event_id = str(commitment["event_id"])
    candidates = [
        event for event in events
        if _is_1251_lock_candidate(event, commitment)
    ]
    if not candidates:
        return result

    result["applicable"] = True
    if len(candidates) != 1:
        result["error"] = "multiple_1251_lock_candidates"
        return result
    event = candidates[0]
    result["target_event_count_matches"] = True
    metadata = event.get("payload", {}).get("metadata", {})
    configuration = event.get("payload", {}).get("configuration", {})
    tickets = event.get("payload", {}).get("tickets", [])
    probabilities = event.get("payload", {}).get("number_probabilities", [])

    raw_sha = _sha256(archive_bytes)
    try:
        parsed_archive = json.loads(archive_bytes.decode("utf-8-sig"))
    except (ValueError, UnicodeDecodeError) as exc:
        result["error"] = f"archive_decode_failed:{type(exc).__name__}"
        return result

    result["raw_file_sha256_matches"] = (
        raw_sha == commitment["archive_commitment"]["raw_file_sha256"]
    )
    result["canonical_archive_sha256_matches"] = (
        _canonical_json_sha256(parsed_archive)
        == commitment["archive_commitment"]["canonical_archive_sha256"]
    )
    result["canonical_event_sha256_matches"] = (
        _canonical_json_sha256(event)
        == commitment["archive_commitment"]["canonical_event_sha256"]
    )
    result["event_id_matches"] = str(event.get("event_id")) == event_id
    result["target_draw_matches"] = (
        int(metadata.get("target_draw", -1)) == int(commitment["target_draw"])
        and int(metadata.get("history_last_draw", -1))
        == int(commitment["history"]["last_draw"])
    )
    result["history_fingerprint_matches"] = (
        metadata.get("history_fingerprint_sha256")
        == commitment["history"]["fingerprint_sha256"]
        and event.get("data_fingerprint_sha256")
        == commitment["history"]["fingerprint_sha256"]
    )
    expected_config = commitment["configuration"]
    config_fields = ("engine_version", "training_window", "ridge_alpha", "pair_strength", "seed")
    result["configuration_matches"] = (
        all(configuration.get(key) == expected_config[key] for key in config_fields)
        and metadata.get("random_seed") == expected_config["random_seed"]
    )
    model_count = sum(
        1 for row in tickets if row.get("策略") == "V5.1联合概率选号"
    )
    random_count = sum(
        1 for row in tickets if row.get("策略") == "均匀随机基准"
    )
    result["model_ticket_count_matches"] = (
        model_count == int(commitment["configuration"]["model_ticket_count"])
    )
    result["random_ticket_count_matches"] = (
        random_count == int(commitment["configuration"]["random_ticket_count"])
    )
    result["probability_count_matches"] = (
        len(probabilities) == int(commitment["configuration"]["probability_count"])
    )
    result["matched"] = all(
        result[key]
        for key in (
            "raw_file_sha256_matches",
            "canonical_archive_sha256_matches",
            "canonical_event_sha256_matches",
            "event_id_matches",
            "target_draw_matches",
            "target_event_count_matches",
            "history_fingerprint_matches",
            "configuration_matches",
            "model_ticket_count_matches",
            "random_ticket_count_matches",
            "probability_count_matches",
        )
    )
    return result


def filter_uncommitted_1251_events(
    events: list[dict[str, Any]],
    verification: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Exclude only the committed 1251 event when its uploaded archive mismatches."""
    commitment = manifest or load_1251_commitment()
    if not verification.get("applicable") or verification.get("matched"):
        return list(events)
    return [
        event for event in events
        if not _is_1251_lock_candidate(event, commitment)
    ]


def merge_events_with_1251_commitment(
    existing_events: list[dict[str, Any]],
    imported_events: list[dict[str, Any]],
    verification: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Merge event imports while replacing/removing any stale committed 1251 copy."""
    commitment = manifest or load_1251_commitment()
    existing = list(existing_events)
    incoming = list(imported_events)
    if verification.get("applicable"):
        # Never retain an older in-session copy when a 1251 archive is uploaded.
        existing = [
            event for event in existing
            if not _is_1251_lock_candidate(event, commitment)
        ]
        if not verification.get("matched"):
            incoming = filter_uncommitted_1251_events(
                incoming, verification, commitment
            )

    seen = {str(event.get("event_id")) for event in existing}
    for event in incoming:
        event_id = str(event.get("event_id"))
        if event_id not in seen:
            existing.append(event)
            seen.add(event_id)
    return existing


def audit_1251_history_prefix(
    history: pd.DataFrame,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check the current manual CSV still contains the exact locked 1–1250 prefix."""
    commitment = manifest or load_1251_commitment()
    clean = validate_frame(history).sort_values("draw_no").reset_index(drop=True)
    history_meta = commitment["history"]
    last_draw = int(history_meta["last_draw"])
    current_last = int(clean["draw_no"].max()) if not clean.empty else 0

    report: dict[str, Any] = {
        "commitment_id": commitment["commitment_id"],
        "target_draw": int(commitment["target_draw"]),
        "current_history_last_draw": current_last,
        "expected_history_last_draw": last_draw,
        "prefix_rows_expected": int(history_meta["rows"]),
        "prefix_rows_found": 0,
        "prefix_fingerprint_expected": history_meta["fingerprint_sha256"],
        "prefix_fingerprint_actual": None,
        "prefix_verified": False,
        "target_result_present": False,
        "status": "pending",
    }
    if clean.empty or current_last < last_draw:
        report["status"] = "waiting_for_locked_history_prefix"
        return report

    prefix = clean.loc[clean["draw_no"] <= last_draw].copy()
    report["prefix_rows_found"] = int(len(prefix))
    if len(prefix) != int(history_meta["rows"]):
        report["status"] = "locked_history_row_count_mismatch"
        return report
    if int(prefix["draw_no"].iloc[0]) != int(history_meta["first_draw"]):
        report["status"] = "locked_history_first_draw_mismatch"
        return report
    if int(prefix["draw_no"].iloc[-1]) != last_draw:
        report["status"] = "locked_history_last_draw_mismatch"
        return report

    actual_fingerprint = data_fingerprint(prefix)
    report["prefix_fingerprint_actual"] = actual_fingerprint
    report["prefix_verified"] = actual_fingerprint == history_meta["fingerprint_sha256"]
    if not report["prefix_verified"]:
        report["status"] = "locked_history_fingerprint_mismatch"
        return report

    report["target_result_present"] = int(commitment["target_draw"]) in set(
        clean["draw_no"].astype(int).tolist()
    )
    report["status"] = (
        "target_result_available_prefix_verified"
        if report["target_result_present"]
        else "awaiting_target_result_prefix_verified"
    )
    return report
