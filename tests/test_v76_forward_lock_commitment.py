# -*- coding: utf-8 -*-
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from forward_lock_commitment import (
    _canonical_json_sha256,
    audit_1251_history_prefix,
    load_1251_commitment,
    verify_1251_archive_bytes,
)
from lotto_core import COLUMNS, validate_frame
from v40_governance import data_fingerprint, make_event
from v40_store import export_event_archive
from v52_rolling_validation import (
    RANDOM_STRATEGY,
    V51_LOCK_EVENT_TYPE,
    V51_STRATEGY,
    evaluate_v51_forward_lock_events,
)

ROOT = Path(__file__).resolve().parents[1]


def synthetic_history(n=5, seed=7601):
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(1, n + 1):
        nums = sorted(rng.choice(np.arange(1, 46), size=6, replace=False).tolist())
        bonus = int(next(x for x in range(1, 46) if x not in nums))
        rows.append({
            "draw_no": draw, "date": "", "n1": nums[0], "n2": nums[1],
            "n3": nums[2], "n4": nums[3], "n5": nums[4], "n6": nums[5],
            "bonus": bonus,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def make_synthetic_v51_event(history, target_draw):
    fingerprint = data_fingerprint(history)
    metadata = {
        "target_draw": target_draw,
        "history_first_draw": int(history["draw_no"].min()),
        "history_last_draw": int(history["draw_no"].max()),
        "history_rows": len(history),
        "history_fingerprint_sha256": fingerprint,
        "engine_version": "5.0.0-predictive-research",
        "training_window": 500,
        "ridge_alpha": 1.0,
        "pair_strength": 0.5,
        "seed": 5050,
        "random_seed": 18861,
        "ticket_count": 10,
    }
    configuration = {
        "model": V51_STRATEGY,
        "engine_version": "5.0.0-predictive-research",
        "training_window": 500,
        "ridge_alpha": 1.0,
        "pair_strength": 0.5,
        "requested_overlap": 4,
        "effective_overlap": 4,
        "ticket_count": 10,
        "seed": 5050,
    }
    base_ticket = [1, 2, 3, 4, 5, 6]
    random_ticket = [7, 8, 9, 10, 11, 12]
    tickets = []
    for strategy, ticket in ((V51_STRATEGY, base_ticket), (RANDOM_STRATEGY, random_ticket)):
        record = {"策略": strategy, "注序": 1, "号码": " · ".join(f"{n:02d}" for n in ticket)}
        record.update({f"n{i}": n for i, n in enumerate(ticket, 1)})
        tickets.append(record)
    payload = {
        "metadata": metadata,
        "configuration": configuration,
        "number_probabilities": [
            {"号码": n, "预测入选概率": 6.0 / 45.0} for n in range(1, 46)
        ],
        "tickets": tickets,
    }
    return make_event(V51_LOCK_EVENT_TYPE, fingerprint, payload)


class V76ForwardLockTests(unittest.TestCase):
    def test_1251_real_history_prefix_matches_committed_manifest(self):
        manifest = load_1251_commitment()
        history = pd.read_csv(ROOT / "data" / "lotto645.csv")
        audit = audit_1251_history_prefix(history, manifest)
        self.assertTrue(audit["prefix_verified"])
        self.assertEqual(audit["prefix_rows_found"], 1250)
        self.assertEqual(audit["current_history_last_draw"], 1250)
        self.assertFalse(audit["target_result_present"])
        self.assertEqual(audit["status"], "awaiting_target_result_prefix_verified")

    def test_prefix_integrity_is_checked_while_target_result_is_pending(self):
        history = synthetic_history(5)
        event = make_synthetic_v51_event(history, target_draw=6)
        _, summary, status = evaluate_v51_forward_lock_events(history, [event])
        self.assertTrue(summary.empty)
        self.assertEqual(
            status.iloc[0]["状态"],
            "等待开奖数据（锁定历史前缀指纹已核验）",
        )

        changed_history = synthetic_history(5, seed=7602)
        _, summary, status = evaluate_v51_forward_lock_events(changed_history, [event])
        self.assertTrue(summary.empty)
        self.assertEqual(
            status.iloc[0]["状态"],
            "拒绝评分：锁定历史前缀不完整或指纹不匹配",
        )

    def test_nonconsecutive_target_is_rejected(self):
        history = synthetic_history(5)
        event = make_synthetic_v51_event(history, target_draw=7)
        _, summary, status = evaluate_v51_forward_lock_events(history, [event])
        self.assertTrue(summary.empty)
        self.assertEqual(status.iloc[0]["状态"], "无效：目标期必须紧接锁定历史末期")

    def test_target_is_scored_only_after_matching_prefix_and_result_are_present(self):
        history = synthetic_history(5)
        event = make_synthetic_v51_event(history, target_draw=6)
        next_row = synthetic_history(6).tail(1)
        extended = pd.concat([history, next_row], ignore_index=True)
        detail, summary, status = evaluate_v51_forward_lock_events(extended, [event])
        self.assertEqual(status.iloc[0]["状态"], "已评分")
        self.assertEqual(len(detail), 2)
        self.assertEqual(len(summary), 1)

    def test_archive_commitment_verifies_exact_event_without_exposing_tickets(self):
        history = synthetic_history(5)
        event = make_synthetic_v51_event(history, target_draw=6)
        archive = export_event_archive([event])
        parsed = json.loads(archive.decode("utf-8"))
        manifest = {
            "target_draw": 6,
            "event_id": event["event_id"],
            "history": {
                "last_draw": 5,
                "fingerprint_sha256": data_fingerprint(history),
            },
            "archive_commitment": {
                "raw_file_sha256": hashlib.sha256(archive).hexdigest(),
                "canonical_archive_sha256": _canonical_json_sha256(parsed),
                "canonical_event_sha256": _canonical_json_sha256(event),
            },
            "configuration": {
                "engine_version": "5.0.0-predictive-research",
                "training_window": 500,
                "ridge_alpha": 1.0,
                "pair_strength": 0.5,
                "seed": 5050,
                "random_seed": 18861,
                "model_ticket_count": 1,
                "random_ticket_count": 1,
                "probability_count": 45,
            },
        }
        result = verify_1251_archive_bytes(archive, manifest)
        self.assertTrue(result["applicable"])
        self.assertTrue(result["matched"])
        self.assertTrue(result["canonical_event_sha256_matches"])
        self.assertTrue(result["history_fingerprint_matches"])
        self.assertTrue(result["configuration_matches"])

        changed = json.loads(archive.decode("utf-8"))
        changed["events"][0]["payload"]["tickets"][0]["n1"] = 45
        changed_bytes = json.dumps(changed, ensure_ascii=False).encode("utf-8")
        changed_result = verify_1251_archive_bytes(changed_bytes, manifest)
        self.assertFalse(changed_result["matched"])

        # Keep the committed event byte-for-byte equivalent while changing
        # the archive envelope: event-only matching must not pass full match.
        envelope_changed = json.loads(archive.decode("utf-8"))
        envelope_changed["extra_uncommitted_metadata"] = {"reviewed": True}
        envelope_bytes = json.dumps(envelope_changed, ensure_ascii=False).encode("utf-8")
        envelope_result = verify_1251_archive_bytes(envelope_bytes, manifest)
        self.assertTrue(envelope_result["canonical_event_sha256_matches"])
        self.assertFalse(envelope_result["raw_file_sha256_matches"])
        self.assertFalse(envelope_result["canonical_archive_sha256_matches"])
        self.assertFalse(envelope_result["matched"])


if __name__ == "__main__":
    unittest.main()
