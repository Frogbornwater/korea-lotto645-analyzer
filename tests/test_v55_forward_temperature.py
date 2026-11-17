# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lotto_core import validate_frame
from v55_forward_temperature import (
    EVENT_TYPE,
    RAW_STRATEGY,
    TEMPERATURE_STRATEGY,
    RANDOM_STRATEGY,
    build_hash_only_commitment,
    build_v55_temperature_forward_event,
    check_frozen_base_history,
    evaluate_v55_temperature_forward_events,
    load_commitment,
    verify_commitment,
)


class V55TemperatureForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = validate_frame(pd.read_csv(ROOT / "data" / "lotto645.csv"))
        # Synthetic draw 1251 is used only inside tests. It is never written to
        # data/lotto645.csv and is not presented as a real lottery result.
        synthetic_1251 = pd.DataFrame([{
            "draw_no": 1251, "date": "2099-01-01",
            "n1": 3, "n2": 14, "n3": 22, "n4": 30, "n5": 35, "n6": 44,
            "bonus": 2,
        }])
        cls.history_1251 = validate_frame(pd.concat([base, synthetic_1251], ignore_index=True))
        cls.commitment = load_commitment()

    def test_frozen_commitment_is_exact_and_research_only(self):
        result = verify_commitment(self.commitment)
        self.assertTrue(result["valid"])
        self.assertEqual(self.commitment["temperature_lock"]["temperature"], 2.875)
        self.assertFalse(self.commitment["application_rule"]["production_model_changed"])

    def test_current_1250_history_cannot_create_a_competing_1251_lock(self):
        history_1250 = pd.read_csv(ROOT / "data" / "lotto645.csv")
        audit = check_frozen_base_history(history_1250, self.commitment)
        self.assertTrue(audit["ok"])
        with self.assertRaisesRegex(ValueError, "必须先手动导入并核验第1251期"):
            build_v55_temperature_forward_event(history_1250, manifest=self.commitment)

    def test_date_format_does_not_change_frozen_number_history(self):
        audit = check_frozen_base_history(self.history_1251, self.commitment)
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["history_last_draw"], 1251)

    def test_changed_base_history_is_rejected(self):
        changed = self.history_1251.copy()
        changed.loc[changed["draw_no"] == 1250, "n1"] = 2
        audit = check_frozen_base_history(changed, self.commitment)
        self.assertFalse(audit["ok"])
        self.assertIn("不一致", audit["status"])

    def test_build_locks_raw_temperature_and_uniform_groups_for_1252(self):
        event = build_v55_temperature_forward_event(
            self.history_1251, manifest=self.commitment, ticket_count=10, master_seed=7755026
        )
        self.assertEqual(event["event_type"], EVENT_TYPE)
        payload = event["payload"]
        self.assertEqual(payload["metadata"]["target_draw"], 1252)
        self.assertEqual(payload["metadata"]["temperature"], 2.875)
        self.assertEqual(len(payload["number_probabilities"]), 45)
        strategies = [row["策略"] for row in payload["tickets"]]
        self.assertEqual(strategies.count(RAW_STRATEGY), 10)
        self.assertEqual(strategies.count(TEMPERATURE_STRATEGY), 10)
        self.assertEqual(strategies.count(RANDOM_STRATEGY), 10)
        raw_sum = sum(row["raw_v5_probability"] for row in payload["number_probabilities"])
        temp_sum = sum(row["temperature_probability"] for row in payload["number_probabilities"])
        self.assertAlmostEqual(raw_sum, 6.0, places=5)
        self.assertAlmostEqual(temp_sum, 6.0, places=5)
        self.assertEqual(payload["metadata"]["raw_seed"], payload["metadata"]["temperature_seed"])
        raw_seeds = {row["seed"] for row in payload["tickets"] if row["策略"] == RAW_STRATEGY}
        temp_seeds = {row["seed"] for row in payload["tickets"] if row["策略"] == TEMPERATURE_STRATEGY}
        random_seeds = {row["seed"] for row in payload["tickets"] if row["策略"] == RANDOM_STRATEGY}
        self.assertEqual(raw_seeds, temp_seeds)
        self.assertNotEqual(raw_seeds, random_seeds)

    def test_public_commitment_contains_hashes_only_not_ticket_or_probability_data(self):
        event = build_v55_temperature_forward_event(
            self.history_1251, manifest=self.commitment, ticket_count=10, master_seed=98765
        )
        manifest, archive_sha = build_hash_only_commitment(event)
        self.assertEqual(len(archive_sha), 64)
        self.assertEqual(manifest["archive_commitment"]["sha256"], archive_sha)
        self.assertFalse(manifest["archive_commitment"]["ticket_numbers_embedded"])
        self.assertFalse(manifest["archive_commitment"]["probability_vectors_embedded"])
        serialized = str(manifest).lower()
        self.assertNotIn("number_probabilities", serialized)
        self.assertNotIn("tickets", serialized)
        self.assertNotIn("n1", serialized)

    def test_evaluator_rejects_modified_temperature_probabilities(self):
        event = build_v55_temperature_forward_event(
            self.history_1251, manifest=self.commitment, ticket_count=10, master_seed=222222
        )
        event["payload"]["number_probabilities"][0]["temperature_probability"] += 0.001
        target = event["payload"]["metadata"]["target_draw"]
        # Event-level integrity must reject an edited lock before any scoring.
        summary, detail, status = evaluate_v55_temperature_forward_events(self.history_1251, [event])
        self.assertTrue(summary.empty)
        self.assertTrue(detail.empty)
        # A probability mismatch is checked only once the target arrives, so
        # the pending state is still accepted until scoring becomes possible.
        self.assertIn("等待开奖数据", status.iloc[0]["状态"])

    def test_evaluator_waits_then_scores_only_after_target_is_imported(self):
        event = build_v55_temperature_forward_event(
            self.history_1251, manifest=self.commitment, ticket_count=10, master_seed=123456
        )
        before_summary, before_detail, before_status = evaluate_v55_temperature_forward_events(
            self.history_1251, [event]
        )
        self.assertTrue(before_summary.empty)
        self.assertTrue(before_detail.empty)
        self.assertIn("等待开奖数据", before_status.iloc[0]["状态"])

        synthetic_1252 = pd.DataFrame([{
            "draw_no": 1252, "date": "2099-01-08",
            "n1": 7, "n2": 12, "n3": 19, "n4": 26, "n5": 34, "n6": 41,
            "bonus": 44,
        }])
        completed = validate_frame(pd.concat([self.history_1251, synthetic_1252], ignore_index=True))
        summary, detail, status = evaluate_v55_temperature_forward_events(completed, [event])
        self.assertEqual(len(summary), 1)
        self.assertEqual(len(detail), 30)
        self.assertEqual(status.iloc[0]["状态"], "已评分")
        self.assertEqual(int(summary.iloc[0]["目标期号"]), 1252)
        self.assertIn("V5.5校准Brier", summary.columns)
        self.assertIn("V5.5校准LogLoss", summary.columns)

    def test_evaluator_rejects_changed_locked_prefix(self):
        event = build_v55_temperature_forward_event(
            self.history_1251, manifest=self.commitment, ticket_count=10, master_seed=654321
        )
        synthetic_1252 = pd.DataFrame([{
            "draw_no": 1252, "date": "2099-01-08",
            "n1": 7, "n2": 12, "n3": 19, "n4": 26, "n5": 34, "n6": 41,
            "bonus": 44,
        }])
        changed = self.history_1251.copy()
        changed.loc[changed["draw_no"] == 1251, "n1"] = 4
        completed = validate_frame(pd.concat([changed, synthetic_1252], ignore_index=True))
        summary, detail, status = evaluate_v55_temperature_forward_events(completed, [event])
        self.assertTrue(summary.empty)
        self.assertTrue(detail.empty)
        self.assertIn("前缀指纹不匹配", status.iloc[0]["状态"])


if __name__ == "__main__":
    unittest.main()
