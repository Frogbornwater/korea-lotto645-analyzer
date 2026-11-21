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
    aggregate_v55_temperature_forward_evidence,
    build_v55_temperature_forward_event,
    check_frozen_base_history,
    evaluate_v55_temperature_forward_events,
    load_commitment,
    verify_commitment,
    verify_hash_only_manifest,
)


class V55TemperatureForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = validate_frame(pd.read_csv(ROOT / "data" / "lotto645.csv"))
        # Keep this fixture anchored to the committed 1-1250 V5.5 prefix.
        # The repository CSV now includes the actual user-provided draw 1251.
        base = base.loc[base["draw_no"] <= 1250].copy()
        # Synthetic draw 1251 is used only inside tests. It is never written to
        # data/lotto645.csv and is not presented as a real lottery result.
        synthetic_1251 = pd.DataFrame([{
            "draw_no": 1251, "date": "2099-01-01",
            "n1": 3, "n2": 14, "n3": 22, "n4": 30, "n5": 35, "n6": 44,
            "bonus": 2,
        }])
        cls.history_1251 = validate_frame(pd.concat([base, synthetic_1251], ignore_index=True))
        cls.commitment = load_commitment()

    def test_multi_draw_gate_requires_100_forward_draws(self):
        rows = []
        for draw in range(1252, 1332):
            rows.append({
                "目标期号": draw,
                "V5原始概率平均命中": 0.80,
                "V5.5校准平均命中": 0.85,
                "随机平均命中": 0.80,
                "V5原始概率单注3+率": 0.02,
                "V5.5校准单注3+率": 0.04,
                "随机单注3+率": 0.02,
                "V5原始概率整批3+": 0.20,
                "V5.5校准整批3+": 0.30,
                "随机整批3+": 0.20,
                "V5原始Brier": 0.1160,
                "V5.5校准Brier": 0.1150,
                "随机Brier": 0.1158,
                "V5原始LogLoss": 0.3930,
                "V5.5校准LogLoss": 0.3920,
                "随机LogLoss": 0.3928,
            })
        frame = pd.DataFrame(rows)
        report, gate = aggregate_v55_temperature_forward_evidence(
            frame, minimum_draws=100, repetitions=1000, block_length=3, seed=7755
        )
        self.assertEqual(len(report), 10)
        self.assertFalse(gate["second_independent_replication_allowed"])
        self.assertFalse(gate["production_model_change_allowed"])
        self.assertIn("至少需要100期", gate["status"])

    def test_multi_draw_gate_only_opens_second_research_stage(self):
        rows = []
        for draw in range(1252, 1372):
            rows.append({
                "目标期号": draw,
                "V5原始概率平均命中": 0.80,
                "V5.5校准平均命中": 0.85,
                "随机平均命中": 0.80,
                "V5原始概率单注3+率": 0.02,
                "V5.5校准单注3+率": 0.04,
                "随机单注3+率": 0.02,
                "V5原始概率整批3+": 0.20,
                "V5.5校准整批3+": 0.30,
                "随机整批3+": 0.20,
                "V5原始Brier": 0.1160,
                "V5.5校准Brier": 0.1150,
                "随机Brier": 0.1158,
                "V5原始LogLoss": 0.3930,
                "V5.5校准LogLoss": 0.3920,
                "随机LogLoss": 0.3928,
            })
        report, gate = aggregate_v55_temperature_forward_evidence(
            pd.DataFrame(rows), minimum_draws=100, repetitions=1000, block_length=3, seed=7756
        )
        self.assertEqual(len(report), 10)
        self.assertTrue(gate["calibration_improvement_supported_vs_raw_v5"])
        self.assertTrue(gate["second_independent_replication_allowed"])
        self.assertFalse(gate["production_model_change_allowed"])
        self.assertFalse(gate["prediction_superiority_proven"])

    def test_frozen_commitment_is_exact_and_research_only(self):
        result = verify_commitment(self.commitment)
        self.assertTrue(result["valid"])
        self.assertEqual(self.commitment["temperature_lock"]["temperature"], 2.875)
        self.assertFalse(self.commitment["application_rule"]["production_model_changed"])

    def test_imported_1251_allows_only_a_1252_temperature_research_lock(self):
        actual_history = pd.read_csv(ROOT / "data" / "lotto645.csv")
        audit = check_frozen_base_history(actual_history, self.commitment)
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["history_last_draw"], 1251)
        event = build_v55_temperature_forward_event(
            actual_history, manifest=self.commitment, ticket_count=10, master_seed=77551252
        )
        self.assertEqual(event["payload"]["metadata"]["target_draw"], 1252)

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
        verified = verify_hash_only_manifest(event, manifest)
        self.assertTrue(verified["verified"])
        tampered = __import__("json").loads(__import__("json").dumps(event))
        original_n1 = int(tampered["payload"]["tickets"][0]["n1"])
        tampered["payload"]["tickets"][0]["n1"] = 45 if original_n1 != 45 else 44
        self.assertFalse(verify_hash_only_manifest(tampered, manifest)["verified"])

    def test_evaluator_rejects_modified_temperature_probabilities(self):
        event = build_v55_temperature_forward_event(
            self.history_1251, manifest=self.commitment, ticket_count=10, master_seed=222222
        )
        event["payload"]["number_probabilities"][0]["temperature_probability"] += 0.001
        synthetic_1252 = pd.DataFrame([{
            "draw_no": 1252, "date": "2099-01-08",
            "n1": 7, "n2": 12, "n3": 19, "n4": 26, "n5": 34, "n6": 41,
            "bonus": 44,
        }])
        completed = validate_frame(pd.concat([self.history_1251, synthetic_1252], ignore_index=True))
        summary, detail, status = evaluate_v55_temperature_forward_events(completed, [event])
        self.assertTrue(summary.empty)
        self.assertTrue(detail.empty)
        self.assertIn("概率记录无效", status.iloc[0]["状态"])

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
