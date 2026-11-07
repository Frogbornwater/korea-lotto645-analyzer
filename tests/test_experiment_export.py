import json
import sys
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_export import build_research_bundle
from model_management import new_project_state


def sample_history(n=24):
    rows = []
    for draw in range(1, n + 1):
        nums = sorted(((draw * 3 + j * 7) % 45) + 1 for j in range(6))
        # deterministic fallback if modular arithmetic collides
        nums = sorted(set(nums))
        while len(nums) < 6:
            candidate = (nums[-1] % 45) + 1
            if candidate not in nums:
                nums.append(candidate)
                nums.sort()
        bonus = next(x for x in range(1, 46) if x not in nums)
        rows.append({"draw_no": draw, "date": f"2025-01-{(draw-1)%28+1:02d}",
                     **{f"n{i+1}": nums[i] for i in range(6)}, "bonus": bonus})
    return pd.DataFrame(rows)


class ExperimentExportTests(unittest.TestCase):
    def test_bundle_contains_allowlisted_session_artifacts(self):
        history = sample_history()
        state = new_project_state()
        session = {
            "v36_bt_raw": pd.DataFrame([{"draw_no": 20, "号码": "1,2,3,4,5,6", "命中数": 2}]),
            "v36_bt_summary": pd.DataFrame([{"策略": "随机", "平均命中数": 0.8}]),
            "v36_bt_metadata": {"engine_version": "3.6.1", "seed": 2026},
            "v361_param_summary": pd.DataFrame([{"window": 50, "score": 0.81}]),
            "v361_param_meta": {"best_config": {"recent_window": 50}},
            "v361_portfolio_report": {"不同号码数": 39},
            "secret_api_token": "must-not-export",
            "unrelated_widget_state": {"some": "value"},
            "v37_joint_result": {
                "search_summary": pd.DataFrame([{"配置编号": 1, "模型": "贝叶斯平滑模型"}]),
                "pareto_front": pd.DataFrame([{"配置编号": 1}]),
                "inner_period_detail": pd.DataFrame([{"期号": 20}]),
                "inner_fold_detail": pd.DataFrame([{"内层折": 1}]),
                "outer_detail": pd.DataFrame([{"期号": 21}]),
                "outer_selections": pd.DataFrame([{"外层折": 1}]),
                "outer_summary": pd.DataFrame([{"指标": "平均命中差_vs随机"}]),
                "holdout_detail": pd.DataFrame([{"期号": 22}]),
                "holdout_ticket_detail": pd.DataFrame([{"号码": "1-2-3-4-5-6"}]),
                "holdout_summary": pd.DataFrame([{"指标": "平均命中差_vs随机"}]),
                "selected_config": {"config_id": 1},
                "metadata": {"experiment_type": "v37_joint_nested_automl"},
            },
            "v38_robust_result": {
                "screen_summary": pd.DataFrame([{"配置编号": 1}]),
                "finalist_summary": pd.DataFrame([{"配置编号": 1}]),
                "predictive_pareto_front": pd.DataFrame([{"配置编号": 1}]),
                "structure_pareto_front": pd.DataFrame([{"配置编号": 2}]),
                "outer_detail": pd.DataFrame([{"期号": 30}]),
                "outer_selections": pd.DataFrame([{"外层折": 1}]),
                "outer_summary": pd.DataFrame([{"指标": "平均命中差_vs随机"}]),
                "screen_period_detail": pd.DataFrame([{"期号": 10}]),
                "screen_fold_detail": pd.DataFrame([{"内层折": 1}]),
                "finalist_period_detail": pd.DataFrame([{"期号": 10}]),
                "finalist_fold_detail": pd.DataFrame([{"内层折": 1}]),
                "selected_config": {"config_id": 1},
                "metadata": {"experiment_type": "v38_robust_dual_objective_research"},
            },
            "v38_forward_locks": {
                "31": {
                    "detail": pd.DataFrame([{"目标期号": 31, "n1": 1}]),
                    "metadata": {"target_draw": 31, "history_last_draw": 30},
                },
            },
        }
        payload, manifest = build_research_bundle(
            history, {"rows": len(history), "range_complete": True},
            "manual upload: test.csv", session, state, include_history=True,
        )
        with zipfile.ZipFile(BytesIO(payload)) as zf:
            names = set(zf.namelist())
            self.assertIn("manifest.json", names)
            self.assertIn("history/cleaned_manual_history.csv", names)
            self.assertIn("backtest/rolling_backtest_detail.csv", names)
            self.assertIn("parameter_search/parameter_search_ranking.csv", names)
            self.assertIn("portfolio/portfolio_quality_report.json", names)
            self.assertIn("joint_optimization_v37/search_ranking.csv", names)
            self.assertIn("joint_optimization_v37/holdout_tickets.csv", names)
            self.assertIn("robust_research_v38/predictive_pareto_front.csv", names)
            self.assertIn("robust_research_v38/structure_pareto_front.csv", names)
            self.assertIn("forward_locks/draw_31/locked_tickets.csv", names)
            self.assertNotIn("secret_api_token", "\n".join(names))
            self.assertNotIn("must-not-export", zf.read("README_CN.txt").decode("utf-8"))
            stored_manifest = json.loads(zf.read("manifest.json"))
            self.assertEqual(stored_manifest["history_rows"], len(history))
            self.assertEqual(stored_manifest["network_policy"], "no_lottery_official_site_api_scraping_or_external_draw_fetching")
        self.assertIn("history/cleaned_manual_history.csv", manifest["included_artifacts"])
        self.assertIn("v361_ensemble_raw", manifest["unavailable_or_empty_artifacts"])

    def test_bundle_can_omit_history(self):
        history = sample_history()
        payload, manifest = build_research_bundle(
            history, {"rows": len(history)}, "manual", {}, None, include_history=False,
        )
        with zipfile.ZipFile(BytesIO(payload)) as zf:
            names = set(zf.namelist())
            self.assertNotIn("history/cleaned_manual_history.csv", names)
            stored_manifest = json.loads(zf.read("manifest.json"))
            self.assertFalse(stored_manifest["history_included"])
        self.assertFalse(manifest["history_included"])

    def test_json_artifacts_are_serialized_safely(self):
        history = sample_history()
        payload, _ = build_research_bundle(
            history, {"metric": float("nan"), "last_date": pd.Timestamp("2025-01-01")},
            "manual", {"v36_bt_metadata": {"value": float("nan")}}, None,
        )
        with zipfile.ZipFile(BytesIO(payload)) as zf:
            meta = json.loads(zf.read("manifest.json"))
            self.assertIsNone(meta["quality_report"]["metric"])
            exp_meta = json.loads(zf.read("backtest/rolling_backtest_metadata.json"))
            self.assertIsNone(exp_meta["value"])

    def test_bundle_handles_dataframe_and_json_missing_cleanly(self):
        history = sample_history()
        payload, manifest = build_research_bundle(history, {}, "manual", {}, None)
        self.assertTrue(payload.startswith(b"PK"))
        self.assertGreaterEqual(len(manifest["unavailable_or_empty_artifacts"]), 10)


if __name__ == "__main__":
    unittest.main()
