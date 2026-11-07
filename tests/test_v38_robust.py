from __future__ import annotations

import sys
import unittest
import io
import json
import zipfile
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robust_optimizer import (
    build_forward_lock,
    evaluate_forward_locks,
    generate_robust_configs,
    make_spread_inner_folds,
    run_robust_research,
    restore_forward_locks_from_bundle,
)
from model_registry import ENGINE_VERSION


def synthetic_history(draws: int = 180, seed: int = 4408) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(1, draws + 1):
        numbers = sorted(int(x) for x in rng.choice(np.arange(1, 46), size=6, replace=False))
        available = sorted(set(range(1, 46)) - set(numbers))
        bonus = int(rng.choice(available))
        rows.append({
            "draw_no": draw,
            "date": f"2024-01-{(draw - 1) % 28 + 1:02d}",
            **{f"n{i}": numbers[i - 1] for i in range(1, 7)},
            "bonus": bonus,
        })
    return pd.DataFrame(rows)


class V38RobustResearchTests(unittest.TestCase):
    def test_engine_version_updated(self):
        self.assertEqual(ENGINE_VERSION, "3.8.0-robust")

    def test_default_run_configuration_matches_real_data_benchmark(self):
        params = inspect.signature(run_robust_research).parameters
        self.assertEqual(params["budget"].default, 500)
        self.assertEqual(params["outer_folds"].default, 3)
        self.assertEqual(params["outer_validation_draws"].default, 12)
        self.assertEqual(params["inner_folds"].default, 3)
        self.assertEqual(params["inner_validation_draws"].default, 6)
        self.assertEqual(params["finalists_per_model"].default, 1)
        self.assertEqual(params["screening_folds"].default, 3)
        self.assertEqual(params["screening_window_draws"].default, 4)
        self.assertEqual(params["finalist_seeds"].default, 3)
        self.assertEqual(params["retrospective_cutoff_draw"].default, 1088)

    def test_v37_reviewed_range_is_persistently_protected(self):
        ledger_path = ROOT / "data" / "experiment_ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        reviewed = ledger["known_reviewed_ranges"]
        self.assertTrue(any(int(x["start_draw"]) == 1089 and int(x["end_draw"]) == 1248 for x in reviewed))
        self.assertEqual(ledger["policy"], "offline_manual_history_only")

    def test_latin_hypercube_config_generation_is_unique_and_balanced(self):
        configs = generate_robust_configs(500, seed=2028)
        self.assertEqual(len(configs), 500)
        self.assertEqual(len({x["config_id"] for x in configs}), 500)
        counts = pd.Series([x["model_name"] for x in configs]).value_counts()
        self.assertEqual(len(counts), 7)
        self.assertTrue(counts.max() - counts.min() <= 1)
        first = generate_robust_configs(35, seed=99)
        second = generate_robust_configs(35, seed=99)
        self.assertEqual([x["params"] for x in first], [x["params"] for x in second])

    def test_spread_folds_are_non_overlapping_and_chronological(self):
        folds = make_spread_inner_folds(900, min_train=300, inner_folds=4, inner_validation_draws=8)
        self.assertEqual(len(folds), 4)
        windows = [indices for _, indices in folds]
        for indices in windows:
            self.assertEqual(len(indices), 8)
            self.assertGreaterEqual(indices[0], 300)
        for left, right in zip(windows, windows[1:]):
            self.assertLess(left[-1], right[0])
        self.assertGreater(windows[-1][0] - windows[0][0], 400)

    def test_rejects_insufficient_prefix(self):
        with self.assertRaisesRegex(ValueError, "至少需要"):
            make_spread_inner_folds(320, min_train=300, inner_folds=4, inner_validation_draws=8)

    def test_run_robust_research_separates_predictive_and_structure_objectives(self):
        frame = synthetic_history(180)
        result = run_robust_research(
            frame,
            budget=14,
            tickets_per_draw=3,
            minimum_train=80,
            outer_folds=2,
            outer_validation_draws=8,
            inner_folds=3,
            inner_validation_draws=3,
            finalists_per_model=1,
            screening_seeds=1,
            finalist_seeds=2,
            seed=77,
            retrospective_cutoff_draw=180,
        )
        self.assertEqual(len(result["screen_summary"]), 14)
        self.assertEqual(len(result["finalist_summary"]), 7)
        self.assertEqual(len(result["outer_detail"]), 16)
        self.assertEqual(len(result["outer_selections"]), 2)
        self.assertFalse(result["metadata"]["prospective_holdout_used"])
        self.assertIn("尚无预测优势证据", result["metadata"]["predictive_advantage_status"])
        self.assertIn("预测综合分", result["finalist_summary"].columns)
        self.assertIn("结构综合分", result["finalist_summary"].columns)
        self.assertFalse(set(["号码覆盖差_vs随机", "组间重号减少_vs随机"]) & set(
            result["metadata"]["predictive_objectives"]
        ))

    def test_forward_lock_and_later_score(self):
        frame = synthetic_history(300)
        config = {
            "config_id": 5,
            "model_name": "贝叶斯平滑模型",
            "model_version": "1.1",
            "params": {"bayes_prior": 25.0},
        }
        locked, meta = build_forward_lock(frame, config, tickets_per_draw=4, seed_groups=2, seed=123)
        self.assertEqual(meta["target_draw"], 301)
        self.assertEqual(len(locked), 16)
        self.assertTrue((locked["目标期号"] == 301).all())
        next_row = synthetic_history(301).tail(1)
        updated = pd.concat([frame, next_row], ignore_index=True)
        detail, summary = evaluate_forward_locks(updated, {"301": {"detail": locked, "metadata": meta}})
        self.assertEqual(len(detail), 16)
        self.assertEqual(len(summary), 4)
        self.assertEqual(set(summary["策略"]), {"贝叶斯平滑模型", "均匀随机基准"})
        # A lock cannot be credited against a draw that was already in its history.
        bad_meta = dict(meta, history_last_draw=301)
        bad_detail, bad_summary = evaluate_forward_locks(
            updated, {"301": {"detail": locked, "metadata": bad_meta}}
        )
        self.assertTrue(bad_detail.empty)
        self.assertTrue(bad_summary.empty)


    def test_restore_forward_locks_from_research_bundle(self):
        frame = synthetic_history(300)
        config = {
            "config_id": 5,
            "model_name": "贝叶斯平滑模型",
            "model_version": "1.1",
            "params": {"bayes_prior": 25.0},
        }
        locked, meta = build_forward_lock(frame, config, tickets_per_draw=2, seed_groups=1, seed=123)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("forward_locks/draw_301/locked_tickets.csv", locked.to_csv(index=False))
            zf.writestr("forward_locks/draw_301/lock_metadata.json", json.dumps(meta))
            zf.writestr("secrets/unrelated.txt", "must be ignored")
        restored = restore_forward_locks_from_bundle(buf.getvalue())
        self.assertEqual(set(restored), {"301"})
        self.assertEqual(len(restored["301"]["detail"]), 4)
        self.assertEqual(restored["301"]["metadata"]["target_draw"], 301)
        with self.assertRaisesRegex(ValueError, "有效的研究 ZIP"):
            restore_forward_locks_from_bundle(b"not a zip")

    def test_robust_source_has_no_network_or_official_site_integration(self):
        source = (ROOT / "robust_optimizer.py").read_text(encoding="utf-8").lower()
        for token in ["requests", "httpx", "urllib", "dhlottery", "dhlottery.co.kr", "http://", "https://"]:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
