import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from joint_optimizer import (
    OBJECTIVE_COLUMNS,
    RANDOM_MODEL,
    SEARCH_MODELS,
    _pareto_mask,
    generate_joint_configs,
    run_joint_optimization,
)
from lotto_core import COLUMNS, MODEL_NAMES
from model_management import add_experiment, new_project_state


def synthetic_history(draws=150, seed=7307):
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(1, draws + 1):
        nums = sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False))
        bonus = int(rng.choice(sorted(set(range(1, 46)) - set(nums))))
        rows.append({"draw_no": draw, "date": f"2025-01-{(draw-1)%28+1:02d}",
                     **{f"n{i}": nums[i-1] for i in range(1, 7)}, "bonus": bonus})
    return pd.DataFrame(rows, columns=COLUMNS)


class JointOptimizerTests(unittest.TestCase):
    def test_eight_models_are_in_scope_with_random_control(self):
        self.assertEqual(len(MODEL_NAMES), 8)
        self.assertEqual(len(SEARCH_MODELS), 7)
        self.assertIn(RANDOM_MODEL, MODEL_NAMES)

    def test_budget_500_creates_exactly_500_unique_configs_and_covers_all_models(self):
        configs = generate_joint_configs(500, seed=2026)
        self.assertEqual(len(configs), 500)
        self.assertEqual(len({(x["model_name"], str(sorted(x["params"].items()))) for x in configs}), 500)
        counts = {name: sum(x["model_name"] == name for x in configs) for name in SEARCH_MODELS}
        self.assertEqual(set(counts), set(SEARCH_MODELS))
        self.assertEqual(sum(counts.values()), 500)
        self.assertTrue(all(value in (71, 72) for value in counts.values()))

    def test_config_generation_is_reproducible(self):
        self.assertEqual(generate_joint_configs(21, 91), generate_joint_configs(21, 91))

    def test_pareto_front_dominance(self):
        values = np.array([[1, 1], [0, 0], [2, 0], [0, 2]], dtype=float)
        mask = _pareto_mask(values)
        self.assertEqual(mask.tolist(), [True, False, True, True])

    def test_nested_run_keeps_holdout_separate_and_includes_baseline(self):
        history = synthetic_history(150)
        result = run_joint_optimization(
            history,
            budget=14,
            tickets_per_draw=10,
            n_seeds=3,
            minimum_train=50,
            holdout_draws=30,
            outer_folds=2,
            outer_validation_draws=5,
            inner_folds=2,
            inner_validation_draws=4,
            seed=2026,
            objective="balanced",
        )
        meta = result["metadata"]
        self.assertEqual(meta["objective"], "balanced")
        self.assertTrue(meta["holdout_consumed"])
        self.assertEqual(meta["holdout_draws"], 30)
        self.assertEqual(meta["seed_groups"], 3)
        self.assertEqual(meta["tickets_per_draw_per_model"], 10)
        self.assertEqual(len(result["search_summary"]) - 1, 14)  # plus one control row
        self.assertIn(RANDOM_MODEL, set(result["search_summary"]["模型"]))
        self.assertTrue((result["inner_period_detail"]["期号"] < meta["holdout_start_draw"]).all())
        self.assertTrue((result["holdout_detail"]["期号"] >= meta["holdout_start_draw"]).all())
        self.assertEqual(result["outer_selections"].shape[0], 2)
        self.assertEqual(result["selected_config"]["model_name"] in SEARCH_MODELS, True)
        self.assertTrue(set(OBJECTIVE_COLUMNS).issubset(result["search_summary"].columns))

    def test_holdout_ticket_rows_are_legal_and_seed_paired(self):
        result = run_joint_optimization(
            synthetic_history(140), budget=14, tickets_per_draw=10, n_seeds=3,
            minimum_train=45, holdout_draws=30, outer_folds=2,
            outer_validation_draws=5, inner_folds=2, inner_validation_draws=3,
            seed=77,
        )
        tickets = result["holdout_ticket_detail"]
        self.assertEqual(set(tickets["策略"]), {result["selected_config"]["model_name"], RANDOM_MODEL})
        self.assertTrue(tickets[[f"n{i}" for i in range(1, 7)]].ge(1).all().all())
        self.assertTrue(tickets[[f"n{i}" for i in range(1, 7)]].le(45).all().all())
        self.assertEqual(len(tickets), 30 * 3 * 10 * 2)

    def test_rejects_gaps(self):
        frame = synthetic_history(150).drop(index=70).reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "缺期"):
            run_joint_optimization(
                frame, budget=14, tickets_per_draw=10, n_seeds=3,
                minimum_train=40, holdout_draws=30, outer_folds=2,
                outer_validation_draws=5, inner_folds=2, inner_validation_draws=3,
            )

    def test_rejects_too_small_holdout_or_invalid_objective(self):
        with self.assertRaisesRegex(ValueError, "至少需要30期"):
            run_joint_optimization(
                synthetic_history(150), budget=14, tickets_per_draw=10, n_seeds=3,
                minimum_train=40, holdout_draws=10, outer_folds=2,
                outer_validation_draws=5, inner_folds=2, inner_validation_draws=3,
            )
        with self.assertRaisesRegex(ValueError, "balanced"):
            run_joint_optimization(
                synthetic_history(150), budget=14, tickets_per_draw=10, n_seeds=3,
                minimum_train=40, holdout_draws=30, outer_folds=2,
                outer_validation_draws=5, inner_folds=2, inner_validation_draws=3,
                objective="mean",
            )


    def test_joint_experiment_archive_records_consumed_holdout_without_certifying_one_model(self):
        state = new_project_state()
        meta = {
            "experiment_id": "joint-test-1",
            "engine_version": "3.7.0-joint",
            "data_fingerprint": "abc123",
            "holdout_evaluated": True,
            "holdout_consumed": True,
            "holdout_start_draw": 901,
            "holdout_end_draw": 1000,
        }
        updated = add_experiment(state, "V3.7全模型联合优化", meta, [])
        record = updated["experiments"][-1]
        self.assertTrue(record["holdout_evaluated"])
        self.assertEqual(record["data_fingerprint"], "abc123")
        self.assertIsNone(record["model_name"])
        # A joint experiment is not model-specific evidence and cannot certify
        # any one strategy as independently verified.
        from model_management import change_model_status
        with self.assertRaises(ValueError):
            change_model_status(updated, SEARCH_MODELS[0], "验证通过", "joint test", "joint-test-1")

    def test_no_network_access_in_joint_module(self):
        source = (ROOT / "joint_optimizer.py").read_text(encoding="utf-8").lower()
        for token in ["http://", "https://", "dhlottery", "requests.get", "urllib.request", "httpx"]:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
