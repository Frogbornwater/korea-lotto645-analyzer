import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest_engine import _make_weights
from lotto_core import COLUMNS, MODEL_NAMES, model_weights
from model_management import (
    add_experiment, change_model_status, export_project_state,
    import_project_state, new_project_state,
)
from parameter_optimizer import (
    default_parameter_grid, evaluate_reserved_holdout, run_parameter_search,
)
from portfolio_optimizer import optimize_portfolio


def synthetic_history(draws=160, seed=609):
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(1, draws + 1):
        numbers = sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False))
        bonus = int(rng.choice(sorted(set(range(1, 46)) - set(numbers))))
        rows.append({
            "draw_no": draw, "date": f"2025-01-{(draw - 1) % 28 + 1:02d}",
            **{f"n{i}": numbers[i - 1] for i in range(1, 7)}, "bonus": bonus,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


class MultiModelTests(unittest.TestCase):
    def setUp(self):
        self.history = synthetic_history()

    def test_multi_model_weight_definition_matches_core(self):
        numbers = self.history[[f"n{i}" for i in range(1, 7)]].to_numpy(dtype=np.int16)
        expected = _make_weights(
            numbers, "多模型集成模型", 45, 30.0, 60.0,
            (0.4, 0.35, 0.25), (0.2, 0.5, 0.3),
        )
        actual = model_weights(
            self.history, "多模型集成模型", window=45, half_life=30.0,
            bayes_prior=60.0, multi_model_weights=(0.2, 0.5, 0.3),
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_tunable_ensemble_weights_match_backtest_engine(self):
        numbers = self.history[[f"n{i}" for i in range(1, 7)]].to_numpy(dtype=np.int16)
        chosen = (0.2, 0.5, 0.3)
        expected = _make_weights(numbers, "综合评分模型", 60, 40.0, 80.0, chosen)
        actual = model_weights(self.history, "综合评分模型", ensemble_weights=chosen)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_registry_includes_multi_model_ensemble(self):
        self.assertIn("多模型集成模型", MODEL_NAMES)


class ParameterOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.history = synthetic_history(180)

    def test_search_reserves_latest_draws(self):
        detail, summary, metadata = run_parameter_search(
            self.history, "贝叶斯平滑模型", min_train=60,
            validation_draws=20, holdout_draws=25,
            tickets_per_draw=3, n_seeds=1, seed=88,
            parameter_grid=[{"bayes_prior": 40}, {"bayes_prior": 80}],
        )
        self.assertEqual(len(summary), 2)
        self.assertGreater(len(detail), 0)
        self.assertTrue((detail["期数"] < metadata["reserved_holdout_start_draw"]).all())
        self.assertEqual(metadata["reserved_holdout_draws"], 25)
        self.assertFalse(metadata["holdout_evaluated"])
        self.assertLess(metadata["validation_end_draw"], metadata["reserved_holdout_start_draw"])
        self.assertIn("bayes_prior", metadata["best_config"])

    def test_parameter_search_reproducible(self):
        args = dict(model_name="近期频率模型", min_train=60, validation_draws=15,
                    holdout_draws=25, tickets_per_draw=2, n_seeds=1, seed=101,
                    parameter_grid=[{"recent_window": 30}, {"recent_window": 50}])
        detail_a, first, _ = run_parameter_search(self.history, **args)
        detail_b, second, _ = run_parameter_search(self.history, **args)
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(detail_a, detail_b)

    def test_fixed_holdout_uses_frozen_parameters_and_random_control(self):
        raw, summary, meta = evaluate_reserved_holdout(
            self.history, "贝叶斯平滑模型", {"bayes_prior": 80},
            min_train=60, holdout_draws=20, tickets_per_draw=2,
            n_seeds=1, seed=9,
        )
        self.assertEqual(set(summary["策略"]), {"贝叶斯平滑模型", "均匀随机基准"})
        self.assertTrue(meta["holdout_evaluated"])
        self.assertTrue(meta["holdout_consumed"])
        self.assertTrue((raw["训练截止期号"] < raw["期数"]).all())

    def test_default_grids_exist_for_searchable_models(self):
        for name in ["近期频率模型", "贝叶斯平滑模型", "指数时间衰减模型", "综合评分模型", "多模型集成模型"]:
            self.assertGreaterEqual(len(default_parameter_grid(name)), 5)


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.history = synthetic_history(140, seed=42)

    def test_optimizer_returns_ten_unique_legal_tickets(self):
        detail, report = optimize_portfolio(
            self.history, model_name="多模型集成模型", count=10,
            window=50, max_overlap=4, candidate_pool_size=1000,
            seed=123,
        )
        self.assertEqual(len(detail), 10)
        parsed = [tuple(int(x) for x in row.split(" - ")) for row in detail["号码"]]
        self.assertEqual(len(set(parsed)), 10)
        self.assertTrue(all(len(x) == 6 and len(set(x)) == 6 and min(x) >= 1 and max(x) <= 45 for x in parsed))
        self.assertEqual(report["组合组数"], 10)
        self.assertLessEqual(report["最大组间重号"], report["实际允许最大组间重号"])

    def test_objective_weights_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "总和为1"):
            optimize_portfolio(self.history, candidate_pool_size=500,
                               quality_weight=0.5, coverage_weight=0.4,
                               overlap_penalty=0.2)


class ModelManagementTests(unittest.TestCase):
    def test_state_export_import_roundtrip(self):
        state = new_project_state()
        text = export_project_state(state)
        restored = import_project_state(text)
        self.assertEqual(set(state["models"]), set(restored["models"]))
        self.assertEqual(restored["experiments"], [])

    def test_verified_status_requires_model_specific_holdout(self):
        state = new_project_state()
        state = add_experiment(state, "常规回测", {"experiment_id": "bt1"}, [])
        with self.assertRaisesRegex(ValueError, "保留区间验证"):
            change_model_status(state, "贝叶斯平滑模型", "验证通过", "测试", "bt1")
        meta = {"experiment_id": "hold1", "evaluated_model": "贝叶斯平滑模型", "holdout_evaluated": True}
        state = add_experiment(state, "固定参数保留区间验证", meta, [])
        changed = change_model_status(state, "贝叶斯平滑模型", "验证通过", "完成固定参数验证", "hold1")
        self.assertEqual(changed["models"]["贝叶斯平滑模型"]["status"], "验证通过")

    def test_import_rejects_unsubstantiated_verified_status(self):
        state = new_project_state()
        state["models"]["贝叶斯平滑模型"]["status"] = "验证通过"
        state["models"]["贝叶斯平滑模型"]["evidence_experiment_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "缺少匹配的保留区间验证证据"):
            import_project_state(json.dumps(state, ensure_ascii=False))

    def test_status_change_requires_reason(self):
        state = new_project_state()
        with self.assertRaisesRegex(ValueError, "填写原因"):
            change_model_status(state, "贝叶斯平滑模型", "候选", "")


class OfflinePolicyExtensionTests(unittest.TestCase):
    def test_new_modules_have_no_network_integration(self):
        forbidden = ["http://", "https://", "dhlottery", "requests.get", "requests.post", "urllib.request", "httpx"]
        for name in ["parameter_optimizer.py", "portfolio_optimizer.py", "model_management.py"]:
            source = (ROOT / name).read_text(encoding="utf-8").lower()
            for token in forbidden:
                self.assertNotIn(token, source, f"{name} contains {token}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
