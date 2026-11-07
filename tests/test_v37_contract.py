"""Deployment contract tests for the V3.7 joint optimizer integration."""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backtest_engine
from joint_optimizer import SEARCH_MODELS, _weights_for_config, generate_joint_configs
from model_registry import ENGINE_VERSION


class V37DeploymentContractTests(unittest.TestCase):
    def test_engine_version_is_joint_release(self):
        self.assertEqual(ENGINE_VERSION, "3.8.0-robust")

    def test_make_weights_accepts_joint_optimizer_eleven_argument_contract(self):
        signature = inspect.signature(backtest_engine._make_weights)
        self.assertEqual(len(signature.parameters), 11)
        self.assertEqual(
            list(signature.parameters),
            [
                "all_past", "strategy", "recent_window", "half_life",
                "bayes_prior", "ensemble_weights", "multi_model_weights",
                "recent_prior", "long_prior", "ensemble_short_window",
                "ensemble_mid_window",
            ],
        )
        matrix = np.random.default_rng(2026).integers(1, 46, size=(320, 6))
        result = backtest_engine._make_weights(
            matrix, SEARCH_MODELS[0], 50, 40.0, 80.0,
            (0.40, 0.35, 0.25), (0.40, 0.35, 0.25),
            8.0, 40.0, 50, 200,
        )
        self.assertEqual(result.shape, (45,))
        self.assertTrue(np.isfinite(result).all())

    def test_all_seven_joint_search_models_call_engine_successfully(self):
        matrix = np.random.default_rng(7307).integers(1, 46, size=(320, 6))
        configs = generate_joint_configs(70, seed=77)
        covered = set()
        for config in configs:
            weights, bounds = _weights_for_config(matrix, config, 300)
            self.assertEqual(weights.shape, (45,), config["model_name"])
            self.assertTrue(np.isfinite(weights).all(), config["model_name"])
            if config["model_name"] == "历史结构约束模型":
                self.assertIsNotNone(bounds)
            covered.add(config["model_name"])
        self.assertEqual(covered, set(SEARCH_MODELS))

    def test_app_wires_joint_optimizer_and_displays_engine_version(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("from joint_optimizer import run_joint_optimization", source)
        self.assertIn("V3.7联合优化", source)
        self.assertIn("ENGINE_VERSION", source)
        self.assertIn("run_joint_optimization(", source)

    def test_sync_entrypoint_is_a_disabled_offline_stub(self):
        source = (ROOT / "sync_data.py").read_text(encoding="utf-8").lower()
        for token in ["http://", "https://", "dhlottery", "requests", "urllib", "httpx"]:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
