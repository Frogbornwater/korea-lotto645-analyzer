import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest_engine import _make_weights, run_backtest, run_multi_window_backtest
from data_manager import read_manual_upload
from lotto_core import COLUMNS, MODEL_NAMES, model_weights, quality_report, validate_frame
from model_registry import ENGINE_VERSION, get_model_registry


def synthetic_history(draws=90, seed=111):
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(1, draws + 1):
        nums = sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False))
        available = sorted(set(range(1, 46)) - set(nums))
        bonus = int(rng.choice(available))
        rows.append({
            "draw_no": draw, "date": f"2024-01-{(draw - 1) % 28 + 1:02d}",
            **{f"n{i}": nums[i - 1] for i in range(1, 7)}, "bonus": bonus,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


class ValidationTests(unittest.TestCase):
    def test_manual_chinese_headers(self):
        raw = pd.DataFrame([{
            "회차": 1, "추첨일": "20240106", "번호1": 1, "번호2": 2,
            "번호3": 3, "번호4": 4, "번호5": 5, "번호6": 6, "보너스번호": 7,
        }])
        clean = validate_frame(raw)
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean.iloc[0]["n6"], 6)

    def test_partial_but_contiguous_history_range_is_distinguished(self):
        frame = synthetic_history(5)
        frame["draw_no"] = frame["draw_no"] + 100
        report = quality_report(frame)
        self.assertTrue(report["range_complete"])
        self.assertFalse(report["sequence_complete"])
        self.assertEqual(report["missing_before_first"], 100)

    def test_conflicting_duplicate_draw_rejected(self):
        frame = synthetic_history(3)
        conflicting = frame.iloc[[0]].copy()
        conflicting.loc[:, "n1"] = 44
        combined = pd.concat([frame, conflicting], ignore_index=True)
        with self.assertRaises(ValueError):
            validate_frame(combined)

    def test_manual_csv_upload(self):
        frame = synthetic_history(5)
        content = frame.to_csv(index=False).encode("utf-8-sig")
        clean, report = read_manual_upload(io.BytesIO(content), "history.csv")
        self.assertEqual(len(clean), 5)
        self.assertEqual(report["source_mode"], "manual_upload")

    def test_manual_xlsx_upload(self):
        frame = synthetic_history(5)
        buffer = io.BytesIO()
        frame.to_excel(buffer, index=False)
        buffer.seek(0)
        clean, report = read_manual_upload(buffer, "history.xlsx")
        self.assertEqual(len(clean), 5)
        self.assertEqual(report["source_mode"], "manual_upload")


class RegistryTests(unittest.TestCase):
    def test_registry_matches_core_models(self):
        self.assertEqual([x["name"] for x in get_model_registry()], MODEL_NAMES)
        self.assertEqual(ENGINE_VERSION, "4.0.0-governed")


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = synthetic_history(90, seed=20261106)

    def test_backtest_row_count_and_summary(self):
        raw, summary, meta = run_backtest(
            self.history, lookback=40, test_draws=20,
            tickets_per_draw=3, n_seeds=2, seed=77, bootstrap_reps=300,
        )
        self.assertEqual(len(raw), 20 * 3 * 2 * 8)
        self.assertEqual(len(summary), 8)
        self.assertEqual(meta["test_draws"], 20)
        self.assertIn("号码JSON", raw.columns)
        self.assertIn("平均每批号码覆盖数", summary.columns)
        self.assertIn("平均命中差_Holm校正p值", summary.columns)
        self.assertIn("实验编号", raw.columns)
        self.assertEqual(meta["data_source_label"], "用户手动提供的历史数据")

    def test_internal_draw_gap_blocks_backtest(self):
        incomplete = self.history.drop(index=[50]).reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "存在缺失期号"):
            run_backtest(
                incomplete, lookback=40, test_draws=10,
                tickets_per_draw=2, n_seeds=1, seed=88, bootstrap_reps=100, signflip_reps=100,
            )

    def test_no_future_leakage_metadata(self):
        raw, _, _ = run_backtest(
            self.history, lookback=40, test_draws=10,
            tickets_per_draw=2, n_seeds=1, seed=88, bootstrap_reps=200,
        )
        self.assertTrue((raw["训练截止期号"] < raw["期数"]).all())
        self.assertTrue((raw["训练可用期数"] >= 40).all())

    def test_same_seed_reproducible_ticket_numbers(self):
        kwargs = dict(lookback=40, test_draws=8, tickets_per_draw=3,
                      n_seeds=2, seed=9001, bootstrap_reps=200)
        raw1, _, _ = run_backtest(self.history, **kwargs)
        raw2, _, _ = run_backtest(self.history, **kwargs)
        cols = ["期数", "随机种子组", "策略", "注序", "号码", "命中数"]
        pd.testing.assert_frame_equal(raw1[cols], raw2[cols])

    def test_each_ticket_has_six_distinct_valid_numbers(self):
        raw, _, _ = run_backtest(
            self.history, lookback=40, test_draws=5,
            tickets_per_draw=4, n_seeds=1, seed=14, bootstrap_reps=100,
        )
        for _, row in raw.iterrows():
            nums = [int(row[f"n{i}"]) for i in range(1, 7)]
            self.assertEqual(len(set(nums)), 6)
            self.assertTrue(all(1 <= n <= 45 for n in nums))

    def test_group_size_and_unique_ticket_numbers(self):
        raw, _, _ = run_backtest(
            self.history, lookback=40, test_draws=6,
            tickets_per_draw=5, n_seeds=1, seed=45, bootstrap_reps=100,
        )
        sizes = raw.groupby(["期数", "随机种子组", "策略"]).size()
        self.assertTrue((sizes == 5).all())
        dupes = raw.duplicated(["期数", "随机种子组", "策略", "号码"])
        self.assertEqual(int(dupes.sum()), 0)

    def test_multi_window_backtest(self):
        raw, summary, meta = run_multi_window_backtest(
            self.history, windows=[30, 40], test_draws=8,
            tickets_per_draw=2, n_seeds=1, seed=33, bootstrap_reps=100,
        )
        self.assertEqual(set(raw["比较训练窗口"].unique()), {30, 40})
        self.assertEqual(set(summary["比较训练窗口"].unique()), {30, 40})
        self.assertEqual(len(meta), 2)


    def test_adjusted_p_values_are_not_below_raw_p_values(self):
        _, summary, _ = run_backtest(
            self.history, lookback=40, test_draws=15,
            tickets_per_draw=2, n_seeds=1, seed=23,
            bootstrap_reps=150, signflip_reps=250,
        )
        non_random = summary[summary["策略"] != "均匀随机基准"]
        self.assertTrue((non_random["平均命中差_Holm校正p值"] >= non_random["平均命中差_配对检验p值"]).all())

    def test_candidate_and_backtest_weight_definitions_match(self):
        numbers = self.history[[f"n{i}" for i in range(1, 7)]].to_numpy(dtype=np.int16)
        for model in MODEL_NAMES:
            expected = _make_weights(numbers, model, recent_window=35, half_life=40.0,
                                     bayes_prior=80.0, ensemble_weights=(0.40, 0.35, 0.25),
                                     multi_model_weights=(0.40, 0.35, 0.25))
            actual = model_weights(self.history, model, window=35, half_life=40.0)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_random_baseline_theoretical_mean_is_not_claimed_as_empirical(self):
        _, summary, _ = run_backtest(
            self.history, lookback=40, test_draws=40,
            tickets_per_draw=10, n_seeds=2, seed=121, bootstrap_reps=300,
        )
        baseline = summary.loc[summary["策略"] == "均匀随机基准", "平均命中数"].iloc[0]
        self.assertGreater(baseline, 0.55)
        self.assertLess(baseline, 1.05)


class OfflinePolicyTests(unittest.TestCase):
    def test_application_python_sources_have_no_network_integration(self):
        targets = ["app.py", "lotto_core.py", "backtest_engine.py", "model_registry.py", "data_manager.py", "sync_data.py", "parameter_optimizer.py", "portfolio_optimizer.py", "model_management.py", "experiment_export.py"]
        forbidden = ["http://", "https://", "dhlottery", "requests.get", "requests.post", "urllib.request", "httpx"]
        for filename in targets:
            source = (ROOT / filename).read_text(encoding="utf-8").lower()
            for token in forbidden:
                self.assertNotIn(token, source, f"{filename} contains forbidden network token: {token}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
