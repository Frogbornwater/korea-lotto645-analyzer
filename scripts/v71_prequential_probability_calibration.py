# -*- coding: utf-8 -*-
"""V7.1 expanding prequential calibration audit; research only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    # Direct script execution places the scripts/ directory on sys.path.
    from v64_exact_null_holm_audit import bootstrap_ci, sign_flip_pvalue, holm_adjust
except ModuleNotFoundError:
    # Unit-test imports from the repository root use the package-style path.
    from scripts.v64_exact_null_holm_audit import bootstrap_ci, sign_flip_pvalue, holm_adjust

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ((850, 899), (900, 949), (950, 999), (1000, 1049))
UNIFORM = 6.0 / 45.0
LAMBDA_GRID = tuple(float(x) for x in np.linspace(0.0, 1.0, 21))
SEED = 71026
REPS = 20000


def calibrate_probabilities(probabilities, shrinkage: float) -> np.ndarray:
    """Shrink marginal inclusion probabilities toward the exact uniform prior."""
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (45,) or not np.all(np.isfinite(p)):
        raise ValueError("概率必须是45维有限向量。")
    if np.any(p < 0.0) or np.any(p > 1.0) or not np.isclose(p.sum(), 6.0, atol=1e-5):
        raise ValueError("输入概率必须位于[0,1]且总和为6。")
    lam = float(shrinkage)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("shrinkage必须位于[0,1]。")
    out = UNIFORM + lam * (p - UNIFORM)
    if not np.isclose(out.sum(), 6.0, atol=1e-8):
        raise RuntimeError("校准后概率总和不为6。")
    return out


def brier_vector(probabilities, actual) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.zeros(45, dtype=float)
    nums = [int(x) for x in actual]
    if len(nums) != 6 or len(set(nums)) != 6 or min(nums) < 1 or max(nums) > 45:
        raise ValueError("实际号码必须是1—45内六个不重复号码。")
    y[np.asarray(nums, dtype=int) - 1] = 1.0
    return float(np.mean((p - y) ** 2))


def logloss_vector(probabilities, actual) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
    y = np.zeros(45, dtype=float)
    y[np.asarray([int(x) for x in actual], dtype=int) - 1] = 1.0
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _decode(row):
    p = np.asarray(json.loads(row["V5_45号码概率_JSON"]), dtype=float)
    actual = [int(x) for x in json.loads(row["实际主号码"])]
    if p.shape != (45,) or not np.isclose(p.sum(), 6.0, atol=1e-5):
        raise ValueError("V7.1发现V5概率向量异常。")
    return p, actual


def choose_shrinkage(training_rows: pd.DataFrame) -> tuple[float, dict]:
    """Select lambda using only earlier out-of-fold predictions."""
    if training_rows.empty:
        return 1.0, {"selection": "no_prior_oof_data_raw_model_fallback"}
    decoded = [_decode(row) for row in training_rows.to_dict(orient="records")]
    scores = []
    for lam in LAMBDA_GRID:
        losses = [brier_vector(calibrate_probabilities(p, lam), actual) for p, actual in decoded]
        scores.append(float(np.mean(losses)))
    # np.argmin deterministically chooses the smaller lambda on exact ties.
    best = int(np.argmin(scores))
    return LAMBDA_GRID[best], {
        "selection": "minimum_prior_oof_brier",
        "training_draws": len(decoded),
        "grid": list(LAMBDA_GRID),
        "grid_mean_brier": scores,
    }


def run_audit(detail: pd.DataFrame) -> dict:
    required = {"target_draw", "fold", "V5_45号码概率_JSON", "实际主号码"}
    missing = required - set(detail.columns)
    if missing:
        raise ValueError("V7.1明细缺少字段：" + ", ".join(sorted(missing)))
    detail = detail.sort_values("target_draw").reset_index(drop=True)
    if len(detail) != 200 or detail["target_draw"].astype(int).tolist() != list(range(850, 1050)):
        raise ValueError("V7.1要求V7.0提供850—1049期完整200期逐期明细。")

    records = []
    fold_reports = []
    for fold_index, (start, end) in enumerate(FOLDS, 1):
        prior = detail.loc[detail["target_draw"] < start]
        current = detail.loc[detail["target_draw"].between(start, end)]
        lam, selection = choose_shrinkage(prior)
        for row in current.to_dict(orient="records"):
            p, actual = _decode(row)
            calibrated = calibrate_probabilities(p, lam)
            records.append({
                "fold": fold_index,
                "target_draw": int(row["target_draw"]),
                "calibration_training_draws": len(prior),
                "shrinkage_lambda": lam,
                "raw_brier": brier_vector(p, actual),
                "calibrated_brier": brier_vector(calibrated, actual),
                "uniform_brier": brier_vector(np.full(45, UNIFORM), actual),
                "raw_logloss": logloss_vector(p, actual),
                "calibrated_logloss": logloss_vector(calibrated, actual),
                "uniform_logloss": logloss_vector(np.full(45, UNIFORM), actual),
            })
        fold_reports.append({
            "fold": fold_index, "target_range": [start, end],
            "calibration_training_range": (
                [int(prior["target_draw"].min()), int(prior["target_draw"].max())]
                if len(prior) else None
            ),
            "calibration_training_draws": len(prior),
            "selected_shrinkage_lambda": lam,
            "selection": selection,
        })

    result = pd.DataFrame(records)
    if len(result) != 200 or result["target_draw"].nunique() != 200:
        raise RuntimeError("V7.1目标期数量错误。")
    for report in fold_reports:
        start = report["target_range"][0]
        training_range = report["calibration_training_range"]
        if training_range is not None and training_range[1] >= start:
            raise RuntimeError("V7.1校准训练包含当前折或未来目标期，检测到时间泄漏。")
        expected_prior = start - FOLDS[0][0]
        if report["calibration_training_draws"] != expected_prior:
            raise RuntimeError("V7.1校准训练样本数与严格历史前缀不一致。")

    summary = []
    raw_p = {}
    for index, metric in enumerate(("brier", "logloss")):
        raw_delta = result[f"raw_{metric}"] - result[f"uniform_{metric}"]
        cal_delta = result[f"calibrated_{metric}"] - result[f"uniform_{metric}"]
        paired = result[f"calibrated_{metric}"] - result[f"raw_{metric}"]
        low, high = bootstrap_ci(paired.to_numpy(), repetitions=REPS, seed=SEED + index * 31)
        pvalue = sign_flip_pvalue(paired.to_numpy(), repetitions=REPS, seed=SEED + index * 37)
        raw_p[metric] = pvalue
        summary.append({
            "metric": metric,
            "raw_v5_mean": float(result[f"raw_{metric}"].mean()),
            "prequential_calibrated_mean": float(result[f"calibrated_{metric}"].mean()),
            "uniform_mean": float(result[f"uniform_{metric}"].mean()),
            "calibrated_minus_raw": float(paired.mean()),
            "calibrated_minus_raw_95ci_low": low,
            "calibrated_minus_raw_95ci_high": high,
            "p_value_two_sided_sign_flip": pvalue,
            "raw_minus_uniform": float(raw_delta.mean()),
            "calibrated_minus_uniform": float(cal_delta.mean()),
        })
    adjusted = holm_adjust(raw_p)
    for row in summary:
        row["holm_adjusted_p_value_2_tests"] = adjusted[row["metric"]]
        row["passes_holm_0_05"] = bool(adjusted[row["metric"]] < 0.05)

    return {
        "experiment": "V7.1 expanding prequential probability-shrinkage calibration",
        "status": "historical_development_screen_not_independent_or_prospective_evidence",
        "target_range": [850, 1049],
        "target_draws": len(result),
        "method": "For each chronological fold, select shrinkage on earlier V5 out-of-fold predictions only; apply once to the next untouched fold.",
        "calibration_grid": list(LAMBDA_GRID),
        "fold_reports": fold_reports,
        "summary": summary,
        "limitations": [
            "All 200 target draws were already used in earlier historical development audits; this is not independent or prospective evidence.",
            "The first fold uses the raw V5 probabilities because no earlier out-of-fold calibration sample exists.",
            "Shrinkage changes marginal probabilities but not their ranking or any fixed ticket's theoretical jackpot probability.",
            "The calibration selection uses Brier score; LogLoss is a secondary predeclared evaluation metric.",
            "Target draws are the statistical units; intervals and tests remain sensitive to temporal dependence.",
            "This audit does not change production code or any frozen forward lock."
        ],
        "detail": result,
    }


def main():
    source = ROOT / "artifacts" / "v70" / "v70_signal_integrity_detail.csv"
    if not source.exists():
        raise FileNotFoundError("缺少V7.0逐期明细；必须先运行V7.0。")
    report = run_audit(pd.read_csv(source))
    out = ROOT / "artifacts" / "v71"
    out.mkdir(parents=True, exist_ok=True)
    report["detail"].to_csv(out / "v71_prequential_calibration_detail.csv", index=False, encoding="utf-8-sig")
    (out / "v71_prequential_calibration_report.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "detail"}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "detail"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
