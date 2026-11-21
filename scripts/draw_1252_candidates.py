# -*- coding: utf-8 -*-
"""Build a reproducible, research-only candidate comparison for draw 1252.

All history is read from the repository's manually maintained CSV. This script
does not access any lottery website or API. V8.2 and V8.3 are represented only
by their actual capabilities: a pair-probability diagnostic and a feature
audit, respectively; neither is misrepresented as a six-number ticket model.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lotto_core import NUMBER_COLS, quality_report, validate_frame
from v50_predictive_engine import fit_predict_next, generate_tickets
from v80_probability_sampling import generate_probability_faithful_tickets
from v81_batch_sampler import generate_batch_quota_portfolio
from v82_pairwise_joint import (
    PAIR_FEATURE_EXCLUSIONS,
    fit_pair_residuals,
    forecast_pair_probabilities,
)

HISTORY_PATH = ROOT / "data" / "lotto645.csv"
OUTPUT_DIR = ROOT / "artifacts" / "draw1252"
TARGET_DRAW = 1252
TRAINING_WINDOW = 500
RIDGE_ALPHA = 1.0
MIN_HISTORY = 180
V5_SEED = 12520501
V80_SEED = 12520800
V81_SEED = 12520801
V82_PAIR_PRIOR_DRAWS = 1000.0
V82_PAIR_STRENGTH = 0.5


def fmt_tickets(tickets: list[list[int]]) -> list[list[str]]:
    return [[f"{int(n):02d}" for n in row] for row in tickets]


def validate_history() -> pd.DataFrame:
    raw = pd.read_csv(HISTORY_PATH)
    clean = validate_frame(raw).reset_index(drop=True)
    quality = quality_report(raw)
    if clean.empty or int(clean["draw_no"].iloc[0]) != 1:
        raise ValueError("历史数据必须从第1期开始。")
    if not quality.get("sequence_complete", False):
        raise ValueError("历史数据存在缺期，拒绝生成第1252期候选。")
    last = int(clean["draw_no"].iloc[-1])
    if last != 1251:
        raise ValueError(f"目标第1252期要求历史截止1251；当前截止{last}。")
    if len(clean) != 1251:
        raise ValueError(f"预期1251条历史记录，实际为{len(clean)}。")
    actual = clean.loc[clean["draw_no"] == 1251, NUMBER_COLS + ["bonus"]].iloc[0].astype(int).tolist()
    if actual != [14, 18, 19, 29, 40, 45, 1]:
        raise ValueError(f"第1251期号码与用户手动提供记录不一致：{actual}")
    return clean


def build_report() -> dict[str, Any]:
    clean = validate_history()
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    matrix = clean[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)

    # V5 production baseline: original ten-feature engine and established sampler.
    v5 = fit_predict_next(
        clean, training_window=TRAINING_WINDOW, min_history=MIN_HISTORY,
        ridge_alpha=RIDGE_ALPHA,
    )
    p_v5 = (
        v5["number_probabilities"].sort_values("号码")["预测入选概率"]
        .to_numpy(dtype=float)
    )
    v5_tickets = generate_tickets(
        p_v5, v5["pair_scores"], count=10, seed=V5_SEED, pair_strength=0.5
    )

    # V8.0: same V5 marginal probabilities, but exact fixed-size conditional
    # Bernoulli sampling; independent ticket draws, duplicates are permitted.
    v80_tickets, v80_fit, v80_batch = generate_probability_faithful_tickets(
        p_v5, count=10, seed=V80_SEED, sample_size=6
    )

    # V8.1: batch-level unbiased integer quotas; not identical per-ticket marginals.
    v81_tickets, v81_batch = generate_batch_quota_portfolio(
        p_v5, ticket_count=10, sample_size=6, seed=V81_SEED
    )

    # V8.2: actual pairwise inclusion probabilities. It does not define a
    # complete distribution over 6-number subsets, so no V8.2 ticket is emitted.
    v82_base = fit_predict_next(
        clean, training_window=TRAINING_WINDOW, min_history=MIN_HISTORY,
        ridge_alpha=RIDGE_ALPHA,
        feature_exclusions=PAIR_FEATURE_EXCLUSIONS,
    )
    p_v82 = (
        v82_base["number_probabilities"].sort_values("号码")["预测入选概率"]
        .to_numpy(dtype=float)
    )
    pair_fit = fit_pair_residuals(
        matrix[-TRAINING_WINDOW:], prior_draws=V82_PAIR_PRIOR_DRAWS
    )
    v82_pair, v82_pair_baseline, v82_diag = forecast_pair_probabilities(
        p_v82, pair_fit, strength=V82_PAIR_STRENGTH
    )
    pair_i, pair_j = np.triu_indices(45, 1)
    pair_delta = v82_pair[pair_i, pair_j] - v82_pair_baseline[pair_i, pair_j]
    top_pair_order = np.argsort(pair_delta)[::-1][:10]
    top_pairs = [
        {
            "numbers": [int(pair_i[k] + 1), int(pair_j[k] + 1)],
            "baseline_probability": float(v82_pair_baseline[pair_i[k], pair_j[k]]),
            "adjusted_probability": float(v82_pair[pair_i[k], pair_j[k]]),
            "adjustment": float(pair_delta[k]),
        }
        for k in top_pair_order
    ]

    # V8.3 is an ablation audit, not a ticket generator. The last completed
    # 500-draw audit found no feature passing its 32-test Holm gate; hence all
    # eight first-order features remain in the V5 baseline.
    v83_audit = {
        "role": "first_order_feature_contribution_audit_only",
        "last_completed_historical_audit": "500-draw prequential audit through draw 1250",
        "supported_features_after_holm": [],
        "action": "retain all eight first-order features; do not change V5 weights",
        "not_recomputed_for_this_single_target": True,
        "features_retained": [
            "recent_30_z", "recent_60_z", "recent_120_z", "recent_240_z",
            "long_frequency_z", "trend_30_minus_120", "ewma_35_z",
            "gap_log_scaled",
        ],
    }

    normalized_history_hash = hashlib.sha256(
        clean.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    raw_csv_hash = hashlib.sha256(HISTORY_PATH.read_bytes()).hexdigest()

    report: dict[str, Any] = {
        "schema_version": 1,
        "commitment_id": "lotto645-draw-1252-research-candidates",
        "event_type": "draw_1252_forward_candidate_comparison",
        "created_at_utc": created,
        "target_draw": TARGET_DRAW,
        "history": {
            "source": "repository data/lotto645.csv, manually maintained only",
            "first_draw": int(clean["draw_no"].iloc[0]),
            "last_draw": int(clean["draw_no"].iloc[-1]),
            "rows": int(len(clean)),
            "raw_csv_sha256": raw_csv_hash,
            "normalized_history_sha256": normalized_history_hash,
            "draw_1251": {
                "main_numbers": [14, 18, 19, 29, 40, 45],
                "bonus": 1,
                "date": "",
                "provided_by_user": True,
            },
        },
        "common_model_configuration": {
            "marginal_model": "V5.0.0-predictive-research",
            "training_window": TRAINING_WINDOW,
            "ridge_alpha": RIDGE_ALPHA,
            "min_history": MIN_HISTORY,
            "ticket_count_per_group": 10,
            "ticket_size": 6,
        },
        "candidate_groups": {
            "V5_baseline": {
                "tickets": fmt_tickets(v5_tickets),
                "seed": V5_SEED,
                "sampler": "V5 sequential weighted sampling with pair_strength=0.5",
                "probability_sum": float(v5["probability_sum"]),
            },
            "V8.0": {
                "tickets": fmt_tickets(v80_tickets),
                "seed": V80_SEED,
                "sampler": "conditional Bernoulli / rejective sampling",
                "calibration_max_abs_marginal_error": float(v80_fit.max_abs_error),
                "batch_diagnostics": v80_batch,
                "interpretation": "per-ticket marginals match V5 probabilities; independent draws may repeat tickets",
            },
            "V8.1": {
                "tickets": fmt_tickets(v81_tickets),
                "seed": V81_SEED,
                "sampler": "dependent-rounding batch quota portfolio",
                "batch_diagnostics": v81_batch,
                "interpretation": "batch expected appearances are preserved; individual ticket marginals are not identical",
            },
        },
        "V8.2_pairwise_diagnostic": {
            "role": "pair_inclusion_probability_research_only",
            "tickets_generated": False,
            "reason": "V8.2 supplies a 45x45 pair-inclusion forecast, not a normalized distribution over six-number subsets",
            "marginal_model_features_excluded": list(PAIR_FEATURE_EXCLUSIONS),
            "pair_training_draws": int(pair_fit.training_draws),
            "pair_prior_draws": V82_PAIR_PRIOR_DRAWS,
            "pair_strength": V82_PAIR_STRENGTH,
            "max_row_sum_error": float(v82_diag["max_row_sum_error"]),
            "forecast_pair_probability_sum": float(v82_diag["pair_probability_sum"]),
            "top_positive_pair_adjustments": top_pairs,
        },
        "V8.3_feature_audit": v83_audit,
        "evidence_boundary": {
            "prediction_superiority_proven": False,
            "probability_of_any_ticket_claimed_above_uniform": False,
            "production_model_changed": False,
            "historical_research_not_prospective_evidence": True,
            "notes": [
                "V8.0 and V8.1 change sampling design, not the learned first-order predictive signal.",
                "V8.2 and V8.3 are diagnostics, not independent six-number ticket predictors.",
                "The three candidate groups have not been prospectively shown to outperform the uniform lottery baseline.",
                "No official lottery website or API is accessed.",
            ],
        },
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["canonical_payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report


def main() -> None:
    report = build_report()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "draw_1252_candidate_comparison.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DRAW_1252_CANDIDATE_REPORT_BEGIN")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("DRAW_1252_CANDIDATE_REPORT_END")
    print(f"REPORT_PATH={output}")


if __name__ == "__main__":
    main()
