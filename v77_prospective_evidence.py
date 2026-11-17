# -*- coding: utf-8 -*-
"""Pre-registered cumulative evidence for the V5.1 forward experiment.

The confirmatory window is fixed at draws 1251–1300. Before all 50 unique
targets are scored, this module reports descriptive summaries only. It never
selects among multiple locks for the same target draw.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


V77_PROTOCOL_ID = "v77-v51-forward-50-draws"
V77_START_DRAW = 1251
V77_PLANNED_DRAWS = 50
V77_END_DRAW = V77_START_DRAW + V77_PLANNED_DRAWS - 1
V77_BOOTSTRAP_REPLICATES = 20_000
V77_BOOTSTRAP_SEED = 77_125_130

P_UNIFORM = 6.0 / 45.0
UNIFORM_BRIER = (
    6.0 * (1.0 - P_UNIFORM) ** 2 + 39.0 * P_UNIFORM**2
) / 45.0
UNIFORM_LOG_LOSS = -(
    6.0 * np.log(P_UNIFORM) + 39.0 * np.log(1.0 - P_UNIFORM)
) / 45.0

_REQUIRED_COLUMNS = (
    "目标期号",
    "V5.1注数",
    "随机注数",
    "V5.1平均命中数",
    "随机平均命中数",
    "V5.1单注3+率",
    "随机单注3+率",
    "V5.1整批至少一注3+",
    "随机整批至少一注3+",
    "V5.1_Brier",
    "V5.1_LogLoss",
)

_FIXED_CONFIG_FIELDS = (
    "model",
    "engine_version",
    "training_window",
    "ridge_alpha",
    "pair_strength",
    "requested_overlap",
    "ticket_count",
    "seed",
)

_OUTPUT_COLUMNS = (
    "目标期号",
    "V5.1注数",
    "随机注数",
    "V5.1平均命中数",
    "随机平均命中数",
    "平均命中差值",
    "V5.1单注3+率",
    "随机单注3+率",
    "单注3+率差值",
    "V5.1整批至少一注3+",
    "随机整批至少一注3+",
    "整批3+率差值",
    "V5.1_Brier",
    "均匀基准_Brier",
    "Brier差值",
    "V5.1_LogLoss",
    "均匀基准_LogLoss",
    "LogLoss差值",
)


def _finite_number(value: Any, field: str, draw_no: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"第{draw_no}期的{field}不是有效数值。") from exc
    if not np.isfinite(number):
        raise ValueError(f"第{draw_no}期的{field}不是有限数值。")
    return number


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("计算跨期区间至少需要两个有效目标期。")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    estimates = values[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def build_v77_evidence_report(
    forward_summary: pd.DataFrame | None,
    *,
    lock_events: list[dict[str, Any]] | None = None,
    start_draw: int = V77_START_DRAW,
    planned_draws: int = V77_PLANNED_DRAWS,
    bootstrap_repetitions: int = V77_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = V77_BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Build a fixed-horizon report, per-draw table, and descriptive metric table.

    Primary endpoint: per-target difference in mean hits per ticket
    (V5.1 minus its same-draw random control). The confirmatory interval is
    computed once the complete, unique fixed horizon is available.

    Secondary ticket/batch/probability metrics remain descriptive. No early
    stopping, interim significance claim, or best-of-multiple-lock selection
    is performed here.
    """
    start_draw = int(start_draw)
    planned_draws = int(planned_draws)
    if start_draw < 1 or planned_draws < 2:
        raise ValueError("固定观察窗口至少需要从正期号开始并包含两个目标期。")
    if bootstrap_repetitions < 100:
        raise ValueError("Bootstrap重复次数至少为100。")

    end_draw = start_draw + planned_draws - 1
    expected_draws = list(range(start_draw, end_draw + 1))
    empty_per_draw = pd.DataFrame(columns=_OUTPUT_COLUMNS)

    if forward_summary is None or forward_summary.empty:
        rows = pd.DataFrame(columns=_OUTPUT_COLUMNS)
        duplicate_draws: list[int] = []
        out_of_window_draws: list[int] = []
    else:
        missing_columns = [column for column in _REQUIRED_COLUMNS if column not in forward_summary.columns]
        if missing_columns:
            raise ValueError("前瞻复盘摘要缺少必要字段：" + "、".join(missing_columns))

        source = forward_summary.copy()
        try:
            source["目标期号"] = pd.to_numeric(source["目标期号"], errors="raise").astype(int)
        except (TypeError, ValueError) as exc:
            raise ValueError("前瞻复盘摘要包含无效目标期号。") from exc

        in_window = source.loc[
            source["目标期号"].between(start_draw, end_draw)
        ].copy()
        out_of_window_draws = sorted(
            set(source.loc[~source["目标期号"].between(start_draw, end_draw), "目标期号"].astype(int).tolist())
        )
        counts = in_window["目标期号"].value_counts()
        duplicate_draws = sorted(int(draw) for draw, count in counts.items() if int(count) > 1)
        accepted = in_window.loc[~in_window["目标期号"].isin(duplicate_draws)].copy()

        normalized_rows: list[dict[str, Any]] = []
        for _, raw in accepted.sort_values("目标期号").iterrows():
            draw_no = int(raw["目标期号"])
            model_ticket_count_value = _finite_number(raw["V5.1注数"], "V5.1注数", draw_no)
            random_ticket_count_value = _finite_number(raw["随机注数"], "随机注数", draw_no)
            if (
                not model_ticket_count_value.is_integer()
                or not random_ticket_count_value.is_integer()
            ):
                raise ValueError(f"第{draw_no}期注数必须为整数。")
            model_ticket_count = int(model_ticket_count_value)
            random_ticket_count = int(random_ticket_count_value)
            if model_ticket_count < 1 or model_ticket_count != random_ticket_count:
                raise ValueError(f"第{draw_no}期V5.1与随机对照的注数不一致或为零。")
            model_hits = _finite_number(raw["V5.1平均命中数"], "V5.1平均命中数", draw_no)
            random_hits = _finite_number(raw["随机平均命中数"], "随机平均命中数", draw_no)
            model_ticket3 = _finite_number(raw["V5.1单注3+率"], "V5.1单注3+率", draw_no)
            random_ticket3 = _finite_number(raw["随机单注3+率"], "随机单注3+率", draw_no)
            model_batch3 = _finite_number(raw["V5.1整批至少一注3+"], "V5.1整批至少一注3+", draw_no)
            random_batch3 = _finite_number(raw["随机整批至少一注3+"], "随机整批至少一注3+", draw_no)

            if not (0.0 <= model_hits <= 6.0 and 0.0 <= random_hits <= 6.0):
                raise ValueError(f"第{draw_no}期平均命中数超出0—6范围。")
            for field, value in (
                ("V5.1单注3+率", model_ticket3),
                ("随机单注3+率", random_ticket3),
                ("V5.1整批至少一注3+", model_batch3),
                ("随机整批至少一注3+", random_batch3),
            ):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"第{draw_no}期{field}超出0—1范围。")

            try:
                model_brier = float(raw["V5.1_Brier"])
            except (TypeError, ValueError):
                model_brier = float("nan")
            try:
                model_log_loss = float(raw["V5.1_LogLoss"])
            except (TypeError, ValueError):
                model_log_loss = float("nan")
            if np.isfinite(model_brier) and not 0.0 <= model_brier <= 1.0:
                raise ValueError(f"第{draw_no}期Brier分数超出0—1范围。")
            if np.isfinite(model_log_loss) and model_log_loss < 0.0:
                raise ValueError(f"第{draw_no}期Log Loss不能为负数。")

            normalized_rows.append({
                "目标期号": draw_no,
                "V5.1注数": model_ticket_count,
                "随机注数": random_ticket_count,
                "V5.1平均命中数": model_hits,
                "随机平均命中数": random_hits,
                "平均命中差值": model_hits - random_hits,
                "V5.1单注3+率": model_ticket3,
                "随机单注3+率": random_ticket3,
                "单注3+率差值": model_ticket3 - random_ticket3,
                "V5.1整批至少一注3+": model_batch3,
                "随机整批至少一注3+": random_batch3,
                "整批3+率差值": model_batch3 - random_batch3,
                "V5.1_Brier": model_brier,
                "均匀基准_Brier": UNIFORM_BRIER,
                "Brier差值": model_brier - UNIFORM_BRIER if np.isfinite(model_brier) else np.nan,
                "V5.1_LogLoss": model_log_loss,
                "均匀基准_LogLoss": UNIFORM_LOG_LOSS,
                "LogLoss差值": model_log_loss - UNIFORM_LOG_LOSS if np.isfinite(model_log_loss) else np.nan,
            })
        rows = pd.DataFrame(normalized_rows, columns=_OUTPUT_COLUMNS)

    # Count every imported/active lock, not only events that happened to score.
    # This prevents a failed or pending duplicate lock from disappearing from
    # the protocol audit and then allowing the best-scoring sibling to survive.
    lock_target_counts: dict[int, int] = {}
    lock_events_by_target: dict[int, list[dict[str, Any]]] = {}
    lock_outside_window: set[int] = set()
    for event in lock_events or []:
        if not isinstance(event, dict) or event.get("event_type") != "v51_forward_lock":
            continue
        metadata = event.get("payload", {}).get("metadata", {})
        try:
            lock_target = int(metadata.get("target_draw", -1))
        except (TypeError, ValueError):
            continue
        if start_draw <= lock_target <= end_draw:
            lock_target_counts[lock_target] = lock_target_counts.get(lock_target, 0) + 1
            lock_events_by_target.setdefault(lock_target, []).append(event)
        else:
            lock_outside_window.add(lock_target)

    duplicate_lock_draws = {
        draw_no for draw_no, count in lock_target_counts.items() if count > 1
    }
    duplicate_draws = sorted(set(duplicate_draws) | duplicate_lock_draws)
    out_of_window_draws = sorted(set(out_of_window_draws) | lock_outside_window)
    missing_lock_draws = sorted(set(expected_draws) - set(lock_target_counts))

    # The committed 1251 lock is the configuration reference for the complete
    # prospective window. All later locks must preserve its model settings.
    reference_events = lock_events_by_target.get(start_draw, [])
    reference_config = (
        reference_events[0].get("payload", {}).get("configuration", {})
        if len(reference_events) == 1
        else {}
    )
    reference_config_available = (
        isinstance(reference_config, dict)
        and all(field in reference_config for field in _FIXED_CONFIG_FIELDS)
    )
    configuration_mismatch_draws: list[int] = []
    if lock_events is not None:
        for target, target_events in lock_events_by_target.items():
            if len(target_events) != 1:
                continue
            event = target_events[0]
            payload = event.get("payload", {})
            config = payload.get("configuration", {})
            metadata = payload.get("metadata", {})
            if not reference_config_available:
                configuration_mismatch_draws.append(target)
                continue
            config_matches = all(
                config.get(field) == reference_config.get(field)
                for field in _FIXED_CONFIG_FIELDS
            )
            metadata_echo_fields = (
                "engine_version",
                "training_window",
                "ridge_alpha",
                "pair_strength",
                "requested_overlap",
                "seed",
                "ticket_count",
            )
            metadata_matches = all(
                metadata.get(field) == config.get(field)
                for field in metadata_echo_fields
            )
            try:
                expected_random_seed = int(config["seed"]) ^ 0x5A17
                # 1251 keeps its already committed legacy control seed.
                # Later target draws use independent deterministic seeds.
                if target > start_draw:
                    expected_random_seed ^= target
                random_seed_matches = int(metadata.get("random_seed", -1)) == expected_random_seed
            except (KeyError, TypeError, ValueError):
                random_seed_matches = False
            if not (config_matches and metadata_matches and random_seed_matches):
                configuration_mismatch_draws.append(target)
    configuration_mismatch_draws = sorted(set(configuration_mismatch_draws))

    if duplicate_draws:
        rows = rows.loc[~rows["目标期号"].isin(duplicate_draws)].reset_index(drop=True)

    valid_draws = sorted(int(value) for value in rows["目标期号"].tolist())
    missing_draws = sorted(set(expected_draws) - set(valid_draws))
    missing_reference_lock = lock_events is not None and not reference_config_available
    complete_horizon = (
        not missing_draws
        and not duplicate_draws
        and not missing_lock_draws
        and lock_events is not None
        and not missing_reference_lock
        and not configuration_mismatch_draws
    )
    status = (
        "protocol_violation_duplicate_target"
        if duplicate_draws
        else "protocol_violation_missing_reference_lock"
        if missing_reference_lock
        else "protocol_violation_configuration_mismatch"
        if configuration_mismatch_draws
        else "complete_fixed_horizon"
        if complete_horizon
        else "collecting_fixed_horizon"
    )

    n_scored = len(rows)
    primary_values = (
        rows["平均命中差值"].to_numpy(dtype=float)
        if n_scored
        else np.asarray([], dtype=float)
    )
    primary_estimate = float(primary_values.mean()) if n_scored else None
    primary_ci = None
    interpretation = "固定观察段尚未完成；当前仅作描述，不提供确认性区间或显著性结论。"
    if complete_horizon:
        lower, upper = _bootstrap_mean_interval(
            primary_values,
            repetitions=int(bootstrap_repetitions),
            seed=int(bootstrap_seed),
        )
        primary_ci = {"lower": lower, "upper": upper, "level": 0.95}
        if lower > 0.0:
            interpretation = "固定观察段中，平均命中差值的95% Bootstrap区间高于0；这支持该观察段内存在正向差异，但不保证未来长期优势。"
        elif upper < 0.0:
            interpretation = "固定观察段中，平均命中差值的95% Bootstrap区间低于0；该模型在此观察段内落后于随机对照。"
        else:
            interpretation = "固定观察段中，平均命中差值的95% Bootstrap区间包含0；尚无明确的正向优势证据。"

    metric_definitions = (
        ("每注平均命中数", "V5.1平均命中数", "随机平均命中数", "越高越好"),
        ("单注3+命中率", "V5.1单注3+率", "随机单注3+率", "越高越好"),
        ("每期整批至少一注3+", "V5.1整批至少一注3+", "随机整批至少一注3+", "越高越好"),
        ("Brier分数", "V5.1_Brier", "均匀基准_Brier", "差值越低越好；均匀基准为6/45"),
        ("Log Loss", "V5.1_LogLoss", "均匀基准_LogLoss", "差值越低越好；均匀基准为6/45"),
    )
    metric_rows: list[dict[str, Any]] = []
    for label, model_col, baseline_col, direction in metric_definitions:
        if rows.empty:
            continue
        model_values = rows[model_col].to_numpy(dtype=float)
        baseline_values = rows[baseline_col].to_numpy(dtype=float)
        valid = np.isfinite(model_values) & np.isfinite(baseline_values)
        if not np.any(valid):
            continue
        model_mean = float(model_values[valid].mean())
        baseline_mean = float(baseline_values[valid].mean())
        metric_rows.append({
            "指标": label,
            "V5.1均值": model_mean,
            "对照均值": baseline_mean,
            "平均差值（V5.1－对照）": model_mean - baseline_mean,
            "有效期数": int(valid.sum()),
            "评价方向": direction,
            "确认性状态": "仅主要终点有固定窗口区间；本项为描述性指标",
        })
    metric_table = pd.DataFrame(metric_rows)

    report: dict[str, Any] = {
        "protocol_id": V77_PROTOCOL_ID,
        "status": status,
        "analysis_ready": bool(complete_horizon),
        "target_window": {"start_draw": start_draw, "end_draw": end_draw},
        "planned_draws": planned_draws,
        "scored_unique_draws": n_scored,
        "missing_draws": missing_draws,
        "missing_lock_draws": missing_lock_draws,
        "duplicate_draws_excluded": duplicate_draws,
        "configuration_reference_available": bool(reference_config_available),
        "configuration_mismatch_draws": configuration_mismatch_draws,
        "reference_configuration": {
            field: reference_config.get(field)
            for field in _FIXED_CONFIG_FIELDS
        } if reference_config_available else None,
        "out_of_window_draws_ignored": out_of_window_draws,
        "primary_endpoint": {
            "name": "每注平均命中数差值（V5.1－同期开奖随机对照）",
            "estimate": primary_estimate,
            "bootstrap_95_ci": primary_ci,
            "bootstrap_repetitions": int(bootstrap_repetitions) if complete_horizon else 0,
            "bootstrap_seed": int(bootstrap_seed) if complete_horizon else None,
            "interpretation": interpretation,
        },
        "secondary_metrics_policy": "单注3+率、整批3+率、Brier与Log Loss仅作描述性报告，不据此提前停止或选择模型。",
        "uniform_probability_baseline": {
            "per_number_probability": P_UNIFORM,
            "brier": float(UNIFORM_BRIER),
            "log_loss": float(UNIFORM_LOG_LOSS),
        },
        "no_early_stopping": True,
        "no_best_of_multiple_locks": True,
        "timestamp_anchor_policy": {
            "draw_1251": "A hash-only GitHub commitment exists; verify its commit time against the actual draw date in the manually imported history.",
            "draws_1252_to_1300": "Event created_at_utc values are self-reported until separate pre-draw hash-only repository commitments are recorded.",
            "warning": "A completed 50-draw statistical interval is not proof of pre-draw locking. Do not claim proven prospective advantage until every included draw has an independently verifiable pre-draw timestamp anchor.",
        },
    }
    return report, rows if not rows.empty else empty_per_draw, metric_table
