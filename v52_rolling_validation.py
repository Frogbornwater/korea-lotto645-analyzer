# -*- coding: utf-8 -*-
"""V5.2 rolling-validation helpers for manual, pre-draw Lotto research.

All draw history is supplied by the user. This module performs no network
access and never retrieves official lottery results.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from lotto_core import NUMBER_COLS, validate_frame
from v40_governance import configuration_fingerprint, data_fingerprint, make_event
from v40_store import validate_event
from v50_predictive_engine import V50_ENGINE_VERSION, generate_tickets


V51_LOCK_EVENT_TYPE = "v51_forward_lock"
V51_STRATEGY = "V5.1联合概率选号"
RANDOM_STRATEGY = "均匀随机基准"


def _ticket_numbers(value: Any) -> list[int]:
    if isinstance(value, str):
        parts = value.replace("·", " ").replace("-", " ").split()
        numbers = [int(part) for part in parts]
    else:
        numbers = [int(number) for number in value]
    if len(numbers) != 6 or len(set(numbers)) != 6 or any(n < 1 or n > 45 for n in numbers):
        raise ValueError("每注必须包含6个互不相同的1—45号码。")
    return sorted(numbers)


def build_v51_forward_lock_event(
    history: pd.DataFrame,
    prediction: dict[str, Any],
    candidate_frame: pd.DataFrame,
    *,
    seed: int,
    pair_strength: float,
    requested_overlap: int,
    effective_overlap: int,
) -> dict[str, Any]:
    """Create an append-only pre-draw lock for V5.1 candidates and matched random tickets."""
    clean = validate_frame(history).reset_index(drop=True)
    if clean.empty:
        raise ValueError("历史数据为空，不能锁定预测。")
    draw_numbers = clean["draw_no"].to_numpy(dtype=np.int64)
    if len(draw_numbers) > 1 and np.any(np.diff(draw_numbers) != 1):
        raise ValueError("历史期号不连续，不能锁定V5.1前瞻预测。")
    history_last = int(draw_numbers[-1])
    target_draw = int(prediction.get("target_draw", -1))
    if int(prediction.get("history_last_draw", -1)) != history_last or target_draw != history_last + 1:
        raise ValueError("预测目标期或训练数据末期与当前历史不一致，拒绝锁定。")
    if prediction.get("engine_version") != V50_ENGINE_VERSION:
        raise ValueError("V5预测引擎版本不匹配，拒绝锁定。")
    if not isinstance(candidate_frame, pd.DataFrame) or candidate_frame.empty or "号码" not in candidate_frame:
        raise ValueError("没有可锁定的V5.1候选组合。")

    probability_frame = prediction.get("number_probabilities")
    if not isinstance(probability_frame, pd.DataFrame) or not {"号码", "预测入选概率"}.issubset(probability_frame.columns):
        raise ValueError("预测结果缺少45个号码的入选概率。")
    probability_frame = probability_frame.sort_values("号码").reset_index(drop=True)
    probability_numbers = pd.to_numeric(probability_frame["号码"], errors="raise").to_numpy(dtype=int)
    probabilities = pd.to_numeric(probability_frame["预测入选概率"], errors="raise").to_numpy(dtype=float)
    if not np.array_equal(probability_numbers, np.arange(1, 46)):
        raise ValueError("号码概率表必须完整包含1—45号。")
    if not np.all(np.isfinite(probabilities)) or not np.isclose(float(probabilities.sum()), 6.0, atol=1e-5):
        raise ValueError("45个号码的概率必须为有限值且总和约为6。")

    candidate_records: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(candidate_frame.iterrows(), 1):
        numbers = _ticket_numbers(row["号码"])
        record: dict[str, Any] = {
            "策略": V51_STRATEGY,
            "注序": index,
            "号码": " · ".join(f"{number:02d}" for number in numbers),
            **{f"n{position}": number for position, number in enumerate(numbers, 1)},
        }
        for column in ("模型相对得分", "六号码边际概率合计", "共现调整项"):
            if column in candidate_frame.columns and pd.notna(row[column]):
                record[column] = float(row[column])
        candidate_records.append(record)

    if not candidate_records:
        raise ValueError("V5.1候选组合为空。")
    random_seed = int(seed) ^ 0x5A17
    random_tickets = generate_tickets(
        np.full(45, 6.0 / 45.0, dtype=float),
        None,
        count=len(candidate_records),
        seed=random_seed,
        pair_strength=0.0,
    )
    random_records = []
    for index, numbers in enumerate(random_tickets, 1):
        random_records.append({
            "策略": RANDOM_STRATEGY,
            "注序": index,
            "号码": " · ".join(f"{number:02d}" for number in numbers),
            **{f"n{position}": number for position, number in enumerate(numbers, 1)},
        })

    configuration = {
        "model": V51_STRATEGY,
        "engine_version": V50_ENGINE_VERSION,
        "training_window": int(prediction["training_window"]),
        "ridge_alpha": float(prediction["ridge_alpha"]),
        "pair_strength": float(pair_strength),
        "requested_overlap": int(requested_overlap),
        "effective_overlap": int(effective_overlap),
        "ticket_count": len(candidate_records),
        "seed": int(seed),
    }
    fingerprint = data_fingerprint(clean)
    payload = {
        "metadata": {
            "target_draw": target_draw,
            "history_first_draw": int(draw_numbers[0]),
            "history_last_draw": history_last,
            "history_rows": int(len(clean)),
            "history_fingerprint_sha256": fingerprint,
            "engine_version": V50_ENGINE_VERSION,
            "training_window": int(prediction["training_window"]),
            "ridge_alpha": float(prediction["ridge_alpha"]),
            "pair_strength": float(pair_strength),
            "requested_overlap": int(requested_overlap),
            "effective_overlap": int(effective_overlap),
            "overlap_relaxed": int(effective_overlap) > int(requested_overlap),
            "seed": int(seed),
            "random_seed": int(random_seed),
            "configuration_fingerprint_sha256": configuration_fingerprint(configuration),
            "probability_sum": float(probabilities.sum()),
            "ticket_count": len(candidate_records),
        },
        "configuration": configuration,
        "number_probabilities": [
            {"号码": int(number), "预测入选概率": float(probability)}
            for number, probability in zip(probability_numbers, probabilities)
        ],
        "tickets": candidate_records + random_records,
    }
    return make_event(V51_LOCK_EVENT_TYPE, fingerprint, payload)


def event_tickets_frame(event: dict[str, Any]) -> pd.DataFrame:
    """Return the locked candidate and random-control tickets as a CSV-ready frame."""
    validated = validate_event(event)
    if validated.get("event_type") != V51_LOCK_EVENT_TYPE:
        raise ValueError("该事件不是V5.1前瞻锁定记录。")
    records = validated["payload"].get("tickets")
    if not isinstance(records, list) or not records:
        raise ValueError("V5.1前瞻锁定记录没有号码明细。")
    return pd.DataFrame(records)


def evaluate_v51_forward_lock_events(
    history: pd.DataFrame,
    events: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score imported V5.1 locks only after target draws are present in manual history.

    The exact historical prefix used to create each lock must still match its
    stored SHA-256 fingerprint before the result is accepted.
    """
    clean = validate_frame(history).reset_index(drop=True)
    if clean.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    draw_map = {
        int(row["draw_no"]): {int(row[column]) for column in NUMBER_COLS}
        for _, row in clean.iterrows()
    }
    current_last = int(clean["draw_no"].max())
    status_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for raw_event in events:
        if not isinstance(raw_event, dict) or raw_event.get("event_type") != V51_LOCK_EVENT_TYPE:
            continue
        try:
            event = validate_event(raw_event)
            payload = event["payload"]
            metadata = payload["metadata"]
            target_draw = int(metadata["target_draw"])
            history_last = int(metadata["history_last_draw"])
            event_created = str(event["created_at_utc"])
            tickets = payload["tickets"]
        except (ValueError, KeyError, TypeError):
            status_rows.append({"目标期号": None, "状态": "锁定档案格式无效"})
            continue

        base_status = {
            "目标期号": target_draw,
            "锁定时间UTC": event_created,
            "锁定时历史末期": history_last,
            "数据指纹前缀": str(metadata.get("history_fingerprint_sha256", ""))[:12],
        }
        if target_draw <= history_last:
            status_rows.append({**base_status, "状态": "无效：目标期必须晚于锁定历史末期"})
            continue
        if target_draw not in draw_map:
            status_rows.append({**base_status, "状态": "等待开奖数据"})
            continue

        prefix = clean[clean["draw_no"] <= history_last].copy()
        try:
            prefix_fingerprint = data_fingerprint(prefix)
        except ValueError:
            prefix_fingerprint = ""
        if prefix_fingerprint != metadata.get("history_fingerprint_sha256"):
            status_rows.append({**base_status, "状态": "拒绝评分：锁定时历史前缀指纹不匹配"})
            continue

        actual = draw_map[target_draw]
        if len(actual) != 6:
            status_rows.append({**base_status, "状态": "拒绝评分：目标期主号码无效"})
            continue
        per_strategy: dict[str, list[int]] = {}
        valid = True
        for record in tickets:
            try:
                strategy = str(record["策略"])
                numbers = _ticket_numbers([record[f"n{i}"] for i in range(1, 7)])
                ticket_no = int(record["注序"])
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            hits = len(set(numbers) & actual)
            hit_numbers = sorted(set(numbers) & actual)
            per_strategy.setdefault(strategy, []).append(hits)
            detail_rows.append({
                "目标期号": target_draw,
                "策略": strategy,
                "注序": ticket_no,
                "号码": " · ".join(f"{number:02d}" for number in numbers),
                "命中号码": " · ".join(f"{number:02d}" for number in hit_numbers),
                "命中数": hits,
                "命中3个以上": hits >= 3,
                "命中4个以上": hits >= 4,
                "模型相对得分": record.get("模型相对得分"),
            })
        if not valid or V51_STRATEGY not in per_strategy or RANDOM_STRATEGY not in per_strategy:
            status_rows.append({**base_status, "状态": "拒绝评分：锁定号码明细不完整"})
            continue

        summary_by_strategy: dict[str, dict[str, float]] = {}
        for strategy, hits_list in per_strategy.items():
            hits = np.asarray(hits_list, dtype=float)
            summary_by_strategy[strategy] = {
                "mean_hits": float(hits.mean()),
                "ticket_3plus": float(np.mean(hits >= 3)),
                "ticket_4plus": float(np.mean(hits >= 4)),
                "batch_3plus": float(np.any(hits >= 3)),
                "batch_4plus": float(np.any(hits >= 4)),
                "best_hits": float(hits.max()),
            }

        v51 = summary_by_strategy[V51_STRATEGY]
        random = summary_by_strategy[RANDOM_STRATEGY]
        probability_rows = payload.get("number_probabilities", [])
        brier = float("nan")
        log_loss = float("nan")
        try:
            probability_frame = pd.DataFrame(probability_rows).sort_values("号码")
            p = pd.to_numeric(probability_frame["预测入选概率"], errors="raise").to_numpy(dtype=float)
            number_ids = pd.to_numeric(probability_frame["号码"], errors="raise").to_numpy(dtype=int)
            if len(p) == 45 and np.array_equal(number_ids, np.arange(1, 46)) and np.all(np.isfinite(p)):
                p = np.clip(p, 1e-9, 1.0 - 1e-9)
                y = np.zeros(45, dtype=float)
                y[np.asarray(sorted(actual), dtype=int) - 1] = 1.0
                brier = float(np.mean((p - y) ** 2))
                log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        except (KeyError, TypeError, ValueError):
            pass

        summary_rows.append({
            "目标期号": target_draw,
            "V5.1平均命中数": v51["mean_hits"],
            "随机平均命中数": random["mean_hits"],
            "平均命中差值": v51["mean_hits"] - random["mean_hits"],
            "V5.1单注3+率": v51["ticket_3plus"],
            "随机单注3+率": random["ticket_3plus"],
            "V5.1整批至少一注3+": v51["batch_3plus"],
            "随机整批至少一注3+": random["batch_3plus"],
            "V5.1最佳命中数": v51["best_hits"],
            "随机最佳命中数": random["best_hits"],
            "V5.1_Brier": brier,
            "V5.1_LogLoss": log_loss,
        })
        status_rows.append({**base_status, "状态": "已评分"})

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows), pd.DataFrame(status_rows)
