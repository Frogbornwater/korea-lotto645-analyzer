# -*- coding: utf-8 -*-
"""V5.3 repeated-seed random-control audit for prequential Lotto research.

This module consumes only a locally supplied walk-forward detail DataFrame.
It performs no network access and never retrieves official lottery results.
Each random replicate is a complete experiment spanning the same target draws,
with the same number of tickets per draw as the evaluated V5/V4 models.
"""
from __future__ import annotations

import ast
import hashlib
import json
from typing import Any, Callable

import numpy as np
import pandas as pd

V53_AUDIT_VERSION = "5.3.0-multiseed-random-control"
RNG_SALT = 5303

_REQUIRED_COLUMNS = {
    "目标期号",
    "训练结束期",
    "实际主号码",
    "V5_号码列表",
    "V4_号码列表",
    "V5_45号码概率_JSON",
    "V4_45号码概率_JSON",
}


def _parse_array(value: Any, label: str) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    elif isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            try:
                raw = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"{label}不是有效的号码数组。") from exc
    else:
        raise ValueError(f"{label}不是有效的号码数组。")
    if not isinstance(raw, (list, tuple)) or len(raw) != 6:
        raise ValueError(f"{label}必须包含6个号码。")
    try:
        numbers = [int(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}包含非整数号码。") from exc
    if len(set(numbers)) != 6 or any(number < 1 or number > 45 for number in numbers):
        raise ValueError(f"{label}必须是6个互不相同的1—45号码。")
    return sorted(numbers)


def _parse_tickets(value: Any, label: str) -> list[list[int]]:
    if isinstance(value, (list, tuple)):
        raw_tickets = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{label}为空。")
        raw_tickets = []
        for part in text.split("|"):
            try:
                raw_tickets.append(ast.literal_eval(part.strip()))
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"{label}包含无法解析的号码组合。") from exc
    else:
        raise ValueError(f"{label}格式无效。")
    if not raw_tickets:
        raise ValueError(f"{label}没有候选组合。")
    return [_parse_array(ticket, f"{label}第{index}注") for index, ticket in enumerate(raw_tickets, 1)]


def _parse_probabilities(value: Any, label: str) -> np.ndarray:
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}不是有效JSON。") from exc
    else:
        raw = value
    probabilities = np.asarray(raw, dtype=float)
    if probabilities.shape != (45,) or not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{label}必须包含45个有限概率值。")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(f"{label}必须全部位于0—1之间。")
    if not np.isclose(float(probabilities.sum()), 6.0, atol=1e-5):
        raise ValueError(f"{label}总和必须约等于6。")
    return probabilities


def _ticket_metrics(tickets: list[list[int]], actual: list[int]) -> dict[str, float]:
    actual_set = set(actual)
    hits = np.asarray([len(set(ticket).intersection(actual_set)) for ticket in tickets], dtype=float)
    return {
        "mean_hits": float(hits.mean()),
        "ticket_3plus": float(np.mean(hits >= 3)),
        "ticket_4plus": float(np.mean(hits >= 4)),
        "batch_3plus": float(np.any(hits >= 3)),
        "batch_4plus": float(np.any(hits >= 4)),
        "best_hits": float(hits.max()),
    }


def _probability_scores(probabilities: np.ndarray, actual: list[int]) -> tuple[float, float]:
    target = np.zeros(45, dtype=float)
    target[np.asarray(actual, dtype=int) - 1] = 1.0
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    brier = float(np.mean((clipped - target) ** 2))
    log_loss = float(-np.mean(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)))
    return brier, log_loss


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in original order."""
    if not p_values:
        return []
    count = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.ones(count, dtype=float)
    running_max = 0.0
    for rank, original_index in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[int(original_index)]))
        running_max = max(running_max, value)
        adjusted[int(original_index)] = running_max
    return [float(value) for value in adjusted]


def _uniform_ticket_batch(
    rng: np.random.Generator,
    repetitions: int,
    tickets_per_draw: int,
) -> np.ndarray:
    """Draw uniform 6-of-45 tickets; each replicate's within-draw tickets are unique."""
    keys = rng.random((repetitions, tickets_per_draw, 45))
    selected = np.argpartition(keys, kth=5, axis=-1)[..., :6]
    selected.sort(axis=-1)

    # Duplicate six-number combinations within one 10-ticket batch are extremely
    # rare, but reject them so every batch has the requested number of distinct
    # combinations just like the V5 constrained portfolio.
    for replicate in range(repetitions):
        seen: set[tuple[int, ...]] = set()
        for ticket_index in range(tickets_per_draw):
            ticket = tuple(int(item) + 1 for item in selected[replicate, ticket_index])
            while ticket in seen:
                replacement = rng.random(45)
                chosen = np.argpartition(replacement, kth=5)[:6]
                chosen.sort()
                selected[replicate, ticket_index] = chosen
                ticket = tuple(int(item) + 1 for item in chosen)
            seen.add(ticket)
    return selected


def run_multi_seed_random_audit(
    detail: pd.DataFrame,
    *,
    random_repetitions: int = 500,
    master_seed: int = 5302026,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Compare V5/V4 with repeated complete random experiments over the same draws.

    A replicate represents one full 200-draw (or supplied-range) random experiment:
    it generates the same number of unique 6-number tickets per target draw as V5.
    The returned empirical p-value is the one-sided fraction of random experiments
    whose metric is at least as good as the observed V5 metric, with the standard
    plus-one correction. Holm adjustment is applied across the six ticket metrics.
    """
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        raise ValueError("没有逐期回放明细，无法进行多随机种子审计。")
    missing = sorted(_REQUIRED_COLUMNS.difference(detail.columns))
    if missing:
        raise ValueError(
            "当前逐期明细缺少V5.3审计字段：" + "、".join(missing)
            + "。请使用V5.3版本重新运行逐期滚动回放。"
        )
    repetitions = int(random_repetitions)
    seed = int(master_seed)
    if not 10 <= repetitions <= 5000:
        raise ValueError("随机对照重复次数必须在10—5000之间。")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("主随机种子必须位于0—4294967295之间。")

    ordered = detail.sort_values("目标期号").reset_index(drop=True).copy()
    target_draws = pd.to_numeric(ordered["目标期号"], errors="raise").to_numpy(dtype=np.int64)
    train_ends = pd.to_numeric(ordered["训练结束期"], errors="raise").to_numpy(dtype=np.int64)
    if np.any(np.diff(target_draws) != 1):
        raise ValueError("审计要求目标期号连续且不重复。")
    if not np.array_equal(train_ends, target_draws - 1):
        raise ValueError("检测到训练结束期不等于目标期前一期，拒绝审计以防未来数据泄漏。")

    prepared: list[dict[str, Any]] = []
    tickets_per_draw: int | None = None
    v5_per_draw: list[dict[str, float]] = []
    v4_per_draw: list[dict[str, float]] = []
    v5_brier: list[float] = []
    v5_log_loss: list[float] = []
    v4_brier: list[float] = []
    v4_log_loss: list[float] = []
    uniform_brier: list[float] = []
    uniform_log_loss: list[float] = []

    for row_index, row in ordered.iterrows():
        target_draw = int(target_draws[row_index])
        actual = _parse_array(row["实际主号码"], f"第{target_draw}期实际主号码")
        v5_tickets = _parse_tickets(row["V5_号码列表"], f"第{target_draw}期V5组合")
        v4_tickets = _parse_tickets(row["V4_号码列表"], f"第{target_draw}期V4组合")
        if len(v5_tickets) != len(v4_tickets):
            raise ValueError(f"第{target_draw}期V5和V4注数不一致。")
        if tickets_per_draw is None:
            tickets_per_draw = len(v5_tickets)
        if len(v5_tickets) != tickets_per_draw:
            raise ValueError("各期候选注数不一致，不能进行配对随机对照。")
        v5_prob = _parse_probabilities(row["V5_45号码概率_JSON"], f"第{target_draw}期V5概率")
        v4_prob = _parse_probabilities(row["V4_45号码概率_JSON"], f"第{target_draw}期V4概率")
        v5_score = _ticket_metrics(v5_tickets, actual)
        v4_score = _ticket_metrics(v4_tickets, actual)
        v5_per_draw.append(v5_score)
        v4_per_draw.append(v4_score)
        v5_brier.append(_probability_scores(v5_prob, actual)[0])
        v5_log_loss.append(_probability_scores(v5_prob, actual)[1])
        v4_brier.append(_probability_scores(v4_prob, actual)[0])
        v4_log_loss.append(_probability_scores(v4_prob, actual)[1])
        uniform_scores = _probability_scores(np.full(45, 6.0 / 45.0), actual)
        uniform_brier.append(uniform_scores[0])
        uniform_log_loss.append(uniform_scores[1])
        prepared.append({
            "target_draw": target_draw,
            "actual": actual,
            "actual_mask": np.isin(np.arange(1, 46), np.asarray(actual, dtype=int)),
        })

    metric_names = ("mean_hits", "ticket_3plus", "ticket_4plus", "batch_3plus", "batch_4plus", "best_hits")
    metric_labels = {
        "mean_hits": "平均命中数",
        "ticket_3plus": "单注3+率",
        "ticket_4plus": "单注4+率",
        "batch_3plus": "整批至少一注3+率",
        "batch_4plus": "整批至少一注4+率",
        "best_hits": "整批最佳命中数平均",
    }
    v5_values = {name: float(np.mean([item[name] for item in v5_per_draw])) for name in metric_names}
    v4_values = {name: float(np.mean([item[name] for item in v4_per_draw])) for name in metric_names}

    random_totals = {name: np.zeros(repetitions, dtype=float) for name in metric_names}
    rng_seed_records: list[dict[str, int]] = []
    for index, prepared_row in enumerate(prepared, 1):
        target_draw = int(prepared_row["target_draw"])
        draw_seed = int(np.random.SeedSequence([seed, target_draw, RNG_SALT]).generate_state(1)[0])
        rng = np.random.default_rng(draw_seed)
        random_tickets = _uniform_ticket_batch(rng, repetitions, int(tickets_per_draw))
        hits = prepared_row["actual_mask"][random_tickets].sum(axis=-1).astype(np.int8)
        random_totals["mean_hits"] += hits.mean(axis=1)
        random_totals["ticket_3plus"] += (hits >= 3).mean(axis=1)
        random_totals["ticket_4plus"] += (hits >= 4).mean(axis=1)
        random_totals["batch_3plus"] += (hits >= 3).any(axis=1).astype(float)
        random_totals["batch_4plus"] += (hits >= 4).any(axis=1).astype(float)
        random_totals["best_hits"] += hits.max(axis=1)
        rng_seed_records.append({"目标期号": target_draw, "随机流种子": draw_seed})
        if progress_callback:
            progress_callback(index, len(prepared))

    random_results = pd.DataFrame({
        "随机实验编号": np.arange(1, repetitions + 1, dtype=int),
        "主随机种子": seed,
        **{
            name: random_totals[name] / len(prepared)
            for name in metric_names
        },
    })

    raw_p_values: list[float] = []
    summary_rows: list[dict[str, Any]] = []
    for name in metric_names:
        values = random_results[name].to_numpy(dtype=float)
        observed = v5_values[name]
        raw_p = float((1 + np.count_nonzero(values >= observed - 1e-12)) / (repetitions + 1))
        raw_p_values.append(raw_p)
        summary_rows.append({
            "指标": metric_labels[name],
            "V5观察值": observed,
            "V4冻结模型观察值": v4_values[name],
            "随机基准均值": float(values.mean()),
            "随机基准标准差": float(values.std(ddof=1)) if repetitions > 1 else 0.0,
            "随机基准P2.5": float(np.quantile(values, 0.025)),
            "随机基准P97.5": float(np.quantile(values, 0.975)),
            "V5减随机均值": observed - float(values.mean()),
            "随机实验中达到或超过V5的比例": float(np.mean(values >= observed - 1e-12)),
            "V5随机百分位": float(100.0 * np.mean(values < observed)),
            "单侧经验p值": raw_p,
        })
    adjusted = _holm_adjust(raw_p_values)
    for row, adjusted_p in zip(summary_rows, adjusted):
        row["Holm校正p值"] = adjusted_p

    probability_summary_rows = [
        {
            "指标": "概率Brier分数（越低越好）",
            "V5观察值": float(np.mean(v5_brier)),
            "V4冻结模型观察值": float(np.mean(v4_brier)),
            "随机基准均值": float(np.mean(uniform_brier)),
            "随机基准标准差": 0.0,
            "随机基准P2.5": float(np.mean(uniform_brier)),
            "随机基准P97.5": float(np.mean(uniform_brier)),
            "V5减随机均值": float(np.mean(v5_brier) - np.mean(uniform_brier)),
            "随机实验中达到或超过V5的比例": np.nan,
            "V5随机百分位": np.nan,
            "单侧经验p值": np.nan,
            "Holm校正p值": np.nan,
        },
        {
            "指标": "概率Log Loss（越低越好）",
            "V5观察值": float(np.mean(v5_log_loss)),
            "V4冻结模型观察值": float(np.mean(v4_log_loss)),
            "随机基准均值": float(np.mean(uniform_log_loss)),
            "随机基准标准差": 0.0,
            "随机基准P2.5": float(np.mean(uniform_log_loss)),
            "随机基准P97.5": float(np.mean(uniform_log_loss)),
            "V5减随机均值": float(np.mean(v5_log_loss) - np.mean(uniform_log_loss)),
            "随机实验中达到或超过V5的比例": np.nan,
            "V5随机百分位": np.nan,
            "单侧经验p值": np.nan,
            "Holm校正p值": np.nan,
        },
    ]
    summary = pd.DataFrame(summary_rows + probability_summary_rows)

    fingerprint_source = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    manifest = {
        "audit_version": V53_AUDIT_VERSION,
        "random_repetitions": repetitions,
        "master_seed": seed,
        "rng_bit_generator": "PCG64",
        "rng_seed_derivation": "SeedSequence([master_seed, target_draw, 5303])",
        "random_ticket_sampling": "uniform 6-of-45 by six smallest IID continuous random keys; no duplicate ticket within one batch",
        "target_start_draw": int(target_draws[0]),
        "target_end_draw": int(target_draws[-1]),
        "target_draws": int(len(prepared)),
        "tickets_per_draw": int(tickets_per_draw),
        "total_random_tickets_per_replicate": int(len(prepared) * int(tickets_per_draw)),
        "detail_sha256": hashlib.sha256(fingerprint_source).hexdigest(),
        "training_end_alignment_check": "passed: every training end equals target draw minus one",
        "ticket_metrics_multiple_testing": "Holm-Bonferroni across six ticket metrics",
        "evidence_status": "retrospective_exploratory_only",
        "interpretation": (
            "Each random replicate covers the complete target interval. The empirical p-value "
            "measures how often a random full-interval experiment matches or exceeds V5; it is "
            "not a probability of winning and does not establish prospective predictive advantage."
        ),
    }
    return {
        "audit_version": V53_AUDIT_VERSION,
        "summary": summary,
        "random_distribution": random_results,
        "rng_seed_records": pd.DataFrame(rng_seed_records),
        "manifest": manifest,
    }
