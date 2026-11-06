# -*- coding: utf-8 -*-
"""Offline 10-ticket portfolio diversification heuristic.

This optimizes ticket-set coverage and overlap diagnostics only. It does not
change the probability of any individual six-number combination.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from backtest_engine import _make_weights, _passes_structure, _structure_bounds
from lotto_core import NUMBER_COLS, MODEL_NAMES, validate_frame


def _portfolio_metrics(tickets: list[tuple[int, ...]]) -> dict[str, Any]:
    cover = sorted({number for ticket in tickets for number in ticket})
    overlaps = [
        len(set(tickets[i]) & set(tickets[j]))
        for i in range(len(tickets)) for j in range(i + 1, len(tickets))
    ]
    return {
        "组合组数": len(tickets),
        "不同号码数": len(cover),
        "号码覆盖率": len(cover) / 45.0,
        "未覆盖号码": sorted(set(range(1, 46)) - set(cover)),
        "最大组间重号": max(overlaps, default=0),
        "平均组间重号": float(np.mean(overlaps)) if overlaps else 0.0,
        "两两组合比较数": len(overlaps),
    }


def optimize_portfolio(
    frame: pd.DataFrame,
    model_name: str = "多模型集成模型",
    count: int = 10,
    window: int = 100,
    max_overlap: int = 4,
    candidate_pool_size: int = 5000,
    quality_weight: float = 0.30,
    coverage_weight: float = 0.45,
    overlap_penalty: float = 0.25,
    half_life: float = 40.0,
    bayes_prior: float = 80.0,
    ensemble_weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
    multi_model_weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
    seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a model-weighted candidate pool, then greedily diversify 10 tickets."""
    clean = validate_frame(frame)
    if clean.empty:
        raise ValueError("没有历史数据，请先手动导入 CSV 或 Excel。")
    if model_name not in MODEL_NAMES:
        raise ValueError(f"未知模型：{model_name}")
    count = int(count)
    if count != 10:
        raise ValueError("V3.7 当前组合优化模块固定输出10组号码。")
    if not 0 <= int(max_overlap) <= 5:
        raise ValueError("最大组间重号限制必须在0—5之间。")
    candidate_pool_size = int(candidate_pool_size)
    if not 500 <= candidate_pool_size <= 30000:
        raise ValueError("候选池规模必须在500—30,000之间。")
    objective = np.asarray([quality_weight, coverage_weight, overlap_penalty], dtype=float)
    if np.any(objective < 0) or not np.isclose(objective.sum(), 1.0, atol=1e-8):
        raise ValueError("模型质量、覆盖、重叠惩罚三个目标权重必须非负且总和为1。")

    matrix = clean[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)
    weights = _make_weights(
        matrix, model_name, int(window), float(half_life), float(bayes_prior),
        tuple(float(x) for x in ensemble_weights), tuple(float(x) for x in multi_model_weights),
    )
    probs = np.asarray(weights, dtype=float)
    if probs.shape != (45,) or not np.all(np.isfinite(probs)) or np.any(probs <= 0):
        raise ValueError("模型返回了无效号码权重。")
    probs /= probs.sum()
    score_range = np.asarray(weights, dtype=float)
    lo, hi = float(score_range.min()), float(score_range.max())
    normalized = np.full(45, 0.5) if hi <= lo else (score_range - lo) / (hi - lo)
    bounds = _structure_bounds(matrix) if model_name == "历史结构约束模型" else None

    rng = np.random.default_rng(int(seed))
    pool: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = candidate_pool_size * 40
    for _ in range(max_attempts):
        ticket = tuple(sorted(int(x) for x in rng.choice(np.arange(1, 46), size=6, replace=False, p=probs)))
        if ticket in seen:
            continue
        if bounds is not None and not _passes_structure(ticket, bounds):
            continue
        seen.add(ticket)
        pool.append(ticket)
        if len(pool) >= candidate_pool_size:
            break
    if len(pool) < count:
        raise RuntimeError(f"只生成了{len(pool)}个不同候选组合，不足以选择10组；请增加候选池或更换模型。")

    candidate_quality = {
        ticket: float(np.mean([normalized[n - 1] for n in ticket])) for ticket in pool
    }
    requested_overlap = int(max_overlap)
    allowed_overlap = requested_overlap
    selected: list[tuple[int, ...]] = []
    selected_scores: list[float] = []
    remaining = set(pool)
    q_weight, c_weight, o_weight = (float(x) for x in objective)

    while len(selected) < count:
        feasible: list[tuple[float, tuple[int, ...], int]] = []
        for ticket in remaining:
            overlaps = [len(set(ticket) & set(chosen)) for chosen in selected]
            max_shared = max(overlaps, default=0)
            if max_shared > allowed_overlap:
                continue
            new_count = len(set(ticket) - {n for old in selected for n in old})
            mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
            score = (
                q_weight * candidate_quality[ticket]
                + c_weight * (new_count / 6.0)
                - o_weight * (mean_overlap / 6.0)
            )
            feasible.append((score, ticket, max_shared))
        if not feasible:
            if allowed_overlap >= 5:
                raise RuntimeError("候选池无法在最多共享5个号码的限制下凑齐10组。")
            allowed_overlap += 1
            continue
        feasible.sort(key=lambda item: (item[0], tuple(-n for n in item[1])), reverse=True)
        score, chosen, _ = feasible[0]
        selected.append(chosen)
        selected_scores.append(float(score))
        remaining.remove(chosen)

    report = _portfolio_metrics(selected)
    report.update({
        "模型": model_name,
        "历史训练截止期号": int(clean["draw_no"].max()),
        "历史训练期数": int(len(clean)),
        "近期窗口": int(window),
        "候选池规模": int(len(pool)),
        "请求最大组间重号": requested_overlap,
        "实际允许最大组间重号": allowed_overlap,
        "是否放宽重号限制": allowed_overlap != requested_overlap,
        "目标权重": {
            "模型号码评分": q_weight,
            "新增号码覆盖": c_weight,
            "组间重叠惩罚": o_weight,
        },
        "参数": {
            "half_life": float(half_life), "bayes_prior": float(bayes_prior),
            "ensemble_weights": list(ensemble_weights),
            "multi_model_weights": list(multi_model_weights),
            "seed": int(seed),
        },
        "解释边界": "这是根据历史统计权重和组合多样性做的启发式筛选；不改变任何单注组合的理论中奖概率。",
    })
    rows = []
    for idx, ticket in enumerate(selected, start=1):
        rows.append({
            "组号": f"{idx:02d}",
            "模型": model_name,
            "号码": " - ".join(f"{n:02d}" for n in ticket),
            "号码JSON": json.dumps(list(ticket), ensure_ascii=False),
            "和值": sum(ticket),
            "奇数个数": sum(n % 2 for n in ticket),
            "低区(1-15)": sum(n <= 15 for n in ticket),
            "中区(16-30)": sum(16 <= n <= 30 for n in ticket),
            "高区(31-45)": sum(n >= 31 for n in ticket),
            "号码相对评分": round(candidate_quality[ticket], 5),
            "贪心选择目标分": round(selected_scores[idx - 1], 5),
        })
    report["组合明细"] = rows
    return pd.DataFrame(rows), report
