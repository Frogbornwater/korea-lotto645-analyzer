# -*- coding: utf-8 -*-
"""V9.4 experimental multi-timescale smart picker.

Uses only caller-supplied local draw history. This is an experimental
candidate generator, not evidence of improved lottery predictability.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from lotto_core import NUMBER_COLS, validate_frame
from v93_multiscale_prequential import ALPHA, WINDOWS, _forecast
from v51_smart_picker import generate_constrained_tickets

ENGINE_VERSION = "9.4.0-multiscale-experimental"


def fit_v94_multiscale_next(history: pd.DataFrame) -> dict[str, Any]:
    """Fit the fixed V9.3 equal-weight multi-window probability ensemble."""
    clean = validate_frame(history).sort_values("draw_no").reset_index(drop=True)
    if len(clean) < max(WINDOWS) + 1:
        raise ValueError("V9.4多尺度选号至少需要1001期连续有效历史数据。")
    draws = clean["draw_no"].to_numpy(dtype=np.int64)
    if np.any(np.diff(draws) != 1):
        raise ValueError("历史期号存在缺失，不能运行V9.4多尺度选号。")
    matrix = clean[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)
    forecasts = {window: _forecast(matrix.tolist(), window, ALPHA) for window in WINDOWS}
    probabilities = np.mean(np.stack(list(forecasts.values())), axis=0)
    if probabilities.shape != (45,) or not np.all(np.isfinite(probabilities)):
        raise RuntimeError("V9.4概率向量无效。")
    if not np.isclose(float(probabilities.sum()), 6.0, atol=1e-8):
        raise RuntimeError("V9.4的45维边际概率之和不等于6。")
    probability_frame = pd.DataFrame({
        "号码": np.arange(1, 46, dtype=int),
        "预测入选概率": probabilities,
        "均匀随机基准概率": 6.0 / 45.0,
        "相对基准差": probabilities - 6.0 / 45.0,
    }).sort_values(["预测入选概率", "号码"], ascending=[False, True]).reset_index(drop=True)
    return {
        "model_name": "V9.3等权多尺度概率模型",
        "engine_version": ENGINE_VERSION,
        "target_draw": int(draws[-1]) + 1,
        "history_first_draw": int(draws[0]),
        "history_last_draw": int(draws[-1]),
        "history_rows": int(len(clean)),
        "training_windows": list(WINDOWS),
        "dirichlet_alpha": ALPHA,
        "probability_sum": float(probabilities.sum()),
        "number_probabilities": probability_frame,
        "interpretation": "固定窗口20至1000期的等权边际概率融合；研究版本尚未证明优于均匀概率基准。",
    }


def generate_v94_tickets(
    prediction: dict[str, Any],
    count: int = 10,
    seed: int = 2026,
    max_overlap: int = 4,
) -> tuple[pd.DataFrame, int, bool]:
    """Generate a diversified candidate batch without pairwise lift adjustment."""
    frame = prediction["number_probabilities"].sort_values("号码")
    probabilities = frame["预测入选概率"].to_numpy(dtype=float)
    tickets, effective_overlap, relaxed = generate_constrained_tickets(
        probabilities, pair_scores=None, count=count, seed=seed,
        pair_strength=0.0, max_overlap=max_overlap,
    )
    logits = np.log(np.clip(probabilities, 1e-9, 1.0 - 1e-9) /
                    np.clip(1.0 - probabilities, 1e-9, 1.0))
    rows = []
    for index, ticket in enumerate(tickets, 1):
        marginal_sum = float(sum(probabilities[number - 1] for number in ticket))
        score = float(sum(logits[number - 1] for number in ticket))
        rows.append({
            "组合": f"{index:02d}",
            "号码": " · ".join(f"{number:02d}" for number in ticket),
            "和值": int(sum(ticket)),
            "奇数个数": int(sum(number % 2 for number in ticket)),
            "六号码边际概率合计": marginal_sum,
            "模型相对分": score,
        })
    result = pd.DataFrame(rows).sort_values(
        ["模型相对分", "组合"], ascending=[False, True]
    ).reset_index(drop=True)
    result.insert(0, "排名", np.arange(1, len(result) + 1))
    return result, int(effective_overlap), bool(relaxed)
