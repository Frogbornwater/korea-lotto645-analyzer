# -*- coding: utf-8 -*-
"""V4.0 governance primitives: immutable data fingerprints, strict append and evidence gates.

This module never retrieves draw data. All draw history must be supplied manually.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from lotto_core import COLUMNS, NUMBER_COLS, quality_report, validate_frame

RANDOM_MODEL = "均匀随机基准"
PRIMARY_FORWARD_METRICS = ("平均命中数", "单注3+率", "整批3+率")


def canonical_history(frame: pd.DataFrame) -> pd.DataFrame:
    clean = validate_frame(frame)
    if clean.empty:
        raise ValueError("历史数据为空，不能生成数据指纹。")
    return clean[COLUMNS].sort_values("draw_no").reset_index(drop=True)


def data_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint a canonical full or sliced history dataframe."""
    clean = canonical_history(frame)
    payload = clean.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    payload = json.dumps(configuration, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append_draw(
    frame: pd.DataFrame,
    draw_no: int,
    draw_date: str,
    numbers: list[int] | tuple[int, ...],
    bonus: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append exactly one next draw; duplicate-identical submissions are idempotent."""
    clean = canonical_history(frame)
    draw_no = int(draw_no)
    bonus = int(bonus)
    nums = [int(value) for value in numbers]
    if len(nums) != 6 or len(set(nums)) != 6 or any(value < 1 or value > 45 for value in nums):
        raise ValueError("主号码必须为6个互不相同的1—45整数。")
    if bonus < 1 or bonus > 45 or bonus in nums:
        raise ValueError("Bonus 必须是1—45之间且不在六个主号码中的整数。")
    try:
        parsed_date = pd.to_datetime(str(draw_date), format="%Y-%m-%d", errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("开奖日期必须使用 YYYY-MM-DD 格式。") from exc
    normalized_date = parsed_date.strftime("%Y-%m-%d")
    nums = sorted(nums)
    existing = clean[clean["draw_no"] == draw_no]
    candidate = {
        "draw_no": draw_no, "date": normalized_date,
        **{f"n{i}": nums[i - 1] for i in range(1, 7)}, "bonus": bonus,
    }
    if not existing.empty:
        old = existing.iloc[0].to_dict()
        old_values = {key: ("" if pd.isna(old[key]) else old[key]) for key in COLUMNS}
        same = all(str(old_values[key]) == str(candidate[key]) for key in COLUMNS)
        if same:
            return clean, {"status": "already_exists", "draw_no": draw_no, "rows": len(clean)}
        raise ValueError(f"第{draw_no}期已经存在，但内容不一致；拒绝覆盖历史记录。")
    expected = int(clean["draw_no"].max()) + 1
    if draw_no != expected:
        raise ValueError(f"只能追加下一期。当前最新期为{expected - 1}，应追加第{expected}期。")
    result = pd.concat([clean, pd.DataFrame([candidate], columns=COLUMNS)], ignore_index=True)
    result = validate_frame(result)
    report = quality_report(result)
    if not report["sequence_complete"] or report["missing_draws"]:
        raise ValueError("追加后期号序列不连续，已取消更新。")
    return result, {"status": "appended", "draw_no": draw_no, "rows": len(result)}


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p值必须是0—1之间的有限数值。")
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    n = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (n - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_draw_bootstrap(
    model_values: list[float] | np.ndarray,
    random_values: list[float] | np.ndarray,
    *,
    repetitions: int = 5000,
    block_length: int = 3,
    seed: int = 2040,
) -> dict[str, Any]:
    """Moving-block bootstrap over draw-level paired differences.

    Each observation is a draw-level aggregate, never an individual ticket.
    The p-value is a centered bootstrap diagnostic; it is not a causal claim.
    """
    model = np.asarray(model_values, dtype=float)
    baseline = np.asarray(random_values, dtype=float)
    if model.ndim != 1 or baseline.ndim != 1 or len(model) != len(baseline):
        raise ValueError("模型与随机基准必须是长度相同的一维逐期开奖指标。")
    if len(model) < 2 or not np.all(np.isfinite(model)) or not np.all(np.isfinite(baseline)):
        raise ValueError("配对检验至少需要2期有限数值。")
    if repetitions < 1000:
        raise ValueError("Bootstrap repetitions 至少为1000。")
    differences = model - baseline
    n = len(differences)
    block = max(1, min(int(block_length), n))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    samples = np.empty(int(repetitions), dtype=float)
    offsets = np.arange(block)
    for index in range(int(repetitions)):
        starts = rng.integers(0, n, size=n_blocks)
        sampled = np.concatenate([(start + offsets) % n for start in starts])[:n]
        samples[index] = float(np.mean(differences[sampled]))
    observed = float(np.mean(differences))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    centered = samples - observed
    p_value = float((1 + np.count_nonzero(np.abs(centered) >= abs(observed))) / (repetitions + 1))
    return {
        "n_draws": int(n),
        "mean_difference": observed,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_p_raw": p_value,
        "block_length": block,
        "repetitions": int(repetitions),
    }


def evaluate_forward_pairs(
    scored_detail: pd.DataFrame,
    model_name: str,
    *,
    random_name: str = RANDOM_MODEL,
    repetitions: int = 5000,
    seed: int = 2040,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare a locked model with the matched random control by draw."""
    required = {"目标期号", "策略", "随机种子组", "命中数", "命中3个以上"}
    if not required.issubset(scored_detail.columns):
        raise ValueError("前瞻复盘明细缺少配对比较所需字段。")
    work = scored_detail[scored_detail["策略"].isin([model_name, random_name])].copy()
    if work.empty or set(work["策略"].unique()) != {model_name, random_name}:
        return pd.DataFrame(), {"status": "等待模型与随机基准均有可用记录", "n_draws": 0}
    work["命中数"] = pd.to_numeric(work["命中数"], errors="raise")
    work["命中3个以上"] = pd.to_numeric(work["命中3个以上"], errors="raise")
    per_draw = work.groupby(["目标期号", "策略"], sort=True).agg(
        平均命中数=("命中数", "mean"),
        单注3加率=("命中3个以上", "mean"),
    ).reset_index()
    batch = work.groupby(["目标期号", "策略", "随机种子组"], sort=True)["命中3个以上"].any().groupby(
        level=["目标期号", "策略"]
    ).mean().rename("整批3加率").reset_index()
    per_draw = per_draw.merge(batch, on=["目标期号", "策略"], how="left")
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    metric_columns = [
        ("平均命中数", "平均命中数"),
        ("单注3+率", "单注3加率"),
        ("整批3+率", "整批3加率"),
    ]
    for metric_index, (label, column) in enumerate(metric_columns):
        pivot = per_draw.pivot(index="目标期号", columns="策略", values=column).dropna()
        if pivot.empty:
            continue
        result = paired_draw_bootstrap(
            pivot[model_name].to_numpy(), pivot[random_name].to_numpy(),
            repetitions=repetitions, block_length=3, seed=seed + metric_index,
        )
        p_values.append(result["bootstrap_p_raw"])
        rows.append({
            "指标": label,
            "开奖期数": result["n_draws"],
            "模型均值": float(pivot[model_name].mean()),
            "随机基准均值": float(pivot[random_name].mean()),
            "模型减随机": result["mean_difference"],
            "95%CI下限": result["ci95_lower"],
            "95%CI上限": result["ci95_upper"],
            "Bootstrap原始p值": result["bootstrap_p_raw"],
            "区块长度": result["block_length"],
        })
    if not rows:
        return pd.DataFrame(), {"status": "没有可配对的完整开奖期", "n_draws": 0}
    adjusted = _holm_adjust(p_values)
    for row, p_adj in zip(rows, adjusted):
        row["Holm校正p值"] = p_adj
    report = pd.DataFrame(rows)
    n_draws = int(report["开奖期数"].min())
    passed = bool(
        n_draws >= 100
        and (report["模型减随机"] > 0).all()
        and (report["95%CI下限"] > 0).all()
        and (report["Holm校正p值"] < 0.05).all()
    )
    gate = {
        "status": "达到候选晋升证据门槛（仍需持续监测）" if passed else "尚未达到正式模型晋升门槛",
        "n_draws": n_draws,
        "minimum_draws": 100,
        "all_primary_effects_positive": bool((report["模型减随机"] > 0).all()),
        "all_confidence_intervals_above_zero": bool((report["95%CI下限"] > 0).all()),
        "all_holm_adjusted_p_below_005": bool((report["Holm校正p值"] < 0.05).all()),
        "promotion_allowed": passed,
        "warning": "本门槛是保守的研究治理规则，不代表公平独立开奖中的预测优势保证。",
    }
    return report, gate


def make_event(event_type: str, data_fingerprint_value: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not event_type.strip():
        raise ValueError("事件类型不能为空。")
    if not data_fingerprint_value or len(data_fingerprint_value) != 64:
        raise ValueError("事件必须关联有效的SHA-256数据指纹。")
    return {
        "schema_version": 1,
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_fingerprint_sha256": data_fingerprint_value,
        "payload": payload,
    }
