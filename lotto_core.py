# -*- coding: utf-8 -*-
"""Lotto 6/45 V3 analysis engine.

Historical data describes past draws. No method here can reliably predict a fair,
independent future draw or improve the probability of any exact six-number ticket.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd

COLUMNS = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]
NUMBER_COLS = [f"n{i}" for i in range(1, 7)]
MODEL_NAMES = [
    "近期频率模型",
    "长期频率模型",
    "指数时间衰减模型",
    "贝叶斯平滑模型",
    "综合评分模型",
    "历史结构约束模型",
    "均匀随机基准",
    "多模型集成模型",
]


def normalize_date(value: Any) -> str:
    """Normalize a manually supplied draw date; no network access is performed."""
    raw = str(value or "").strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw[:10]


def validate_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize, validate and sort draw rows; reject conflicting duplicate rounds."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    tmp = frame.copy()
    aliases = {
        "회차": "draw_no", "期号": "draw_no", "期数": "draw_no", "开奖期数": "draw_no",
        "round": "draw_no", "draw": "draw_no", "drwNo": "draw_no", "ltEpsd": "draw_no",
        "추첨일": "date", "日期": "date", "开奖日期": "date", "开奖日": "date",
        "draw_date": "date", "drwNoDate": "date", "ltRflYmd": "date",
        "보너스": "bonus", "보너스번호": "bonus", "奖金号码": "bonus",
        "bonus_no": "bonus", "bonus_number": "bonus", "bnusNo": "bonus", "bnsWnNo": "bonus",
    }
    tmp = tmp.rename(columns={c: aliases[c] for c in tmp.columns if c in aliases})
    for i in range(1, 7):
        for alias in (f"drwtNo{i}", f"tm{i}WnNo", f"num{i}", f"number{i}", f"번호{i}", f"번호 {i}", f"号码{i}", f"당첨번호{i}", f"中奖号码{i}"):
            if alias in tmp.columns and f"n{i}" not in tmp.columns:
                tmp = tmp.rename(columns={alias: f"n{i}"})
    if "date" not in tmp.columns:
        tmp["date"] = ""
    required = ["draw_no", *NUMBER_COLS, "bonus"]
    missing = [c for c in required if c not in tmp.columns]
    if missing:
        raise ValueError("历史数据缺少字段：" + ", ".join(missing))
    for col in required:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    rows = []
    for _, row in tmp.iterrows():
        vals = [row[c] for c in required]
        if any(pd.isna(v) or not float(v).is_integer() for v in vals):
            continue
        draw_no = int(row["draw_no"])
        nums = sorted(int(row[c]) for c in NUMBER_COLS)
        bonus = int(row["bonus"])
        if draw_no < 1 or len(set(nums)) != 6 or any(n < 1 or n > 45 for n in nums):
            continue
        if bonus < 1 or bonus > 45 or bonus in nums:
            continue
        rec = {"draw_no": draw_no, "date": normalize_date(row.get("date", ""))}
        rec.update({f"n{i}": nums[i - 1] for i in range(1, 7)})
        rec["bonus"] = bonus
        rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    result = pd.DataFrame(rows, columns=COLUMNS).sort_values("draw_no").reset_index(drop=True)

    # Exact repeated records can be deduplicated; conflicting rows for one draw
    # are rejected instead of silently allowing "last row wins".
    duplicated = result[result.duplicated("draw_no", keep=False)]
    for draw_no, group in duplicated.groupby("draw_no"):
        compare_cols = list(COLUMNS)
        if len(group[compare_cols].drop_duplicates()) > 1:
            raise ValueError(f"第 {int(draw_no)} 期存在互相冲突的重复记录")
    return result.drop_duplicates("draw_no", keep="last").reset_index(drop=True)

def quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    raw = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    duplicate_count = int(raw["draw_no"].duplicated().sum()) if "draw_no" in raw.columns else 0
    clean = validate_frame(raw)
    if clean.empty:
        return {
            "rows": 0, "first_draw": None, "last_draw": None, "missing_draws": [],
            "date_errors": 0, "duplicates": duplicate_count, "invalid_rows": len(raw),
            "missing_before_first": None, "range_complete": False, "sequence_complete": False,
        }
    present = set(clean["draw_no"].astype(int))
    first, last = int(clean["draw_no"].min()), int(clean["draw_no"].max())
    # Separate gaps inside the supplied file from earlier rounds the user did not include.
    missing = [n for n in range(first, last + 1) if n not in present]
    date_errors = 0
    for value in clean["date"].astype(str):
        if not value:
            date_errors += 1
            continue
        try:
            pd.to_datetime(value, format="%Y-%m-%d", errors="raise")
        except (TypeError, ValueError):
            date_errors += 1
    return {
        "rows": len(clean), "first_draw": first, "last_draw": last,
        "missing_draws": missing, "date_errors": date_errors,
        "duplicates": duplicate_count, "invalid_rows": max(0, len(raw) - len(clean)),
        "missing_before_first": max(0, first - 1),
        "range_complete": not missing,
        "sequence_complete": first == 1 and not missing,
    }

def get_numbers(df: pd.DataFrame) -> list[list[int]]:
    clean = validate_frame(df)
    return clean[NUMBER_COLS].astype(int).values.tolist()


def frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    clean = validate_frame(df)
    counts = Counter(n for row in get_numbers(clean) for n in row)
    draws = len(clean)
    p = 6 / 45
    expected = draws * p
    sd = math.sqrt(draws * p * (1 - p)) if draws else 0.0
    rows = []
    for number in range(1, 46):
        observed = counts[number]
        z = (observed - expected) / sd if sd else 0.0
        rows.append({"号码": number, "出现次数": observed, "每期出现率": observed / draws if draws else 0.0,
                     "理论期望次数": round(expected, 2), "标准化偏差Z": round(z, 3)})
    return pd.DataFrame(rows).sort_values(["出现次数", "号码"], ascending=[False, True]).reset_index(drop=True)


def gap_table(df: pd.DataFrame) -> pd.DataFrame:
    clean = validate_frame(df)
    latest = int(clean["draw_no"].max()) if not clean.empty else 0
    last_seen = {n: None for n in range(1, 46)}
    for _, row in clean.iterrows():
        for n in (int(row[f"n{i}"]) for i in range(1, 7)):
            last_seen[n] = int(row["draw_no"])
    rows = []
    for n in range(1, 46):
        last = last_seen[n]
        rows.append({"号码": n, "距最近出现期数": latest - last if last is not None else len(clean)})
    return pd.DataFrame(rows).sort_values(["距最近出现期数", "号码"], ascending=[False, True]).reset_index(drop=True)


def draw_features(df: pd.DataFrame) -> pd.DataFrame:
    clean = validate_frame(df)
    rows = []
    previous: set[int] = set()
    for _, row in clean.iterrows():
        nums = [int(row[f"n{i}"]) for i in range(1, 7)]
        current = set(nums)
        rows.append({
            "期数": int(row["draw_no"]), "日期": row["date"], "和值": sum(nums),
            "奇数个数": sum(n % 2 for n in nums),
            "低区(1-15)": sum(n <= 15 for n in nums),
            "中区(16-30)": sum(16 <= n <= 30 for n in nums),
            "高区(31-45)": sum(n >= 31 for n in nums),
            "连号对数": sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1),
            "与上期重号数": len(current & previous),
        })
        previous = current
    return pd.DataFrame(rows)


def pair_table(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    counts: Counter[tuple[int, int]] = Counter()
    for nums in get_numbers(df):
        counts.update(combinations(nums, 2))
    return pd.DataFrame([{"号码组合": f"{a:02d} + {b:02d}", "共同出现次数": c}
                         for (a, b), c in counts.most_common(top_n)])


def _counts_for_window(df: pd.DataFrame, window: int | None) -> tuple[np.ndarray, int]:
    clean = validate_frame(df)
    sample = clean if window is None else clean.tail(min(window, len(clean)))
    counts = np.zeros(45, dtype=float)
    for nums in get_numbers(sample):
        for n in nums:
            counts[n - 1] += 1.0
    return counts, len(sample)


def model_weights(
    df: pd.DataFrame, model: str, window: int = 100, half_life: float = 40.0,
    bayes_prior: float = 80.0,
    ensemble_weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
    multi_model_weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
) -> np.ndarray:
    """Return relative number weights aligned with V3.6 backtest definitions.

    Weights are descriptive sampling weights, never calibrated draw probabilities.
    Long-term, Bayesian and time-decay strategies use all supplied prior history.
    """
    clean = validate_frame(df)
    if clean.empty or model in ("均匀随机基准", "历史结构约束模型"):
        return np.ones(45, dtype=float)
    rows = clean[NUMBER_COLS].to_numpy(dtype=np.int16, copy=True)
    recent_window = max(1, int(window))
    prior_rate = 6.0 / 45.0

    def smooth(sample: np.ndarray, prior_strength: float) -> np.ndarray:
        counts = np.bincount(sample.reshape(-1), minlength=46)[1:46].astype(float)
        return (counts + prior_strength * prior_rate) / max(len(sample) + prior_strength, 1.0)

    if model == "近期频率模型":
        return np.maximum(smooth(rows[-min(recent_window, len(rows)):], 8.0), 1e-8)
    if model == "长期频率模型":
        return np.maximum(smooth(rows, 40.0), 1e-8)
    if model == "贝叶斯平滑模型":
        return np.maximum(smooth(rows, float(bayes_prior)), 1e-8)
    if model == "指数时间衰减模型":
        ages = np.arange(len(rows) - 1, -1, -1, dtype=float)
        decay = np.exp(-ages / max(float(half_life), 1.0))
        weighted = np.zeros(45, dtype=float)
        for draw, weight in zip(rows, decay):
            weighted[draw - 1] += weight
        return np.maximum(weighted + 40.0 * prior_rate, 1e-8)
    if model == "综合评分模型":
        short = smooth(rows[-min(50, len(rows)):], 20.0)
        medium = smooth(rows[-min(200, len(rows)):], 40.0)
        long = smooth(rows, 80.0)
        a, b, c = ensemble_weights
        if not np.isclose(a + b + c, 1.0, atol=1e-8):
            raise ValueError("综合评分模型的三个权重之和必须等于1。")
        return np.maximum(a * short + b * medium + c * long, 1e-8)
    if model == "多模型集成模型":
        recent = smooth(rows[-min(recent_window, len(rows)):], 8.0)
        bayes = smooth(rows, float(bayes_prior))
        ages = np.arange(len(rows) - 1, -1, -1, dtype=float)
        decay = np.exp(-ages / max(float(half_life), 1.0))
        decayed = np.zeros(45, dtype=float)
        for draw, weight in zip(rows, decay):
            decayed[draw - 1] += weight
        decayed += 40.0 * prior_rate
        a, b, c = multi_model_weights
        if not np.isclose(a + b + c, 1.0, atol=1e-8):
            raise ValueError("多模型集成的三个权重之和必须等于1。")
        return np.maximum(a * recent + b * bayes + c * decayed, 1e-8)
    raise ValueError(f"未知选号模型：{model}")

def number_score_table(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    rows = []
    models = ["近期频率模型", "长期频率模型", "指数时间衰减模型", "贝叶斯平滑模型", "综合评分模型", "多模型集成模型"]
    weights = {m: model_weights(df, m, window) for m in models}
    for i in range(45):
        values = [float(weights[m][i]) for m in models]
        rows.append({"号码": i + 1, **{m: values[j] for j, m in enumerate(models)},
                     "综合相对得分": float(np.mean(values))})
    table = pd.DataFrame(rows)
    # Relative 0-100 scaling within the current sample; not a win probability.
    lo, hi = table["综合相对得分"].min(), table["综合相对得分"].max()
    table["综合相对得分"] = 50.0 if hi == lo else 100 * (table["综合相对得分"] - lo) / (hi - lo)
    return table.sort_values(["综合相对得分", "号码"], ascending=[False, True]).reset_index(drop=True)


def _weighted_ticket(rng: np.random.Generator, weights: np.ndarray) -> list[int]:
    probs = np.maximum(np.asarray(weights, dtype=float), 1e-12)
    probs /= probs.sum()
    nums = rng.choice(np.arange(1, 46), size=6, replace=False, p=probs)
    return sorted(int(x) for x in nums)


def _structure_ok(nums: list[int], hist_features: pd.DataFrame) -> bool:
    if hist_features.empty:
        return True
    total = sum(nums)
    odd = sum(n % 2 for n in nums)
    low = sum(n <= 15 for n in nums)
    mid = sum(16 <= n <= 30 for n in nums)
    high = sum(n >= 31 for n in nums)
    sum_lo, sum_hi = hist_features["和值"].quantile([0.02, 0.98]).tolist()
    # Broad quantiles are descriptive constraints, not probability improvements.
    return (sum_lo <= total <= sum_hi and 1 <= odd <= 5 and low >= 1 and mid >= 1 and high >= 1)


def generate_candidates(
    df: pd.DataFrame, count: int = 10, model: str = "综合评分模型", window: int = 100,
    max_overlap: int = 5, seed: int | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Generate distinct candidate tickets; any overlap relaxation is explicit and keeps the chosen model."""
    clean = validate_frame(df)
    if clean.empty:
        raise ValueError("没有历史数据。请手动导入历史 CSV 或 Excel 文件。")
    if model not in MODEL_NAMES:
        raise ValueError(f"未知选号模型：{model}")
    count = int(count)
    max_overlap = int(max_overlap)
    if not 1 <= count <= 20:
        raise ValueError("候选组数必须在 1—20 之间。")
    if not 0 <= max_overlap <= 5:
        raise ValueError("跨组最大重号数必须在 0—5 之间。")

    rng = np.random.default_rng(seed)
    history_features = draw_features(clean.tail(min(max(1, int(window)), len(clean))))
    weights = model_weights(clean, model, window)
    if model not in ("均匀随机基准", "历史结构约束模型"):
        raw_scores = np.asarray(weights, dtype=float)
        lo, hi = float(raw_scores.min()), float(raw_scores.max())
        scaled_scores = np.full(45, 50.0) if hi <= lo else 100.0 * (raw_scores - lo) / (hi - lo)
    else:
        scaled_scores = np.full(45, 50.0)

    def draw_one() -> list[int] | None:
        if model in ("均匀随机基准", "历史结构约束模型"):
            cand = sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False))
            if model == "历史结构约束模型" and not _structure_ok(cand, history_features):
                return None
            return cand
        return _weighted_ticket(rng, weights)

    tickets: list[list[int]] = []
    requested_overlap = max_overlap
    # With zero overlap, at most floor(45/6)=7 tickets can exist. Skip an
    # impossible first pass when the user requests more than seven.
    allowed_overlap = 1 if max_overlap == 0 and count > 7 else max_overlap
    relaxed = allowed_overlap != requested_overlap
    max_attempts = max(8000, count * 1000)

    while len(tickets) < count:
        for _ in range(max_attempts):
            cand = draw_one()
            if cand is None or cand in tickets:
                continue
            if tickets and max(len(set(cand) & set(old)) for old in tickets) > allowed_overlap:
                continue
            tickets.append(cand)
            if len(tickets) >= count:
                break
        if len(tickets) >= count:
            break
        if allowed_overlap >= 5:
            raise RuntimeError(
                f"在已允许每两组最多共享 5 个号码的情况下，仍无法生成 {count} 组互不重复组合。"
            )
        allowed_overlap += 1
        relaxed = True

    actual_max_overlap = max(
        (len(set(a) & set(b)) for i, a in enumerate(tickets) for b in tickets[i + 1:]),
        default=0,
    )
    rows = []
    for idx, nums in enumerate(tickets, start=1):
        relative_score = float(np.mean([scaled_scores[n - 1] for n in nums]))
        odd = sum(n % 2 for n in nums)
        rows.append({
            "组合": f"{idx:02d}", "模型": model,
            "号码": " · ".join(f"{n:02d}" for n in nums),
            "号码列表": nums, "和值": sum(nums), "奇数个数": odd,
            "低中高区": f"{sum(n <= 15 for n in nums)}-{sum(16 <= n <= 30 for n in nums)}-{sum(n >= 31 for n in nums)}",
            "模型相对得分": round(relative_score, 2),
            "请求跨组最大重号": requested_overlap,
            "实际允许最大重号": allowed_overlap,
            "实际跨组最大重号": actual_max_overlap,
            "跨组限制已放宽": relaxed,
        })
    return pd.DataFrame(rows), relaxed

def theoretical_random_metrics() -> dict[str, float]:
    total = math.comb(45, 6)
    p3plus = sum(math.comb(6, k) * math.comb(39, 6 - k) for k in range(3, 7)) / total
    return {"mean_hits_per_ticket": 0.8, "p3plus": p3plus, "jackpot_probability": 1 / total}


def run_backtest(*args, **kwargs):
    """Compatibility wrapper; the V3.6 backtest implementation lives in backtest_engine.py."""
    from backtest_engine import run_backtest as _run_backtest
    raw, summary, _metadata = _run_backtest(*args, **kwargs)
    return raw, summary


def compare_ticket_to_history(df: pd.DataFrame, numbers: list[int]) -> dict[str, Any]:
    clean = validate_frame(df)
    ticket = sorted(set(int(x) for x in numbers))
    if len(ticket) != 6 or any(n < 1 or n > 45 for n in ticket):
        raise ValueError("请提供 1—45 内互不重复的 6 个号码。")
    actual = clean[NUMBER_COLS].astype(int).values.tolist()
    hits = [len(set(ticket) & set(row)) for row in actual]
    counts = Counter(hits)
    return {"draws_checked": len(actual), "hit_distribution": {k: counts[k] for k in range(7)},
            "max_hits": max(hits, default=0), "mean_hits": float(np.mean(hits)) if hits else 0.0}
