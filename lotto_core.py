# -*- coding: utf-8 -*-
"""Core logic for Korea Lotto 6/45 research dashboard.

All models are descriptive/experimental. They do not predict a fair independent draw.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

COLUMNS = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]
NUMBER_COLS = [f"n{i}" for i in range(1, 7)]


def _pick(obj: dict[str, Any], *keys: str, default=None):
    for key in keys:
        if key in obj and obj[key] not in (None, ""):
            return obj[key]
    return default


def _normalise_date(value: Any) -> str:
    raw = str(value or "").strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw[:10]


def normalize_record(obj: dict[str, Any]) -> dict[str, Any] | None:
    """Accept legacy mirror fields and the 2026 Donghaeng result-page JSON fields."""
    try:
        draw_no = int(_pick(obj, "draw_no", "drwNo", "round", "회차", "ltEpsd"))
        nums = _pick(obj, "numbers", "winning_numbers", "winningNumbers")
        if nums is None:
            nums = []
            for i in range(1, 7):
                value = _pick(obj, f"drwtNo{i}", f"tm{i}WnNo", f"n{i}", f"num{i}", f"number{i}")
                nums.append(value)
        nums = sorted(int(x) for x in nums)
        bonus = int(_pick(obj, "bonus_no", "bnusNo", "bonus", "보너스", "bnsWnNo", "bonus_number"))
        draw_date = _pick(obj, "date", "drwNoDate", "draw_date", "추첨일", "ltRflYmd", default="")
        if len(nums) != 6 or len(set(nums)) != 6:
            return None
        if any(n < 1 or n > 45 for n in nums) or bonus < 1 or bonus > 45 or bonus in nums:
            return None
        if draw_no < 1:
            return None
        out = {"draw_no": draw_no, "date": _normalise_date(draw_date)}
        out.update({f"n{i}": nums[i - 1] for i in range(1, 7)})
        out["bonus"] = bonus
        return out
    except (TypeError, ValueError, KeyError):
        return None


def extract_records(payload: Any) -> list[dict[str, Any]]:
    """Extract records from legacy arrays or the 2026 {data: {list: [...]}} response."""
    raw = payload
    if isinstance(raw, dict):
        if any(k in raw for k in ("draw_no", "drwNo", "회차", "ltEpsd")):
            raw = [raw]
        elif isinstance(raw.get("data"), dict) and isinstance(raw["data"].get("list"), list):
            raw = raw["data"]["list"]
        else:
            for key in ("data", "results", "draws", "list"):
                value = raw.get(key)
                if isinstance(value, list):
                    raw = value
                    break
                if isinstance(value, dict) and isinstance(value.get("list"), list):
                    raw = value["list"]
                    break
            else:
                raw = list(raw.values())
    if not isinstance(raw, list):
        return []
    out: dict[int, dict[str, Any]] = {}
    for item in raw:
        if isinstance(item, dict):
            rec = normalize_record(item)
            if rec:
                out[rec["draw_no"]] = rec
    return [out[n] for n in sorted(out)]


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a tabular draw history; duplicate rounds keep the last row."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    tmp = frame.copy()
    aliases = {
        "회차": "draw_no", "round": "draw_no", "drwNo": "draw_no", "ltEpsd": "draw_no",
        "추첨일": "date", "draw_date": "date", "drwNoDate": "date", "ltRflYmd": "date",
        "보너스": "bonus", "bonus_no": "bonus", "bnusNo": "bonus", "bnsWnNo": "bonus",
    }
    tmp = tmp.rename(columns={c: aliases[c] for c in tmp.columns if c in aliases})
    for i in range(1, 7):
        aliases_i = {f"drwtNo{i}": f"n{i}", f"tm{i}WnNo": f"n{i}", f"num{i}": f"n{i}", f"number{i}": f"n{i}"}
        tmp = tmp.rename(columns={c: aliases_i[c] for c in tmp.columns if c in aliases_i})
    if "date" not in tmp.columns:
        tmp["date"] = ""
    missing = [c for c in ["draw_no", *NUMBER_COLS, "bonus"] if c not in tmp.columns]
    if missing:
        raise ValueError("数据缺少必要字段：" + ", ".join(missing))
    for col in ["draw_no", *NUMBER_COLS, "bonus"]:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp.dropna(subset=["draw_no", *NUMBER_COLS, "bonus"]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=COLUMNS)
    tmp["draw_no"] = tmp["draw_no"].astype(int)
    for col in [*NUMBER_COLS, "bonus"]:
        tmp[col] = tmp[col].astype(int)
    valid_rows = []
    for _, row in tmp.iterrows():
        nums = sorted(int(row[c]) for c in NUMBER_COLS)
        bonus = int(row["bonus"])
        if int(row["draw_no"]) < 1:
            continue
        if len(set(nums)) != 6 or any(n < 1 or n > 45 for n in nums):
            continue
        if bonus < 1 or bonus > 45 or bonus in nums:
            continue
        rec = {"draw_no": int(row["draw_no"]), "date": _normalise_date(row.get("date", ""))}
        rec.update({f"n{i}": nums[i - 1] for i in range(1, 7)})
        rec["bonus"] = bonus
        valid_rows.append(rec)
    if not valid_rows:
        return pd.DataFrame(columns=COLUMNS)
    return (pd.DataFrame(valid_rows, columns=COLUMNS)
            .drop_duplicates("draw_no", keep="last")
            .sort_values("draw_no").reset_index(drop=True))


def merge_histories(*frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frame in frames:
        if frame is not None and not frame.empty:
            rows.append(validate_frame(frame))
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    merged = pd.concat(rows, ignore_index=True)
    return validate_frame(merged)


def quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    df = validate_frame(frame)
    if df.empty:
        return {"rows": 0, "first_draw": None, "last_draw": None, "missing_draws": [], "date_errors": 0}
    present = set(df["draw_no"].astype(int))
    missing = [n for n in range(int(df["draw_no"].min()), int(df["draw_no"].max()) + 1) if n not in present]
    date_errors = 0
    for value in df["date"].astype(str):
        if value and (len(value) != 10 or value[4:5] != "-" or value[7:8] != "-"):
            date_errors += 1
    return {
        "rows": len(df), "first_draw": int(df["draw_no"].min()),
        "last_draw": int(df["draw_no"].max()), "missing_draws": missing,
        "date_errors": date_errors,
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
    for n in range(1, 46):
        observed = counts[n]
        z = (observed - expected) / sd if sd else 0.0
        # Two-sided normal approximation: descriptive only, no multiplicity correction.
        p_approx = math.erfc(abs(z) / math.sqrt(2)) if sd else 1.0
        rows.append({"号码": n, "出现次数": observed, "理论期望": round(expected, 2),
                     "标准化偏差Z": round(z, 3), "近似P值": round(p_approx, 4)})
    return pd.DataFrame(rows).sort_values(["出现次数", "号码"], ascending=[False, True]).reset_index(drop=True)


def gap_table(df: pd.DataFrame) -> pd.DataFrame:
    clean = validate_frame(df)
    if clean.empty:
        return pd.DataFrame(columns=["号码", "距最近出现期数"])
    latest = int(clean["draw_no"].max())
    last_seen: dict[int, int | None] = {n: None for n in range(1, 46)}
    for _, row in clean.iterrows():
        for n in [int(row[f"n{i}"]) for i in range(1, 7)]:
            last_seen[n] = int(row["draw_no"])
    rows = []
    for n in range(1, 46):
        last = last_seen[n]
        rows.append({"号码": n, "距最近出现期数": latest - last if last is not None else len(clean)})
    return pd.DataFrame(rows).sort_values(["距最近出现期数", "号码"], ascending=[False, True]).reset_index(drop=True)


def pair_table(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    clean = validate_frame(df)
    counts: Counter[tuple[int, int]] = Counter()
    for nums in get_numbers(clean):
        counts.update(combinations(nums, 2))
    draws = len(clean)
    p_pair = (6 / 45) * (5 / 44)  # 1/66 for a specific pair in one draw
    expected = draws * p_pair
    sd = math.sqrt(draws * p_pair * (1 - p_pair)) if draws else 0.0
    rows = []
    for (a, b), count in counts.most_common(top_n):
        z = (count - expected) / sd if sd else 0.0
        rows.append({"号码组合": f"{a:02d} + {b:02d}", "共同出现次数": count,
                     "理论期望": round(expected, 3), "标准化偏差Z": round(z, 3)})
    return pd.DataFrame(rows, columns=["号码组合", "共同出现次数", "理论期望", "标准化偏差Z"])


def draw_features(df: pd.DataFrame) -> pd.DataFrame:
    clean = validate_frame(df)
    rows = []
    previous: set[int] = set()
    for _, row in clean.iterrows():
        nums = [int(row[f"n{i}"]) for i in range(1, 7)]
        current = set(nums)
        rows.append({
            "期数": int(row["draw_no"]), "日期": row["date"], "和值": sum(nums),
            "奇数个数": sum(n % 2 for n in nums), "高区个数(23-45)": sum(n >= 23 for n in nums),
            "连号对数": sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1),
            "与上期重号数": len(current & previous),
            "低区(1-15)": sum(n <= 15 for n in nums),
            "中区(16-30)": sum(16 <= n <= 30 for n in nums),
            "高区(31-45)": sum(n >= 31 for n in nums),
        })
        previous = current
    return pd.DataFrame(rows)


def _weighted_sample(rng: random.Random, weights: np.ndarray) -> list[int]:
    pool = list(range(1, 46))
    w = np.maximum(np.asarray(weights, dtype=float).copy(), 1e-9)
    picked: list[int] = []
    for _ in range(6):
        probs = w / w.sum()
        idx = rng.choices(range(len(pool)), weights=probs.tolist(), k=1)[0]
        picked.append(pool.pop(idx))
        w = np.delete(w, idx)
    return sorted(picked)


def model_weights(df: pd.DataFrame, model: str, window: int = 100, half_life: float = 35.0) -> np.ndarray:
    clean = validate_frame(df)
    recent = clean.tail(min(window, len(clean)))
    nums = get_numbers(recent)
    freq = Counter(n for row in nums for n in row)
    if model == "近期频率倾向（实验）":
        return np.array([1.0 + freq[n] for n in range(1, 46)], dtype=float)
    if model == "指数时间衰减（实验）":
        weights = np.ones(45, dtype=float)
        for age, row in enumerate(reversed(nums)):
            decay = math.exp(-age / max(half_life, 1.0))
            for n in row:
                weights[n - 1] += decay
        return weights
    if model == "遗漏倾向（实验）":
        gaps = gap_table(clean).set_index("号码")["距最近出现期数"].to_dict()
        return np.array([1.0 + math.log1p(float(gaps.get(n, 0))) for n in range(1, 46)], dtype=float)
    return np.ones(45, dtype=float)


def _is_structurally_filtered(nums: list[int], sum_low: float, sum_high: float) -> bool:
    odd = sum(n % 2 for n in nums)
    runs = sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1)
    bins = {0 if n <= 15 else 1 if n <= 30 else 2 for n in nums}
    return sum_low <= sum(nums) <= sum_high and 2 <= odd <= 4 and runs < 3 and len(bins) >= 3


def generate_candidates(
    df: pd.DataFrame, count: int = 5, model: str = "均匀随机基准", window: int = 100,
    max_overlap: int = 3, seed: int | None = None,
) -> tuple[pd.DataFrame, bool]:
    clean = validate_frame(df)
    if clean.empty:
        raise ValueError("没有可用历史数据")
    rng = random.Random(seed)
    recent = clean.tail(min(window, len(clean)))
    weights = model_weights(clean, model, window)
    hist_freq = Counter(n for row in get_numbers(clean) for n in row)
    expected = len(clean) * 6 / 45
    feats = draw_features(recent)
    low, high = (float(feats["和值"].quantile(0.05)), float(feats["和值"].quantile(0.95))) if not feats.empty else (70.0, 200.0)
    tickets: list[list[int]] = []
    attempts = 0
    hard_limit = max(1000, count * 1500)
    while len(tickets) < count and attempts < hard_limit:
        attempts += 1
        if model == "结构约束随机（实验）":
            candidate = sorted(rng.sample(range(1, 46), 6))
            if not _is_structurally_filtered(candidate, low, high):
                continue
        elif model == "均匀随机基准":
            candidate = sorted(rng.sample(range(1, 46), 6))
        else:
            candidate = _weighted_sample(rng, weights)
        if candidate in tickets:
            continue
        if all(len(set(candidate) & set(old)) <= max_overlap for old in tickets):
            tickets.append(candidate)
    relaxed = False
    while len(tickets) < count:
        relaxed = True
        candidate = sorted(rng.sample(range(1, 46), 6))
        if candidate not in tickets:
            tickets.append(candidate)
    rows = []
    for idx, nums in enumerate(tickets, 1):
        max_shared = max((len(set(nums) & set(old)) for j, old in enumerate(tickets) if j != idx - 1), default=0)
        rows.append({
            "组别": f"组合 {idx}", "生成策略": model, "号码": nums,
            "号码展示": "  ·  ".join(f"{n:02d}" for n in nums), "和值": sum(nums),
            "奇偶比": f"{sum(n % 2 for n in nums)}:{6 - sum(n % 2 for n in nums)}",
            "历史频率指数": round(sum(hist_freq[n] / expected for n in nums) / 6, 2) if expected else 0.0,
            "与其他组合最大重号": max_shared,
        })
    return pd.DataFrame(rows), relaxed


def _theoretical_random_metrics() -> dict[str, float]:
    total = math.comb(45, 6)
    p3plus = sum(math.comb(6, k) * math.comb(39, 6 - k) for k in range(3, 7)) / total
    return {"mean_hits": 0.8, "p3plus": p3plus}


def run_backtest(
    df: pd.DataFrame, lookback: int = 100, test_draws: int = 200,
    tickets_per_draw: int = 5, n_seeds: int = 5, seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired walk-forward experiment; each test draw uses only earlier draws."""
    clean = validate_frame(df)
    if len(clean) <= lookback + 5:
        return pd.DataFrame(), pd.DataFrame()
    nums_by_draw = get_numbers(clean)
    start = max(lookback, len(clean) - min(test_draws, len(clean) - lookback))
    strategies = ["均匀随机基准", "近期频率倾向（实验）", "指数时间衰减（实验）"]
    records: list[dict[str, Any]] = []
    for i in range(start, len(clean)):
        train = clean.iloc[max(0, i - lookback):i].copy()
        actual = set(nums_by_draw[i])
        draw_no = int(clean.iloc[i]["draw_no"])
        for seed_idx in range(n_seeds):
            for strategy_idx, strategy in enumerate(strategies):
                local_seed = seed + i * 100_003 + seed_idx * 10_007 + strategy_idx * 101
                rng = random.Random(local_seed)
                weights = model_weights(train, strategy, lookback)
                seen: set[tuple[int, ...]] = set()
                for ticket_no in range(tickets_per_draw):
                    for _ in range(100):
                        ticket = (sorted(rng.sample(range(1, 46), 6)) if strategy == "均匀随机基准"
                                  else _weighted_sample(rng, weights))
                        key = tuple(ticket)
                        if key not in seen:
                            seen.add(key)
                            break
                    hits = len(set(ticket) & actual)
                    records.append({"期数": draw_no, "随机种子组": seed_idx + 1, "策略": strategy,
                                    "注序": ticket_no + 1, "命中数": hits,
                                    "命中3个以上": int(hits >= 3), "命中4个以上": int(hits >= 4),
                                    "命中5个以上": int(hits >= 5), "命中6个": int(hits == 6)})
    raw = pd.DataFrame(records)
    if raw.empty:
        return raw, pd.DataFrame()
    by_draw = raw.groupby(["期数", "策略"], as_index=False).agg(
        每期平均命中=("命中数", "mean"), 每期3个以上率=("命中3个以上", "mean")
    )
    baseline = by_draw[by_draw["策略"] == "均匀随机基准"].set_index("期数")
    rng_boot = np.random.default_rng(seed + 77)
    summary_rows = []
    for strategy in strategies:
        part = raw[raw["策略"] == strategy]
        per = by_draw[by_draw["策略"] == strategy].set_index("期数")
        aligned = per.join(baseline[["每期平均命中", "每期3个以上率"]], rsuffix="_随机", how="inner")
        diff_hits = (aligned["每期平均命中"] - aligned["每期平均命中_随机"]).to_numpy()
        diff_3 = (aligned["每期3个以上率"] - aligned["每期3个以上率_随机"]).to_numpy()
        def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
            if len(values) < 2:
                return (float("nan"), float("nan"))
            samples = rng_boot.choice(values, size=(1500, len(values)), replace=True).mean(axis=1)
            return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
        ci_lo, ci_hi = bootstrap_ci(diff_hits) if strategy != "均匀随机基准" else (0.0, 0.0)
        ci3_lo, ci3_hi = bootstrap_ci(diff_3) if strategy != "均匀随机基准" else (0.0, 0.0)
        summary_rows.append({
            "策略": strategy, "测试注数": len(part), "平均命中数": part["命中数"].mean(),
            "3个以上命中率": part["命中3个以上"].mean(), "4个以上命中率": part["命中4个以上"].mean(),
            "5个以上命中率": part["命中5个以上"].mean(), "6个命中率": part["命中6个"].mean(),
            "平均命中差值_vs随机": 0.0 if strategy == "均匀随机基准" else float(diff_hits.mean()),
            "平均命中差值95%CI下限": ci_lo, "平均命中差值95%CI上限": ci_hi,
            "3+率差值_vs随机": 0.0 if strategy == "均匀随机基准" else float(diff_3.mean()),
            "3+率差值95%CI下限": ci3_lo, "3+率差值95%CI上限": ci3_hi,
        })
    summary = pd.DataFrame(summary_rows)
    return raw, summary


def insights(df: pd.DataFrame, recent_n: int = 100) -> list[str]:
    clean = validate_frame(df)
    recent = clean.tail(min(recent_n, len(clean)))
    if clean.empty:
        return ["暂无有效开奖数据。"]
    freq = frequency_table(recent)
    gaps = gap_table(clean)
    feat = draw_features(recent)
    hot = "、".join(f"{int(x):02d}" for x in freq.head(5)["号码"].tolist())
    cold = "、".join(f"{int(x):02d}" for x in gaps.head(5)["号码"].tolist())
    return [
        f"最近 {len(recent)} 期出现次数较多的号码：{hot}。这是样本描述，不代表下一期更容易出现。",
        f"截至第 {int(clean['draw_no'].max())} 期，当前遗漏较长的号码：{cold}。遗漏不会累积成下一期优势。",
        f"近期和值中位数为 {feat['和值'].median():.0f}，只描述历史分布；公平开奖下每个具体六数组合概率相同。",
    ]
