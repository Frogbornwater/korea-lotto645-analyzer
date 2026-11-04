# -*- coding: utf-8 -*-
"""Lotto 6/45 V3 analysis engine.

Historical data describes past draws. No method here can reliably predict a fair,
independent future draw or improve the probability of any exact six-number ticket.
"""
from __future__ import annotations

import math
import random
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
]


def _first(obj: dict[str, Any], keys: Iterable[str], default=None):
    for key in keys:
        value = obj.get(key)
        if value is not None and value != "":
            return value
    return default


def normalize_date(value: Any) -> str:
    raw = str(value or "").strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw[:10]


def normalize_record(obj: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize official result-page fields and common CSV aliases."""
    try:
        draw_no = int(_first(obj, ["draw_no", "drwNo", "round", "회차", "ltEpsd"]))
        raw_numbers = _first(obj, ["numbers", "winning_numbers", "winningNumbers"])
        if raw_numbers is None:
            raw_numbers = [
                _first(obj, [f"drwtNo{i}", f"tm{i}WnNo", f"n{i}", f"num{i}", f"number{i}"])
                for i in range(1, 7)
            ]
        nums = sorted(int(x) for x in raw_numbers)
        bonus = int(_first(obj, ["bonus", "bonus_no", "bonus_number", "bnusNo", "bnsWnNo", "보너스"]))
        date = normalize_date(_first(obj, ["date", "draw_date", "drwNoDate", "ltRflYmd", "추첨일"], ""))
        if draw_no < 1 or len(nums) != 6 or len(set(nums)) != 6:
            return None
        if any(n < 1 or n > 45 for n in nums) or bonus < 1 or bonus > 45 or bonus in nums:
            return None
        out = {"draw_no": draw_no, "date": date}
        out.update({f"n{i}": nums[i - 1] for i in range(1, 7)})
        out["bonus"] = bonus
        return out
    except (TypeError, ValueError, KeyError):
        return None


def extract_records(payload: Any) -> list[dict[str, Any]]:
    """Extract draw items from the official 2026 result JSON response shapes."""
    raw: Any = payload
    if isinstance(raw, dict):
        # The official result page's specific-round endpoint.
        data = raw.get("data")
        if isinstance(data, dict) and isinstance(data.get("list"), list):
            raw = data["list"]
        else:
            # The official main-info response nests recent Lotto rounds here.
            try:
                nested = data["result"]["pstLtEpstInfo"]["lt645"]
                if isinstance(nested, list):
                    raw = nested
                else:
                    raw = []
            except (TypeError, KeyError):
                if any(k in raw for k in ("draw_no", "drwNo", "ltEpsd")):
                    raw = [raw]
                else:
                    for key in ("results", "draws", "list"):
                        if isinstance(raw.get(key), list):
                            raw = raw[key]
                            break
                    else:
                        raw = []
    if not isinstance(raw, list):
        return []
    by_round: dict[int, dict[str, Any]] = {}
    for item in raw:
        if isinstance(item, dict):
            rec = normalize_record(item)
            if rec:
                by_round[rec["draw_no"]] = rec
    return [by_round[k] for k in sorted(by_round)]


def validate_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
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
        for alias in (f"drwtNo{i}", f"tm{i}WnNo", f"num{i}", f"number{i}"):
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
    tmp = tmp.dropna(subset=required)
    rows = []
    for _, row in tmp.iterrows():
        try:
            draw_no = int(row["draw_no"])
            nums = sorted(int(row[c]) for c in NUMBER_COLS)
            bonus = int(row["bonus"])
        except (TypeError, ValueError):
            continue
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
    return (pd.DataFrame(rows, columns=COLUMNS)
            .drop_duplicates("draw_no", keep="last")
            .sort_values("draw_no").reset_index(drop=True))


def quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    df = validate_frame(frame)
    if df.empty:
        return {"rows": 0, "first_draw": None, "last_draw": None, "missing_draws": [], "date_errors": 0,
                "duplicates": 0, "sequence_complete": False}
    present = set(df["draw_no"].astype(int))
    first, last = int(df["draw_no"].min()), int(df["draw_no"].max())
    missing = [n for n in range(first, last + 1) if n not in present]
    date_errors = sum(1 for d in df["date"].astype(str) if d and (len(d) != 10 or d[4:5] != "-" or d[7:8] != "-"))
    return {"rows": len(df), "first_draw": first, "last_draw": last, "missing_draws": missing,
            "date_errors": date_errors, "duplicates": int(frame["draw_no"].duplicated().sum()) if "draw_no" in frame else 0,
            "sequence_complete": first == 1 and not missing}


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


def model_weights(df: pd.DataFrame, model: str, window: int = 100, half_life: float = 40.0) -> np.ndarray:
    """Return relative number weights, not calibrated winning probabilities."""
    clean = validate_frame(df)
    if clean.empty:
        return np.ones(45, dtype=float)
    counts, n_draws = _counts_for_window(clean, window)
    long_counts, long_n = _counts_for_window(clean, None)
    prior_rate = 6 / 45
    # 40 pseudo-draws shrink observed rates toward the theoretical 6/45 baseline.
    prior_draws = 40.0
    posterior = (counts + prior_draws * prior_rate) / max(n_draws + prior_draws, 1.0)
    if model == "近期频率模型":
        # A lighter prior lets the short-window model react more to recent counts.
        recent_prior = 8.0
        recent_rate = (counts + recent_prior * prior_rate) / max(n_draws + recent_prior, 1.0)
        return np.maximum(recent_rate, 1e-8)
    if model == "长期频率模型":
        return np.maximum((long_counts + prior_draws * prior_rate) / max(long_n + prior_draws, 1.0), 1e-8)
    if model == "指数时间衰减模型":
        decayed = np.zeros(45, dtype=float)
        rows = get_numbers(clean.tail(min(window, len(clean))))
        for age, nums in enumerate(reversed(rows)):
            weight = math.exp(-age / max(float(half_life), 1.0))
            for number in nums:
                decayed[number - 1] += weight
        pseudo = max(prior_draws * prior_rate, 1.0)
        return np.maximum(decayed + pseudo, 1e-8)
    if model == "贝叶斯平滑模型":
        # Stronger prior shrinkage avoids treating noisy short-window leaders as signal.
        bayes_prior = 80.0
        bayes_rate = (counts + bayes_prior * prior_rate) / max(n_draws + bayes_prior, 1.0)
        return np.maximum(bayes_rate, 1e-8)
    if model == "综合评分模型":
        short_counts, short_n = _counts_for_window(clean, min(50, len(clean)))
        mid_counts, mid_n = _counts_for_window(clean, min(200, len(clean)))
        short = (short_counts + 20 * prior_rate) / max(short_n + 20, 1)
        mid = (mid_counts + 40 * prior_rate) / max(mid_n + 40, 1)
        long = (long_counts + 80 * prior_rate) / max(long_n + 80, 1)
        return np.maximum(0.40 * short + 0.35 * mid + 0.25 * long, 1e-8)
    return np.ones(45, dtype=float)


def number_score_table(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    rows = []
    models = ["近期频率模型", "长期频率模型", "指数时间衰减模型", "贝叶斯平滑模型", "综合评分模型"]
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
    clean = validate_frame(df)
    if clean.empty:
        raise ValueError("没有历史数据。请先从 GitHub Actions 同步官方历史开奖数据。")
    rng = np.random.default_rng(seed)
    history_features = draw_features(clean.tail(min(window, len(clean))))
    weights = model_weights(clean, model, window)
    score = number_score_table(clean, window).set_index("号码")["综合相对得分"].to_dict()
    tickets: list[list[int]] = []
    relaxed = False
    max_attempts = max(5000, count * 1200)
    for _ in range(max_attempts):
        if len(tickets) >= count:
            break
        if model == "均匀随机基准" or model == "历史结构约束模型":
            cand = sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False))
            if model == "历史结构约束模型" and not _structure_ok(cand, history_features):
                continue
        else:
            cand = _weighted_ticket(rng, weights)
        if cand in tickets:
            continue
        if tickets and max(len(set(cand) & set(old)) for old in tickets) > max_overlap:
            continue
        tickets.append(cand)
    while len(tickets) < count:
        relaxed = True
        cand = sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False))
        if cand not in tickets:
            tickets.append(cand)
    rows = []
    for idx, nums in enumerate(tickets, start=1):
        relative_score = float(np.mean([score[n] for n in nums]))
        odd = sum(n % 2 for n in nums)
        rows.append({
            "组合": f"{idx:02d}", "模型": model,
            "号码": " · ".join(f"{n:02d}" for n in nums),
            "号码列表": nums, "和值": sum(nums), "奇数个数": odd,
            "低中高区": f"{sum(n <= 15 for n in nums)}-{sum(16 <= n <= 30 for n in nums)}-{sum(n >= 31 for n in nums)}",
            "模型相对得分": round(relative_score, 2),
            "与其他候选最大重号": max((len(set(nums) & set(other)) for j, other in enumerate(tickets) if j != idx - 1), default=0),
        })
    return pd.DataFrame(rows), relaxed


def theoretical_random_metrics() -> dict[str, float]:
    total = math.comb(45, 6)
    p3plus = sum(math.comb(6, k) * math.comb(39, 6 - k) for k in range(3, 7)) / total
    return {"mean_hits_per_ticket": 0.8, "p3plus": p3plus, "jackpot_probability": 1 / total}


def run_backtest(
    df: pd.DataFrame, lookback: int = 200, test_draws: int = 100,
    tickets_per_draw: int = 5, n_seeds: int = 3, seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward backtest. Every test prediction uses earlier draws only."""
    clean = validate_frame(df)
    if len(clean) <= lookback + 5:
        return pd.DataFrame(), pd.DataFrame()
    all_nums = get_numbers(clean)
    start = max(lookback, len(clean) - min(test_draws, len(clean) - lookback))
    strategies = ["均匀随机基准", "近期频率模型", "长期频率模型", "指数时间衰减模型", "贝叶斯平滑模型", "综合评分模型"]
    rows: list[dict[str, Any]] = []
    for i in range(start, len(clean)):
        train = clean.iloc[max(0, i - lookback):i].copy()
        actual = set(all_nums[i])
        draw_no = int(clean.iloc[i]["draw_no"])
        for seed_idx in range(n_seeds):
            for strategy_idx, strategy in enumerate(strategies):
                local_seed = seed + i * 100_003 + seed_idx * 10_007 + strategy_idx * 101
                tickets, _ = generate_candidates(
                    train, count=tickets_per_draw, model=strategy,
                    window=min(lookback, 500), max_overlap=5, seed=local_seed,
                )
                for ticket_idx, ticket_row in tickets.iterrows():
                    ticket = ticket_row["号码列表"]
                    hits = len(set(ticket) & actual)
                    rows.append({
                        "期数": draw_no, "随机种子组": seed_idx + 1, "策略": strategy,
                        "注序": int(ticket_idx) + 1, "命中数": hits,
                        "命中3个以上": int(hits >= 3), "命中4个以上": int(hits >= 4),
                        "命中5个以上": int(hits >= 5), "命中6个": int(hits == 6),
                    })
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, pd.DataFrame()
    per_draw = raw.groupby(["期数", "策略"], as_index=False).agg(
        每期平均命中=("命中数", "mean"), 每期3个以上率=("命中3个以上", "mean"))
    baseline = per_draw[per_draw["策略"] == "均匀随机基准"].set_index("期数")
    rng = np.random.default_rng(seed + 909)
    out = []
    for strategy in strategies:
        subset = raw[raw["策略"] == strategy]
        current = per_draw[per_draw["策略"] == strategy].set_index("期数")
        aligned = current.join(baseline[["每期平均命中", "每期3个以上率"]], rsuffix="_随机", how="inner")
        hit_delta = (aligned["每期平均命中"] - aligned["每期平均命中_随机"]).to_numpy()
        p3_delta = (aligned["每期3个以上率"] - aligned["每期3个以上率_随机"]).to_numpy()
        def ci(values: np.ndarray) -> tuple[float, float]:
            if len(values) < 2:
                return (float("nan"), float("nan"))
            sample = rng.choice(values, size=(1000, len(values)), replace=True).mean(axis=1)
            return float(np.quantile(sample, .025)), float(np.quantile(sample, .975))
        lo, hi = (0.0, 0.0) if strategy == "均匀随机基准" else ci(hit_delta)
        p3lo, p3hi = (0.0, 0.0) if strategy == "均匀随机基准" else ci(p3_delta)
        out.append({
            "策略": strategy, "测试注数": int(len(subset)), "平均命中数": float(subset["命中数"].mean()),
            "3个以上命中率": float(subset["命中3个以上"].mean()),
            "4个以上命中率": float(subset["命中4个以上"].mean()),
            "5个以上命中率": float(subset["命中5个以上"].mean()),
            "6个命中率": float(subset["命中6个"].mean()),
            "平均命中差_vs随机": 0.0 if strategy == "均匀随机基准" else float(hit_delta.mean()),
            "平均命中差95%CI下限": lo, "平均命中差95%CI上限": hi,
            "3+率差_vs随机": 0.0 if strategy == "均匀随机基准" else float(p3_delta.mean()),
            "3+率差95%CI下限": p3lo, "3+率差95%CI上限": p3hi,
        })
    return raw, pd.DataFrame(out)


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
