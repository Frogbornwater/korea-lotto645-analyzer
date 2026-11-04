# -*- coding: utf-8 -*-
"""Korean Lotto 6/45 analyzer. Descriptive statistics, backtesting, and entertainment-only picks."""
from __future__ import annotations

import io
import json
import math
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CACHE_FILE = DATA_DIR / "lotto645.csv"
MIRROR_ALL_URL = "https://smok95.github.io/lotto/results/all.json"
MIRROR_LATEST_URL = "https://smok95.github.io/lotto/results/latest.json"
OFFICIAL_RESULT_URL = "https://www.dhlottery.co.kr/lt645/result"

st.set_page_config(page_title="韩国 Lotto 6/45 分析台", page_icon="🎯", layout="wide")


# ----------------------------- Data layer ----------------------------------
def _pick(d: dict[str, Any], *keys: str, default=None):
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return default


def _normalize_record(obj: dict[str, Any]) -> dict[str, Any] | None:
    try:
        draw_no = int(_pick(obj, "draw_no", "drwNo", "round", "회차"))
        nums = _pick(obj, "numbers")
        if nums is None:
            nums = [_pick(obj, f"drwtNo{i}") for i in range(1, 7)]
        nums = sorted(int(x) for x in nums)
        bonus = int(_pick(obj, "bonus_no", "bnusNo", "bonus", "보너스"))
        draw_date = _pick(obj, "date", "drwNoDate", "draw_date", "추첨일", default="")
        if len(nums) != 6 or len(set(nums)) != 6:
            return None
        if any(n < 1 or n > 45 for n in nums) or bonus < 1 or bonus > 45 or bonus in nums:
            return None
        return {
            "draw_no": draw_no,
            "date": str(draw_date)[:10],
            "n1": nums[0], "n2": nums[1], "n3": nums[2],
            "n4": nums[3], "n5": nums[4], "n6": nums[5],
            "bonus": bonus,
        }
    except (TypeError, ValueError, KeyError):
        return None


def _extract_records(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        if any(k in raw for k in ("draw_no", "drwNo", "회차")):
            raw = [raw]
        else:
            for key in ("data", "results", "draws", "list"):
                if isinstance(raw.get(key), list):
                    raw = raw[key]
                    break
            else:
                raw = list(raw.values())
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, dict):
            rec = _normalize_record(item)
            if rec:
                out.append(rec)
    return sorted({r["draw_no"]: r for r in out}.values(), key=lambda x: x["draw_no"])


def read_cache() -> pd.DataFrame:
    if not CACHE_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(CACHE_FILE, encoding="utf-8-sig")
        required = ["draw_no", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]
        if not set(required).issubset(df.columns):
            return pd.DataFrame()
        return validate_frame(df)
    except Exception:
        return pd.DataFrame()


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]
    tmp = df.copy()
    if "date" not in tmp.columns:
        tmp["date"] = ""
    for c in ["draw_no", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp = tmp.dropna(subset=[c for c in cols if c != "date"])
    tmp = tmp[cols]
    tmp = tmp.astype({c: int for c in cols if c != "date"})
    valid_rows = []
    for _, row in tmp.iterrows():
        ns = [int(row[f"n{i}"]) for i in range(1, 7)]
        b = int(row["bonus"])
        if len(set(ns)) == 6 and all(1 <= n <= 45 for n in ns) and 1 <= b <= 45 and b not in ns:
            rec = row.to_dict()
            ns = sorted(ns)
            for i, n in enumerate(ns, 1):
                rec[f"n{i}"] = n
            valid_rows.append(rec)
    if not valid_rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(valid_rows, columns=cols).drop_duplicates("draw_no").sort_values("draw_no").reset_index(drop=True)


def fetch_mirror_data() -> tuple[pd.DataFrame, str]:
    headers = {"User-Agent": "Mozilla/5.0 Lotto645Analyzer/1.0"}
    errors = []
    for url, label in ((MIRROR_ALL_URL, "公开历史数据镜像"), (MIRROR_LATEST_URL, "公开最新期数据镜像")):
        try:
            resp = requests.get(url, timeout=18, headers=headers)
            resp.raise_for_status()
            records = _extract_records(resp.json())
            if records:
                df = pd.DataFrame(records)
                return validate_frame(df), label
            errors.append(f"{label}返回的数据结构无法识别")
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    raise RuntimeError("；".join(errors))


def save_cache(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")


def parse_uploaded_csv(file) -> pd.DataFrame:
    raw = pd.read_csv(file)
    rename = {}
    aliases = {
        "회차": "draw_no", "round": "draw_no", "drwNo": "draw_no",
        "추첨일": "date", "draw_date": "date", "drwNoDate": "date",
        "보너스": "bonus", "bonus_no": "bonus", "bnusNo": "bonus",
    }
    for c in raw.columns:
        if c in aliases:
            rename[c] = aliases[c]
    raw = raw.rename(columns=rename)
    # Support a wide range of common CSV column names.
    number_aliases = {}
    for i in range(1, 7):
        candidates = [f"n{i}", f"num{i}", f"number{i}", f"drwtNo{i}", f"당첨번호{i}"]
        for c in candidates:
            if c in raw.columns:
                number_aliases[c] = f"n{i}"
                break
    raw = raw.rename(columns=number_aliases)
    needed = ["draw_no", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError("CSV 缺少列：" + ", ".join(missing) + "。需要 draw_no,n1,n2,n3,n4,n5,n6,bonus。")
    return validate_frame(raw)


# ----------------------------- Analytics -----------------------------------
def get_numbers(df: pd.DataFrame) -> list[list[int]]:
    return df[[f"n{i}" for i in range(1, 7)]].astype(int).values.tolist()


def frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = Counter(n for row in get_numbers(df) for n in row)
    total = len(df)
    return pd.DataFrame({
        "号码": list(range(1, 46)),
        "出现次数": [counts[n] for n in range(1, 46)],
        "理论均值": [total * 6 / 45] * 45,
    }).assign(偏离均值=lambda x: x["出现次数"] - x["理论均值"]).sort_values(
        ["出现次数", "号码"], ascending=[False, True]
    )


def gap_table(df: pd.DataFrame) -> pd.DataFrame:
    draws = get_numbers(df)
    latest = int(df["draw_no"].max())
    last_seen = {n: None for n in range(1, 46)}
    for _, row in df.iterrows():
        for n in [int(row[f"n{i}"]) for i in range(1, 7)]:
            last_seen[n] = int(row["draw_no"])
    rows = []
    for n in range(1, 46):
        last = last_seen[n]
        rows.append({"号码": n, "距最近出现期数": (latest - last) if last is not None else len(df)})
    return pd.DataFrame(rows).sort_values(["距最近出现期数", "号码"], ascending=[False, True])


def pair_table(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    counts = Counter()
    for nums in get_numbers(df):
        counts.update(combinations(sorted(nums), 2))
    rows = [{"号码组合": f"{a:02d} + {b:02d}", "共同出现次数": c} for (a, b), c in counts.most_common(top_n)]
    return pd.DataFrame(rows, columns=["号码组合", "共同出现次数"])


def draw_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    previous_numbers: set[int] = set()
    for _, r in df.iterrows():
        nums = [int(r[f"n{i}"]) for i in range(1, 7)]
        current_numbers = set(nums)
        rows.append({
            "期数": int(r["draw_no"]),
            "和值": sum(nums),
            "奇数个数": sum(n % 2 for n in nums),
            "高区个数(23-45)": sum(n >= 23 for n in nums),
            "连号对数": sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1),
            "与上期重号数": len(current_numbers & previous_numbers),
        })
        previous_numbers = current_numbers
    return pd.DataFrame(rows)


def weighted_pick(rng: random.Random, weights: np.ndarray) -> list[int]:
    pool = list(range(1, 46))
    w = np.asarray(weights, dtype=float).copy()
    picked = []
    for _ in range(6):
        total = float(w.sum())
        probs = w / total if total > 0 else np.ones(len(pool)) / len(pool)
        idx = rng.choices(range(len(pool)), weights=probs.tolist(), k=1)[0]
        picked.append(pool.pop(idx))
        w = np.delete(w, idx)
    return sorted(picked)


def make_candidates(df: pd.DataFrame, count: int, strategy: str, recent_n: int, max_overlap: int, seed: int | None = None):
    rng = random.Random(seed)
    recent = df.tail(min(recent_n, len(df)))
    freqs = Counter(n for row in get_numbers(recent) for n in row)
    # Laplace smoothing prevents a never-seen number from receiving zero weight.
    weights = np.array([1.0 + freqs[n] for n in range(1, 46)], dtype=float)
    tickets: list[list[int]] = []
    attempts = 0
    while len(tickets) < count and attempts < count * 500:
        attempts += 1
        if strategy == "历史频率实验（仅作对照）":
            candidate = weighted_pick(rng, weights)
        else:
            candidate = sorted(rng.sample(range(1, 46), 6))
        if candidate in tickets:
            continue
        if all(len(set(candidate) & set(old)) <= max_overlap for old in tickets):
            tickets.append(candidate)
    # If overlap restrictions are too strict, fill the remaining tickets uniformly.
    while len(tickets) < count:
        candidate = sorted(rng.sample(range(1, 46), 6))
        if candidate not in tickets:
            tickets.append(candidate)
    rows = []
    full_freq = Counter(n for row in get_numbers(df) for n in row)
    mean_freq = (len(df) * 6 / 45) if len(df) else 1
    for i, nums in enumerate(tickets, 1):
        other_tickets = tickets[:i - 1] + tickets[i:]
        max_shared = max((len(set(nums) & set(other)) for other in other_tickets), default=0)
        rows.append({
            "组别": f"组合 {i}",
            "号码": nums,
            "号码展示": "  ·  ".join(f"{n:02d}" for n in nums),
            "和值": sum(nums),
            "奇偶比": f"{sum(n % 2 for n in nums)}:{6 - sum(n % 2 for n in nums)}",
            "历史频率指数": round(sum(full_freq[n] / mean_freq for n in nums) / 6, 2),
            "与其他组合最大重号": max_shared,
        })
    return pd.DataFrame(rows)


def run_backtest(df: pd.DataFrame, lookback: int = 100, test_draws: int = 250, seed: int = 2026) -> pd.DataFrame:
    """Walk-forward comparison: recent-frequency weighted picks vs uniform random."""
    if len(df) <= lookback + 5:
        return pd.DataFrame()
    start = max(lookback, len(df) - test_draws)
    rng = random.Random(seed)
    records = []
    nums_by_draw = get_numbers(df)
    for i in range(start, len(df)):
        train = nums_by_draw[max(0, i - lookback):i]
        freq = Counter(n for row in train for n in row)
        weights = np.array([1.0 + freq[n] for n in range(1, 46)], dtype=float)
        actual = set(nums_by_draw[i])
        for _ in range(10):
            for model_name, ticket in (
                ("近期开奖频率法", weighted_pick(rng, weights)),
                ("均匀随机基准", sorted(rng.sample(range(1, 46), 6))),
            ):
                hits = len(set(ticket) & actual)
                records.append({"方法": model_name, "命中个数": hits, "是否中3个以上": int(hits >= 3), "测试期数": int(df.iloc[i]["draw_no"])})
    return pd.DataFrame(records)


def generate_insights(df: pd.DataFrame, recent_n: int) -> list[str]:
    recent = df.tail(min(recent_n, len(df)))
    f = frequency_table(recent).sort_values("出现次数", ascending=False)
    g = gap_table(df)
    feats = draw_features(recent)
    odd_counts = feats["奇数个数"].value_counts(normalize=True).sort_index()
    top_nums = "、".join(f"{int(x):02d}" for x in f.head(5)["号码"].tolist())
    cold_nums = "、".join(f"{int(x):02d}" for x in g.head(5)["号码"].tolist())
    median_sum = int(feats["和值"].median()) if not feats.empty else 0
    common_odd = int(odd_counts.idxmax()) if len(odd_counts) else 3
    return [
        f"最近 {len(recent)} 期中，出现次数较多的号码为：{top_nums}。这是历史描述，不代表下期概率更高。",
        f"截至第 {int(df['draw_no'].max())} 期，遗漏期数较长的号码为：{cold_nums}。遗漏不会累积成下一期的优势。",
        f"最近窗口和值中位数约为 {median_sum}；奇数个数最常见的是 {common_odd} 个。它们只描述过去样本，不是开奖约束。",
    ]


# --------------------------------- UI ---------------------------------------
st.title("🎯 韩国 Lotto 6/45 分析台")
st.caption("历史统计 · 组合分析 · 滚动回测 · 候选号码生成 | 研究与娱乐用途，不构成中奖预测")

with st.sidebar:
    st.header("数据与参数")
    st.write("**数据源**：公开历史 JSON 镜像；可导入 CSV。请以官方开奖结果复核。")
    if st.button("刷新历史数据", use_container_width=True):
        try:
            with st.spinner("正在读取历史开奖数据…"):
                new_df, source = fetch_mirror_data()
                if new_df.empty:
                    st.error("数据源返回空数据。")
                else:
                    save_cache(new_df)
                    st.session_state["loaded_df"] = new_df
                    st.session_state["data_source"] = source
                    st.success(f"已更新 {len(new_df)} 期数据。")
        except Exception as exc:
            st.error(f"在线更新失败：{exc}")
    upload = st.file_uploader("导入开奖 CSV", type=["csv"])
    if upload is not None and st.button("使用导入文件", use_container_width=True):
        try:
            up_df = parse_uploaded_csv(upload)
            if up_df.empty:
                st.error("没有找到有效开奖记录。")
            else:
                save_cache(up_df)
                st.session_state["loaded_df"] = up_df
                st.session_state["data_source"] = "用户导入 CSV"
                st.success(f"已载入 {len(up_df)} 期数据。")
        except Exception as exc:
            st.error(str(exc))
    st.divider()
    recent_window = st.selectbox("近期统计窗口", [30, 50, 100, 200, 500], index=2)
    st.caption("统计窗口只影响展示，不影响真实开奖概率。")

# Load cached data first. First run attempts a live mirror fetch.
df = st.session_state.get("loaded_df")
source = st.session_state.get("data_source", "本地缓存")
if df is None:
    df = read_cache()
    if df.empty:
        try:
            with st.spinner("首次运行：正在下载历史开奖数据…"):
                df, source = fetch_mirror_data()
                save_cache(df)
                st.session_state["loaded_df"] = df
                st.session_state["data_source"] = source
        except Exception as exc:
            st.warning(f"暂时无法自动获取数据：{exc}")
            st.info("请在左侧导入包含 draw_no,n1,n2,n3,n4,n5,n6,bonus 列的 CSV。")
            st.markdown(f"官方开奖结果：[{OFFICIAL_RESULT_URL}]({OFFICIAL_RESULT_URL})")
            st.stop()
    else:
        st.session_state["loaded_df"] = df
        st.session_state["data_source"] = source

if df.empty:
    st.warning("暂无可分析数据。")
    st.stop()

df = validate_frame(df)
source = st.session_state.get("data_source", source)
latest = df.iloc[-1]

k1, k2, k3, k4 = st.columns(4)
k1.metric("已收录期数", f"{len(df):,}")
k2.metric("最新期数", f"第 {int(latest['draw_no'])} 期")
k3.metric("最新开奖日期", str(latest["date"]) or "未提供")
k4.metric("数据来源", source)
st.caption(f"当前数据覆盖：第 {int(df['draw_no'].min())} 期 — 第 {int(df['draw_no'].max())} 期。第三方镜像可能延迟或有误，请通过 [동행복권 官方结果页]({OFFICIAL_RESULT_URL}) 核验。")

latest_nums = [int(latest[f"n{i}"]) for i in range(1, 7)]
st.markdown("#### 最近一期开奖号码")
cols = st.columns(7)
for i, n in enumerate(latest_nums):
    cols[i].markdown(f"<div style='background:#1769aa;color:white;border-radius:50%;width:42px;height:42px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;margin:auto'>{n:02d}</div>", unsafe_allow_html=True)
cols[6].markdown(f"<div style='text-align:center;padding-top:10px'>Bonus<br><b>{int(latest['bonus']):02d}</b></div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["号码统计", "组合结构", "关联号码", "候选生成", "策略回测"])
recent = df.tail(min(recent_window, len(df)))

with tab1:
    st.subheader("号码出现频率")
    f1, f2 = st.columns([2, 1])
    freq_all = frequency_table(df).sort_values("号码")
    with f1:
        fig = px.bar(freq_all, x="号码", y="出现次数", title="全历史号码出现次数")
        fig.add_hline(y=len(df) * 6 / 45, line_dash="dash", annotation_text="理论期望均值")
        fig.update_layout(height=420, xaxis=dict(dtick=2))
        st.plotly_chart(fig, use_container_width=True)
    with f2:
        st.markdown("**全历史高频号码（描述性）**")
        st.dataframe(frequency_table(df).head(10), hide_index=True, use_container_width=True)
        st.markdown("**近期窗口高频号码**")
        st.dataframe(frequency_table(recent).head(10), hide_index=True, use_container_width=True)
    st.subheader("遗漏期数")
    gaps = gap_table(df).sort_values("号码")
    fig2 = px.bar(gaps, x="号码", y="距最近出现期数", title="各号码距离最近一次出现的期数")
    fig2.update_layout(height=360, xaxis=dict(dtick=2))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("“遗漏”仅表示最近一次出现距今多少期，不代表该号码即将出现。")

with tab2:
    feats = draw_features(df)
    st.subheader("和值与奇偶结构")
    a, b = st.columns(2)
    with a:
        fig = px.histogram(feats, x="和值", nbins=25, title="历史开奖号码和值分布")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        odd_df = feats["奇数个数"].value_counts().reindex(range(7), fill_value=0).rename_axis("奇数个数").reset_index(name="期数")
        fig = px.bar(odd_df, x="奇数个数", y="期数", title="每期奇数号码个数")
        st.plotly_chart(fig, use_container_width=True)
    c, d = st.columns(2)
    with c:
        high_df = feats["高区个数(23-45)"].value_counts().reindex(range(7), fill_value=0).rename_axis("高区个数").reset_index(name="期数")
        fig = px.bar(high_df, x="高区个数", y="期数", title="高区号码（23—45）个数分布")
        st.plotly_chart(fig, use_container_width=True)
    with d:
        consecutive = feats["连号对数"].value_counts().sort_index().rename_axis("连号对数").reset_index(name="期数")
        fig = px.bar(consecutive, x="连号对数", y="期数", title="每期连号对数分布")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(feats.tail(20).sort_values("期数", ascending=False), hide_index=True, use_container_width=True)

with tab3:
    st.subheader("两数组合共现频率")
    scope = st.radio("统计范围", ["全部历史", f"最近 {min(recent_window, len(df))} 期"], horizontal=True)
    pair_source = df if scope == "全部历史" else recent
    pairs = pair_table(pair_source, 30)
    st.dataframe(pairs, hide_index=True, use_container_width=True)
    st.caption("组合共现只表示历史上同期开出的次数；每个具体六数组合在下一期的理论概率相同。")

with tab4:
    st.subheader("生成下一期候选组合")
    st.warning("候选号码是统计辅助与娱乐性生成，不是预测结果。每一组具体六号码的理论头奖概率仍为 1/8,145,060。")
    left, right = st.columns(2)
    with left:
        ticket_count = st.slider("生成组数", 1, 10, 5)
        strategy = st.radio("生成方法", ["均匀随机（推荐作为基准）", "历史频率实验（仅作对照）"])
        freq_window = st.select_slider("频率实验使用多少期", options=[30, 50, 100, 200, 500], value=100)
    with right:
        overlap = st.slider("不同组合最多共享几个号码", 0, 5, 3)
        seed_text = st.text_input("随机种子（留空则每次不同）", value="")
        seed = None
        if seed_text.strip():
            try:
                seed = int(seed_text.strip())
            except ValueError:
                st.error("随机种子必须是整数。")
    if st.button("生成候选号码", type="primary"):
        picks = make_candidates(df, ticket_count, strategy, freq_window, overlap, seed)
        st.session_state["last_picks"] = picks
    picks = st.session_state.get("last_picks")
    if picks is not None:
        st.dataframe(picks[["组别", "号码展示", "和值", "奇偶比", "历史频率指数", "与其他组合最大重号"]], hide_index=True, use_container_width=True)
        if (picks["与其他组合最大重号"] > overlap).any():
            st.warning("组数与重号限制组合过严，已自动放宽限制以补足组数。表格中的‘与其他组合最大重号’显示实际值。")
        csv_bytes = picks.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载候选号码 CSV", csv_bytes, file_name="lotto645_candidates.csv", mime="text/csv")
        with st.expander("本期统计解读"):
            for line in generate_insights(df, recent_window):
                st.write("• " + line)
        st.caption("历史频率指数仅用于描述候选号码在全历史中的出现情况，不是中奖概率或模型置信度。")

with tab5:
    st.subheader("滚动历史回测：频率法 vs 均匀随机")
    st.write("对每个测试期，只使用它之前的历史开奖作为训练数据；每个测试期各生成 10 注，减少单次随机波动。")
    c1, c2 = st.columns(2)
    with c1:
        lookback = st.selectbox("训练窗口", [50, 100, 150, 200], index=1)
    with c2:
        tests = st.slider("最近测试期数", 50, min(500, max(50, len(df) - lookback)), min(200, max(50, len(df) - lookback)))
    if st.button("运行回测", type="primary"):
        with st.spinner("正在进行滚动回测…"):
            bt = run_backtest(df, lookback=lookback, test_draws=tests)
            st.session_state["backtest_result"] = bt
    bt = st.session_state.get("backtest_result")
    if bt is not None and not bt.empty:
        summary = bt.groupby("方法").agg(
            平均命中数=("命中个数", "mean"),
            三个及以上命中率=("是否中3个以上", "mean"),
            测试注数=("命中个数", "count"),
        ).reset_index()
        summary["三个及以上命中率"] = (summary["三个及以上命中率"] * 100).round(3).astype(str) + "%"
        summary["平均命中数"] = summary["平均命中数"].round(4)
        st.dataframe(summary, hide_index=True, use_container_width=True)
        fig = px.histogram(bt, x="命中个数", color="方法", barmode="group", histnorm="probability", title="每注命中号码个数分布")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("回测只衡量这段历史样本中的表现。若某方法暂时领先，可能来自随机波动，不代表未来能持续领先；请不要据此增加投注金额。")
    else:
        st.info(f"数据量至少需要超过训练窗口 {lookback} 期。当前收录 {len(df)} 期。")

st.divider()
st.markdown("**方法说明**：Lotto 6/45 每期从 1—45 中抽取 6 个号码。若开奖机制公平且各期独立，历史频率、遗漏和号码共现不能改变下一期单注的理论中奖概率。")
st.markdown(f"数据源项目说明：[公开历史 JSON 数据集](https://github.com/smok95/lotto)；官方核验入口：[동행복권 开奖结果]({OFFICIAL_RESULT_URL})。")
