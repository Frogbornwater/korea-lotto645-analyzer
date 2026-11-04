# -*- coding: utf-8 -*-
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from lotto_core import (
    COLUMNS, draw_features, extract_records, frequency_table, gap_table,
    generate_candidates, insights, merge_histories, pair_table, quality_report,
    run_backtest, validate_frame,
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CSV_PATH = DATA_DIR / "lotto645.csv"
STATUS_PATH = DATA_DIR / "sync_status.json"
OFFICIAL_RESULT_URL = "https://www.dhlottery.co.kr/lt645/result"
LATEST_URL = "https://www.dhlottery.co.kr/lt645/selectPstLt645Info.do"
WINDOW_URL = "https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do"
MIRROR_ALL_URL = "https://smok95.github.io/lotto/results/all.json"
REPO_CSV_URL = "https://raw.githubusercontent.com/Frogbornwater/korea-lotto645-analyzer/main/data/lotto645.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Lotto645Research/2.0)",
           "Accept": "application/json,text/plain,*/*"}

st.set_page_config(page_title="韩国 Lotto 6/45 AI 研究台 V2", page_icon="🎯", layout="wide")


def _get_json(url: str, params: dict[str, Any] | None = None) -> dict:
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, dict):
        raise ValueError("数据源没有返回 JSON 对象")
    return payload


def fetch_online_history() -> tuple[pd.DataFrame, str, list[str]]:
    """Official-site result endpoint first; third-party mirror is fallback/bootstrap."""
    warnings: list[str] = []
    official = pd.DataFrame(columns=COLUMNS)
    mirror = pd.DataFrame(columns=COLUMNS)
    try:
        latest_payload = _get_json(LATEST_URL)
        latest_records = extract_records(latest_payload)
        if not latest_records:
            raise ValueError("最新期接口返回结构无法识别")
        latest_no = max(x["draw_no"] for x in latest_records)
        detail_payload = _get_json(WINDOW_URL, {"srchDir": "center", "srchLtEpsd": latest_no})
        detail_records = extract_records(detail_payload)
        official = merge_histories(pd.DataFrame(latest_records), pd.DataFrame(detail_records))
        if official.empty or int(official["draw_no"].max()) != latest_no:
            raise ValueError("官方结果页的最新期校验未通过")
    except Exception as exc:
        warnings.append(f"官方站点数据暂时不可用：{exc}")
    try:
        mirror_records = extract_records(_get_json(MIRROR_ALL_URL))
        mirror = validate_frame(pd.DataFrame(mirror_records))
        if mirror.empty:
            raise ValueError("镜像没有有效记录")
    except Exception as exc:
        warnings.append(f"第三方历史镜像暂时不可用：{exc}")
    if official.empty and mirror.empty:
        raise RuntimeError("官方站点和备用数据源都无法获取数据。可尝试导入 CSV。\n" + "\n".join(warnings))
    combined = merge_histories(mirror, official)
    source = "官方结果页 + 第三方历史镜像" if not official.empty and not mirror.empty else (
        "官方结果页 JSON" if not official.empty else "第三方历史镜像（备用）")
    if not official.empty and not mirror.empty:
        m = mirror.set_index("draw_no")
        o = official.set_index("draw_no")
        overlap = sorted(set(m.index) & set(o.index))
        conflicts = [n for n in overlap if m.loc[n, NUMBER_COLS + ["bonus"]].tolist() != o.loc[n, NUMBER_COLS + ["bonus"]].tolist()]
        if conflicts:
            warnings.append(f"官方结果页与镜像在 {len(conflicts)} 期存在号码差异；已优先采用官方结果页中有覆盖的期次。差异期数示例：{conflicts[:8]}")
    return combined, source, warnings


def load_repo_history() -> tuple[pd.DataFrame, str]:
    # Pull the tracked CSV from GitHub so a running Streamlit container can see
    # data committed by the scheduled GitHub Actions workflow without relying
    # on its ephemeral local disk. Fall back to the local checkout when offline.
    try:
        response = requests.get(REPO_CSV_URL, headers=HEADERS, timeout=12)
        if response.ok and response.text.strip():
            remote = validate_frame(pd.read_csv(io.StringIO(response.text)))
            if not remote.empty:
                return remote, "GitHub 仓库持久化历史数据"
    except Exception:
        pass
    if CSV_PATH.exists():
        try:
            local = validate_frame(pd.read_csv(CSV_PATH, encoding="utf-8-sig"))
            if not local.empty:
                return local, "部署容器本地数据（临时缓存）"
        except Exception:
            pass
    return pd.DataFrame(columns=COLUMNS), "尚无持久化历史数据"


def parse_uploaded_csv(file) -> pd.DataFrame:
    raw = pd.read_csv(file)
    return validate_frame(raw)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_online_history():
    return fetch_online_history()


st.title("🎯 韩国 Lotto 6/45 AI 研究台")
st.caption("数据质量 · 统计分析 · 多策略实验 · 样本外回测 | 研究与娱乐用途，不构成中奖预测")

with st.sidebar:
    st.header("数据中心")
    st.markdown(f"[동행복권 官方结果核验]({OFFICIAL_RESULT_URL})")
    if st.button("检查并刷新在线数据", use_container_width=True, type="primary"):
        cached_online_history.clear()
        try:
            with st.spinner("正在读取官方结果页并核对备用历史镜像…"):
                live_df, live_source, live_warnings = cached_online_history()
            repo_df, repo_source = load_repo_history()
            current = st.session_state.get("active_df", repo_df)
            st.session_state["active_df"] = merge_histories(repo_df, current, live_df)
            st.session_state["source"] = live_source
            st.session_state["warnings"] = live_warnings
            st.success(f"在线数据检查完成，最新收录第 {int(st.session_state['active_df']['draw_no'].max())} 期。")
        except Exception as exc:
            st.error(str(exc))
    upload = st.file_uploader("导入开奖 CSV", type=["csv"])
    if upload is not None and st.button("载入 CSV", use_container_width=True):
        try:
            imported = parse_uploaded_csv(upload)
            if imported.empty:
                st.error("CSV 中没有有效开奖记录。")
            else:
                st.session_state["active_df"] = imported
                st.session_state["source"] = "用户导入 CSV"
                st.session_state["warnings"] = []
                st.success(f"载入 {len(imported)} 期记录。")
        except Exception as exc:
            st.error(f"CSV 格式错误：{exc}")
    st.divider()
    recent_window = st.selectbox("近期统计窗口", [30, 50, 100, 200, 500], index=2)
    st.caption("统计窗口仅改变样本范围，不改变随机开奖概率。")

# Load repository-tracked data first. If missing, bootstrap via online sources.
df = st.session_state.get("active_df")
source = st.session_state.get("source", "GitHub 仓库持久化历史数据")
warnings = st.session_state.get("warnings", [])
if df is None:
    df, repo_source = load_repo_history()
    source = repo_source
    if not df.empty:
        st.session_state["active_df"] = df
        st.session_state["source"] = source
    else:
        try:
            with st.spinner("首次启动：正在获取历史开奖记录…"):
                online_df, source, warnings = cached_online_history()
            df = online_df
            st.session_state["active_df"] = df
            st.session_state["source"] = source
            st.session_state["warnings"] = warnings
        except Exception as exc:
            st.error(str(exc))
            st.info("也可以先在左侧导入 CSV。CSV 至少需要 draw_no,n1,n2,n3,n4,n5,n6,bonus 字段。")
            st.stop()

if df is None or df.empty:
    st.warning("暂无可分析数据。")
    st.stop()
df = validate_frame(df)
source = st.session_state.get("source", source)
warnings = st.session_state.get("warnings", warnings)
quality = quality_report(df)
latest = df.iloc[-1]
recent = df.tail(min(recent_window, len(df)))

if warnings:
    with st.expander(f"数据源提醒（{len(warnings)}）", expanded=False):
        for warning in warnings:
            st.warning(warning)
if quality["missing_draws"]:
    st.warning(f"历史序列存在 {len(quality['missing_draws'])} 个缺失期数；回测结果可能受数据缺口影响。示例：{quality['missing_draws'][:12]}")
if quality["date_errors"]:
    st.warning(f"有 {quality['date_errors']} 条日期格式需要人工核验。")

c1, c2, c3, c4 = st.columns(4)
c1.metric("已收录期数", f"{quality['rows']:,}")
c2.metric("最新期数", f"第 {quality['last_draw']} 期")
c3.metric("最新开奖日期", str(latest["date"] or "未提供"))
c4.metric("数据来源", source)
st.caption(f"数据覆盖：第 {quality['first_draw']} 期至第 {quality['last_draw']} 期。官方结果页使用的 JSON 查询路径可能随网站改版而变化；如有差异，请以官方页面为准。")

latest_nums = [int(latest[f"n{i}"]) for i in range(1, 7)]
st.markdown("#### 最近一期开奖号码")
ball_cols = st.columns(7)
for i, n in enumerate(latest_nums):
    ball_cols[i].markdown(f"<div style='background:#1769aa;color:white;border-radius:50%;width:42px;height:42px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;margin:auto'>{n:02d}</div>", unsafe_allow_html=True)
ball_cols[6].markdown(f"<div style='text-align:center;padding-top:10px'>Bonus<br><b>{int(latest['bonus']):02d}</b></div>", unsafe_allow_html=True)

# Tabs
tabs = st.tabs(["数据质量", "号码统计", "组合结构", "关联号码", "多策略选号", "科学回测"])
with tabs[0]:
    st.subheader("数据完整性检查")
    q1, q2, q3 = st.columns(3)
    q1.metric("记录数量", f"{quality['rows']:,}")
    q2.metric("缺失期数", len(quality["missing_draws"]))
    q3.metric("日期格式异常", quality["date_errors"])
    st.dataframe(df.tail(20).sort_values("draw_no", ascending=False), hide_index=True, use_container_width=True)
    st.download_button("下载当前历史数据 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="lotto645_history.csv", mime="text/csv")
    if STATUS_PATH.exists():
        try:
            import json
            status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
            st.markdown("**最近一次 GitHub Actions 数据同步**")
            st.json(status)
        except Exception:
            st.caption("同步状态文件暂时无法读取。")
    else:
        st.info("自动同步工作流第一次运行后，会在这里显示同步时间和结果。")

with tabs[1]:
    st.subheader("号码出现频率与标准化偏差")
    freq = frequency_table(recent).sort_values("号码")
    f1, f2 = st.columns([2, 1])
    with f1:
        fig = px.bar(freq, x="号码", y="出现次数", title=f"最近 {len(recent)} 期号码频率")
        fig.add_hline(y=len(recent) * 6 / 45, line_dash="dash", annotation_text="理论期望")
        fig.update_layout(height=420, xaxis=dict(dtick=2))
        st.plotly_chart(fig, use_container_width=True)
    with f2:
        st.markdown("**频率最高的号码（描述性）**")
        st.dataframe(freq.sort_values("出现次数", ascending=False).head(10), hide_index=True, use_container_width=True)
        st.markdown("**标准化偏差较大的号码**")
        st.dataframe(freq.reindex(freq["标准化偏差Z"].abs().sort_values(ascending=False).index).head(10), hide_index=True, use_container_width=True)
    st.caption("Z 值与近似 P 值仅作探索性描述；同时查看 45 个号码会产生多重比较问题，不能据此确认存在可利用的偏差。")
    st.subheader("遗漏期数")
    gaps = gap_table(df).sort_values("号码")
    fig2 = px.bar(gaps, x="号码", y="距最近出现期数", title="各号码距离最近一次出现的期数")
    fig2.update_layout(height=360, xaxis=dict(dtick=2))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("遗漏只表示最近一次出现距今多少期，不表示号码更可能在下一期出现。")

with tabs[2]:
    features = draw_features(df)
    st.subheader("和值、奇偶与区间结构")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.histogram(features, x="和值", nbins=25, title="历史和值分布"), use_container_width=True)
    with b:
        odd = features["奇数个数"].value_counts().reindex(range(7), fill_value=0).rename_axis("奇数个数").reset_index(name="期数")
        st.plotly_chart(px.bar(odd, x="奇数个数", y="期数", title="每期奇数个数分布"), use_container_width=True)
    c, d = st.columns(2)
    with c:
        high = features["高区个数(23-45)"].value_counts().reindex(range(7), fill_value=0).rename_axis("高区个数").reset_index(name="期数")
        st.plotly_chart(px.bar(high, x="高区个数", y="期数", title="高区号码（23—45）数量分布"), use_container_width=True)
    with d:
        runs = features["连号对数"].value_counts().sort_index().rename_axis("连号对数").reset_index(name="期数")
        st.plotly_chart(px.bar(runs, x="连号对数", y="期数", title="每期连号对数分布"), use_container_width=True)
    st.dataframe(features.tail(20).sort_values("期数", ascending=False), hide_index=True, use_container_width=True)

with tabs[3]:
    scope = st.radio("共现统计范围", ["全部历史", f"最近 {len(recent)} 期"], horizontal=True)
    source_df = df if scope == "全部历史" else recent
    st.subheader("两数组合共现")
    st.dataframe(pair_table(source_df, 30), hide_index=True, use_container_width=True)
    st.caption("理论上指定两数同期开出的概率为 1/66。偏离值是探索性统计，未校正对 990 对号码的多重比较。")

with tabs[4]:
    st.subheader("多策略候选号码实验室")
    st.warning("候选组合不是预测。公平独立开奖时，任意一组具体六号码的头奖概率均为 1/8,145,060。")
    models = ["均匀随机基准", "近期频率倾向（实验）", "指数时间衰减（实验）", "遗漏倾向（实验）", "结构约束随机（实验）"]
    col_a, col_b = st.columns(2)
    with col_a:
        count = st.slider("生成组数", 1, 10, 5)
        model = st.selectbox("生成策略", models, index=0)
        window = st.select_slider("策略使用的历史窗口", options=[30, 50, 100, 200, 500], value=100)
    with col_b:
        overlap = st.slider("任意两组最多共享号码数", 0, 5, 3)
        seed_text = st.text_input("随机种子（留空则每次不同）", value="")
        seed = None
        if seed_text.strip():
            try:
                seed = int(seed_text.strip())
            except ValueError:
                st.error("随机种子必须是整数。")
    if st.button("生成候选组合", type="primary", key="generate_v2"):
        try:
            generated, relaxed = generate_candidates(df, count, model, window, overlap, seed)
            st.session_state["generated_v2"] = generated
            st.session_state["generated_v2_relaxed"] = relaxed
        except Exception as exc:
            st.error(f"生成失败：{exc}")
    generated = st.session_state.get("generated_v2")
    if generated is not None:
        st.dataframe(generated[["组别", "生成策略", "号码展示", "和值", "奇偶比", "历史频率指数", "与其他组合最大重号"]], hide_index=True, use_container_width=True)
        if st.session_state.get("generated_v2_relaxed"):
            st.warning("重号限制过严，已放宽限制以补足组数；请以表格显示的实际重号数为准。")
        st.download_button("下载候选号码 CSV", generated.to_csv(index=False).encode("utf-8-sig"),
                           file_name="lotto645_candidates.csv", mime="text/csv")
        with st.expander("本次统计解读"):
            for line in insights(df, recent_window):
                st.write("• " + line)

with tabs[5]:
    st.subheader("滚动样本外回测")
    st.write("每个测试期只使用此前开奖数据构建策略；同一测试期比较随机基准、近期频率、指数时间衰减。")
    ca, cb, cc = st.columns(3)
    with ca:
        lookback = st.selectbox("训练窗口", [50, 100, 150, 200], index=1)
    with cb:
        max_tests = max(50, min(300, len(df) - lookback))
        test_count = st.slider("最近测试期数", 50, max_tests, min(150, max_tests))
    with cc:
        seed_count = st.selectbox("随机种子组数", [3, 5, 10], index=1)
    tickets_per_draw = st.selectbox("每期每策略生成注数", [3, 5, 10], index=1)
    if st.button("运行多模型回测", type="primary", key="backtest_v2"):
        with st.spinner("正在执行滚动样本外回测与按期 Bootstrap…"):
            raw, summary = run_backtest(df, lookback, test_count, tickets_per_draw, seed_count)
            st.session_state["bt_v2_raw"] = raw
            st.session_state["bt_v2_summary"] = summary
    summary = st.session_state.get("bt_v2_summary")
    raw = st.session_state.get("bt_v2_raw")
    if summary is not None and not summary.empty:
        display = summary.copy()
        for c in ["平均命中数", "平均命中差值_vs随机", "平均命中差值95%CI下限", "平均命中差值95%CI上限"]:
            display[c] = display[c].round(4)
        for c in ["3个以上命中率", "4个以上命中率", "5个以上命中率", "6个命中率", "3+率差值_vs随机", "3+率差值95%CI下限", "3+率差值95%CI上限"]:
            display[c] = (display[c] * 100).round(3).astype(str) + "%"
        st.dataframe(display, hide_index=True, use_container_width=True)
        fig = px.histogram(raw, x="命中数", color="策略", barmode="group", histnorm="probability", title="各策略单注命中数分布")
        st.plotly_chart(fig, use_container_width=True)
        st.info("解读：比较模型相对随机基准的差值及 95% Bootstrap 区间。区间若覆盖 0，不能据此认定模型优于随机；即使不覆盖 0，也只表示这段历史样本的表现，不保证未来。")
        st.download_button("下载完整回测明细 CSV", raw.to_csv(index=False).encode("utf-8-sig"),
                           file_name="lotto645_backtest_v2.csv", mime="text/csv")
    else:
        st.caption(f"建议至少有 {lookback + 6} 期有效记录。当前收录 {len(df)} 期。")

st.divider()
st.markdown("**统计原则**：在公平且各期独立的 Lotto 6/45 抽奖机制下，历史频率、遗漏、号码共现和结构分布不会改变下一期的理论概率。这里的模型用于比较选号规则的历史行为，不构成投注建议。")
st.markdown(f"数据查询参考：[동행복권 官方结果页]({OFFICIAL_RESULT_URL})；官方站点接口是页面使用的公开 JSON 查询路径，并非承诺长期稳定的正式开放 API。备用历史数据：[Smok95 公开数据镜像](https://github.com/smok95/lotto)。")
