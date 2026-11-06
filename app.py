# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from lotto_core import (
    COLUMNS, MODEL_NAMES, compare_ticket_to_history, draw_features, frequency_table,
    gap_table, generate_candidates, number_score_table, pair_table, quality_report,
    run_backtest, validate_frame, theoretical_random_metrics,
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CSV_PATH = DATA_DIR / "lotto645.csv"
STATUS_PATH = DATA_DIR / "sync_status.json"
OFFICIAL_RESULT_URL = "https://www.dhlottery.co.kr/lt645/result"

st.set_page_config(page_title="Lotto 645 V3 · 历史数据预测实验室", page_icon="🎯", layout="wide")
st.title("🎯 韩国 Lotto 6/45 · V3 历史数据预测实验室")
st.caption("官方历史数据优先｜多模型候选生成｜滚动样本外回测｜研究与娱乐用途")


@st.cache_data(ttl=300, show_spinner=False)
def load_repo_data() -> tuple[pd.DataFrame, dict]:
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=COLUMNS), {}
    raw = pd.read_csv(CSV_PATH)
    status = {}
    if STATUS_PATH.exists():
        try:
            status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            status = {}
    return validate_frame(raw), status


df, sync_status = load_repo_data()
if df.empty:
    st.error("还没有找到官方历史开奖数据库。先到 GitHub 仓库的 Actions，运行 ‘同步韩国官方 Lotto 历史数据’ 工作流。同步成功后，data/lotto645.csv 会自动提交到仓库，Streamlit 随后重新加载。")
    st.markdown(f"官方开奖查询：[{OFFICIAL_RESULT_URL}]({OFFICIAL_RESULT_URL})")
    if sync_status:
        st.json(sync_status)
    st.stop()

quality = quality_report(df)
last = df.iloc[-1]
last_nums = [int(last[f"n{i}"]) for i in range(1, 7)]

if sync_status.get("status") == "failed":
    st.warning("最近一次官方数据同步失败。当前页面使用仓库中上一次成功保存的数据；请查看数据状态详情。")

m1, m2, m3, m4 = st.columns(4)
m1.metric("官方历史记录", f"{quality['rows']:,} 期")
m2.metric("最新期数", f"第 {quality['last_draw']} 期")
m3.metric("最新开奖日期", str(last["date"] or "未提供"))
m4.metric("历史序列", "完整" if quality["sequence_complete"] else f"缺 {len(quality['missing_draws'])} 期")

st.markdown("### 最近一期官方开奖号码")
ball_cols = st.columns(7)
for i, n in enumerate(last_nums):
    ball_cols[i].markdown(
        f"<div style='background:#1769aa;color:white;border-radius:50%;width:44px;height:44px;display:flex;align-items:center;justify-content:center;font-weight:700;margin:auto'>{n:02d}</div>",
        unsafe_allow_html=True,
    )
ball_cols[6].markdown(f"<div style='text-align:center;padding-top:10px'>Bonus<br><b>{int(last['bonus']):02d}</b></div>", unsafe_allow_html=True)

if not quality["sequence_complete"]:
    st.warning(f"历史数据不是从第 1 期连续覆盖。缺失期示例：{quality['missing_draws'][:20]}。建议先重新运行官方数据同步。")

st.caption(f"数据来源：{sync_status.get('source', '仓库历史数据')}；同步时间：{sync_status.get('updated_at_utc', '未知')}。官方站点页面接口不是稳定承诺的开放 API。")

tabs = st.tabs(["V3 号码预测", "历史数据分析", "模型回测", "数据完整性", "选号复盘"])

with tabs[0]:
    st.subheader("历史数据驱动的候选号码")
    st.warning("候选号码是模型按历史样本偏好生成的研究结果，不是中奖概率预测。公平独立开奖时，任意一组具体六号码的头奖概率都是 1/8,145,060。")
    left, right = st.columns([1, 1])
    with left:
        model = st.selectbox("选号模型", MODEL_NAMES, index=4)
        window_label = st.selectbox("历史数据窗口", ["最近30期", "最近50期", "最近100期", "最近200期", "最近500期", "全部历史"], index=2)
        window = None if window_label == "全部历史" else int(window_label.replace("最近", "").replace("期", ""))
    with right:
        count = st.slider("生成候选组数", 1, 20, 10)
        overlap = st.slider("任意两组最多共享号码数", 0, 5, 4)
        seed_text = st.text_input("随机种子（留空则每次不同）", "")
    seed = None
    if seed_text.strip():
        try:
            seed = int(seed_text.strip())
        except ValueError:
            st.error("随机种子必须是整数。")
    if st.button("生成 V3 候选号码", type="primary", use_container_width=True):
        try:
            with st.spinner("正在根据所选历史窗口计算模型权重并生成候选号码…"):
                generated, relaxed = generate_candidates(df, count, model, window or len(df), overlap, seed)
            st.session_state["v3_candidates"] = generated
            st.session_state["v3_relaxed"] = relaxed
            st.session_state["v3_model"] = model
            st.session_state["v3_window"] = window_label
        except Exception as exc:
            st.error(f"生成失败：{exc}")
    generated = st.session_state.get("v3_candidates")
    if generated is not None:
        st.markdown(f"#### 候选结果：{st.session_state.get('v3_model')} · {st.session_state.get('v3_window')}")
        st.dataframe(generated.drop(columns=["号码列表"]), hide_index=True, use_container_width=True)
        if st.session_state.get("v3_relaxed"):
            allowed = int(generated["实际允许最大重号"].iloc[0])
            requested = int(generated["请求跨组最大重号"].iloc[0])
            actual = int(generated["实际跨组最大重号"].iloc[0])
            st.info(
                f"为满足候选组数，跨组重号上限由 {requested} 放宽到 {allowed}；"
                f"本批候选实际最大重号为 {actual}。仍按所选模型生成，没有切换成纯随机补齐。"
            )
        st.download_button("下载候选号码 CSV", generated.drop(columns=["号码列表"]).to_csv(index=False).encode("utf-8-sig"),
                           "lotto645_v3_candidates.csv", "text/csv")
        st.caption("模型相对得分只反映所选算法下号码的相对偏好，经当前样本缩放；不是单号或组合的中奖概率。")
    st.markdown("#### 五种频率模型的号码综合相对评分")
    selected_window = len(df) if window is None else min(window, len(df))
    score_table = number_score_table(df, selected_window)
    fig = px.bar(score_table.sort_values("号码"), x="号码", y="综合相对得分", title="45 个号码的综合相对评分（0—100）")
    fig.update_layout(height=360, xaxis=dict(dtick=2))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(score_table.head(15).round(3), hide_index=True, use_container_width=True)

with tabs[1]:
    st.subheader("官方历史数据统计")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        stat_scope = st.selectbox("统计窗口", ["最近30期", "最近50期", "最近100期", "最近200期", "最近500期", "全部历史"], index=2, key="stat_scope")
    stat_n = len(df) if stat_scope == "全部历史" else int(stat_scope.replace("最近", "").replace("期", ""))
    sample = df.tail(min(stat_n, len(df)))
    freq = frequency_table(sample)
    gap = gap_table(df)
    with col_b:
        st.caption(f"统计范围：第 {int(sample.iloc[0]['draw_no'])} 期至第 {int(sample.iloc[-1]['draw_no'])} 期，共 {len(sample)} 期")
    a, b = st.columns(2)
    with a:
        chart = px.bar(freq.sort_values("号码"), x="号码", y="出现次数", title="号码出现频率")
        chart.add_hline(y=len(sample) * 6 / 45, line_dash="dash", annotation_text="理论期望")
        chart.update_layout(height=400, xaxis=dict(dtick=2))
        st.plotly_chart(chart, use_container_width=True)
    with b:
        st.markdown("**出现次数靠前的号码**")
        st.dataframe(freq.head(12), hide_index=True, use_container_width=True)
        st.markdown("**当前遗漏较长的号码（仅描述）**")
        st.dataframe(gap.head(10), hide_index=True, use_container_width=True)
    features = draw_features(sample)
    c, d = st.columns(2)
    with c:
        st.plotly_chart(px.histogram(features, x="和值", nbins=25, title="历史和值分布"), use_container_width=True)
    with d:
        odd = features["奇数个数"].value_counts().reindex(range(7), fill_value=0).rename_axis("奇数个数").reset_index(name="期数")
        st.plotly_chart(px.bar(odd, x="奇数个数", y="期数", title="奇数个数分布"), use_container_width=True)
    st.markdown("**两数组合历史共现次数**")
    st.dataframe(pair_table(sample, 30), hide_index=True, use_container_width=True)
    st.download_button("下载官方历史 CSV", df.to_csv(index=False).encode("utf-8-sig"), "lotto645_official_history.csv", "text/csv")

with tabs[2]:
    st.subheader("多模型滚动样本外回测")
    st.write("每次测试一期开奖结果时，模型只能看到该期之前的数据；模型与随机基准在相同测试期进行比较。")
    x, y, z = st.columns(3)
    with x:
        lookback = st.selectbox("训练窗口", [50, 100, 150, 200, 300, 500], index=3)
    with y:
        max_test = max(10, min(150, len(df) - lookback))
        test_count = st.slider("最近测试期数", 10, max_test, min(80, max_test))
    with z:
        seeds = st.selectbox("随机种子组数", [1, 3, 5], index=1)
    tickets = st.selectbox("每期每模型模拟注数", [1, 3, 5, 10], index=2)
    if st.button("运行样本外回测", type="primary", use_container_width=True):
        progress = st.progress(0, text="正在准备历史数据…")
        progress_note = st.empty()
        def update_backtest_progress(completed, total, draw_no):
            ratio = completed / max(total, 1)
            progress.progress(ratio, text=f"回测进度：{completed}/{total} 期（{ratio:.0%}）")
            progress_note.caption(f"正在评估第 {draw_no} 期；每次预测仅使用此前历史数据。")
        try:
            raw, summary = run_backtest(
                df, lookback, test_count, tickets, seeds,
                progress_callback=update_backtest_progress,
            )
            st.session_state["v3_bt_raw"] = raw
            st.session_state["v3_bt_summary"] = summary
            if not summary.empty:
                st.success(f"回测完成：{test_count} 期、{len(summary)} 种策略、{len(raw):,} 注模拟号码。")
            else:
                st.warning(f"可用历史数据不足。当前 {len(df)} 期，训练窗口为 {lookback} 期。")
        except Exception as exc:
            st.error(f"回测失败：{type(exc).__name__}: {exc}")
    summary = st.session_state.get("v3_bt_summary")
    raw = st.session_state.get("v3_bt_raw")
    if summary is not None and not summary.empty:
        view = summary.copy()
        for col in ["平均命中数", "平均命中差_vs随机", "平均命中差95%CI下限", "平均命中差95%CI上限"]:
            view[col] = view[col].round(4)
        for col in ["3个以上命中率", "4个以上命中率", "5个以上命中率", "6个命中率"]:
            view[col] = (view[col] * 100).round(3).astype(str) + "%"
        st.dataframe(view, hide_index=True, use_container_width=True)
        st.plotly_chart(px.histogram(raw, x="命中数", color="策略", barmode="group", histnorm="probability", title="各模型单注命中数分布"), use_container_width=True)
        st.info("重点看模型相对随机基准的平均命中差及 95% Bootstrap 区间。区间跨过 0 时，没有证据表明该模型优于随机。回测领先也不代表未来领先。")
        st.download_button("下载回测明细", raw.to_csv(index=False).encode("utf-8-sig"), "lotto645_v3_backtest.csv", "text/csv")
    else:
        st.caption(f"建议至少准备 {lookback + 6} 期连续历史数据。当前数据：{len(df)} 期。")

with tabs[3]:
    st.subheader("数据来源与完整性")
    q1, q2, q3 = st.columns(3)
    q1.metric("数据记录", quality["rows"])
    q2.metric("缺失期数", len(quality["missing_draws"]))
    q3.metric("日期异常", quality["date_errors"])
    st.write({"首期": quality["first_draw"], "末期": quality["last_draw"], "从第1期开始连续": quality["sequence_complete"]})
    if quality["missing_draws"]:
        st.error(f"缺失期号：{quality['missing_draws'][:50]}")
    st.markdown("**最近一次官方同步状态**")
    st.json(sync_status if sync_status else {"status": "未找到同步状态文件"})
    st.markdown(f"官方查询页面：[동행복권 Lotto 6/45]({OFFICIAL_RESULT_URL})")
    st.download_button("下载同步状态 JSON", json.dumps(sync_status, ensure_ascii=False, indent=2), "lotto645_sync_status.json", "application/json")

with tabs[4]:
    st.subheader("自选号码历史复盘")
    raw_ticket = st.text_input("输入 6 个号码（逗号、空格分隔）", placeholder="例如：3, 11, 18, 27, 34, 42")
    if st.button("复盘这组号码"):
        try:
            values = [int(x) for x in raw_ticket.replace("，", ",").replace(" ", ",").split(",") if x.strip()]
            result = compare_ticket_to_history(df, values)
            st.write(f"检验期数：{result['draws_checked']}；历史最高命中：{result['max_hits']}；平均命中：{result['mean_hits']:.3f}")
            dist = pd.DataFrame([{"命中号码数": k, "历史期数": v} for k, v in result["hit_distribution"].items()])
            st.dataframe(dist, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

st.divider()
metrics = theoretical_random_metrics()
st.caption(
    f"统计边界：每注理论平均命中主号码数为 {metrics['mean_hits_per_ticket']:.1f}；"
    f"至少命中3个主号码的理论概率约为 {metrics['p3plus']:.2%}；"
    f"头奖概率为 1/{int(1 / metrics['jackpot_probability']):,}。历史数据不能改变公平独立开奖的理论概率。"
)
st.caption("数据策略：只把官方结果页 JSON 作为历史数据源；官方接口失败时保留旧数据并报告错误，不以第三方镜像静默补齐。")
