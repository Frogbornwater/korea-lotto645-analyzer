# -*- coding: utf-8 -*-
"""Lotto 6/45 V3.6 Phase 1 - offline/manual-data analysis application."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from backtest_engine import run_backtest, run_multi_window_backtest
from data_manager import load_repository_history, read_manual_upload
from lotto_core import (
    COLUMNS, MODEL_NAMES, compare_ticket_to_history, draw_features, frequency_table,
    gap_table, generate_candidates, number_score_table, pair_table, quality_report,
    theoretical_random_metrics, validate_frame,
)
from model_registry import ENGINE_VERSION, get_model_registry

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
HISTORY_PATH = DATA_DIR / "lotto645.csv"

st.set_page_config(page_title="Lotto 6/45 V3.6 · 离线分析实验室", page_icon="🎯", layout="wide")
st.title("🎯 韩国 Lotto 6/45 · V3.6 离线分析实验室")
st.caption("手动历史数据导入｜多模型分析｜可复现滚动回测｜研究与娱乐用途")


def read_uploaded_history(uploaded_file):
    return read_manual_upload(io.BytesIO(uploaded_file.getvalue()), uploaded_file.name)


with st.sidebar:
    st.header("历史数据")
    upload = st.file_uploader(
        "手动导入历史开奖文件",
        type=["csv", "xlsx", "xls"],
        help="仅读取你手动提供的 CSV 或 Excel 文件；上传文件只在当前应用会话中使用。",
    )
    st.caption("也可以将手动整理的 CSV 放入仓库 data/lotto645.csv。")

history_report: dict = {}
data_source = "仓库手动维护文件"
try:
    if upload is not None:
        df, history_report = read_uploaded_history(upload)
        data_source = f"本次手动上传：{upload.name}"
    else:
        df, history_report = load_repository_history(HISTORY_PATH)
except Exception as exc:
    st.error(f"历史数据读取失败：{exc}")
    st.stop()

if df.empty:
    st.warning("当前没有可用历史数据。请在左侧手动上传 CSV / Excel，或将手动整理的历史 CSV 放入 data/lotto645.csv。")
    template = pd.DataFrame(columns=COLUMNS)
    st.download_button(
        "下载历史数据 CSV 模板",
        template.to_csv(index=False).encode("utf-8-sig"),
        "lotto645_history_template.csv",
        "text/csv",
    )
    st.stop()

quality = quality_report(df)
last = df.iloc[-1]
last_nums = [int(last[f"n{i}"]) for i in range(1, 7)]
if history_report.get("invalid_rows", 0) > 0:
    st.warning(f"导入文件中有 {history_report['invalid_rows']} 行未通过号码格式校验，分析仅使用 {len(df)} 条有效记录。请检查原始文件。")
if not quality["range_complete"]:
    st.warning(f"当前文件覆盖范围内存在 {len(quality['missing_draws'])} 个缺失期号。请手动补齐后再运行回测。")
elif quality["first_draw"] != 1:
    st.info(f"当前手动数据从第{quality['first_draw']}期开始，文件覆盖范围内连续，但不是从第1期开始。")

m1, m2, m3, m4 = st.columns(4)
m1.metric("有效历史记录", f"{quality['rows']:,}期")
m2.metric("最新期号", f"第{quality['last_draw']}期")
m3.metric("最新开奖日期", str(last["date"] or "未提供"))
m4.metric("历史序列", "从第1期连续" if quality["sequence_complete"] else f"缺{len(quality['missing_draws'])}期")
st.caption(f"数据来源：{data_source}｜引擎版本：{ENGINE_VERSION}｜数据仅由用户手动提供")

page_overview, page_pick, page_stats, page_backtest, page_models, page_data, page_review = st.tabs([
    "系统总览", "智能选号", "历史统计", "回测中心", "模型登记", "数据管理", "选号复盘"
])

with page_overview:
    st.subheader("系统状态")
    ball_cols = st.columns(7)
    for idx, number in enumerate(last_nums):
        ball_cols[idx].markdown(
            f"<div style='background:#1769aa;color:white;border-radius:50%;width:42px;height:42px;display:flex;align-items:center;justify-content:center;font-weight:700;margin:auto'>{number:02d}</div>",
            unsafe_allow_html=True,
        )
    ball_cols[6].markdown(
        f"<div style='text-align:center;padding-top:9px'>Bonus<br><b>{int(last['bonus']):02d}</b></div>",
        unsafe_allow_html=True,
    )
    st.markdown("### 当前项目能力")
    overview_cols = st.columns(3)
    overview_cols[0].metric("已注册策略", len(MODEL_NAMES))
    overview_cols[1].metric("每次默认候选组数", 10)
    overview_cols[2].metric("外部数据连接", "未启用")
    st.info("V3.6 第一阶段已拆分独立回测引擎，增加模型版本登记、每注号码留档和回测实验元数据。模型表现只用于统计研究，不代表下一期开奖概率。")

with page_pick:
    st.subheader("生成10组候选号码")
    st.warning("任意一组固定六号码的理论头奖概率相同。模型分数是历史统计权重，不是中奖概率。")
    left, right = st.columns(2)
    with left:
        model = st.selectbox("选号模型", MODEL_NAMES, index=4)
        window_label = st.selectbox("近期数据窗口", ["最近30期", "最近50期", "最近100期", "最近200期", "最近500期", "全部历史"], index=2)
        window = None if window_label == "全部历史" else int(window_label.replace("最近", "").replace("期", ""))
    with right:
        count = st.selectbox("生成组数", [10, 5, 20], index=0)
        overlap = st.slider("任意两组最多共享号码数", 0, 5, 4)
        seed_text = st.text_input("随机种子（留空则每次不同）", "")
    if st.button("生成候选号码", type="primary", use_container_width=True):
        try:
            seed_value = int(seed_text.strip()) if seed_text.strip() else None
            generated, relaxed = generate_candidates(df, count, model, window or len(df), overlap, seed_value)
            st.session_state["v36_candidates"] = generated
            st.session_state["v36_candidate_model"] = model
            st.session_state["v36_candidate_window"] = window_label
            st.session_state["v36_candidate_relaxed"] = relaxed
        except Exception as exc:
            st.error(f"选号失败：{exc}")
    generated = st.session_state.get("v36_candidates")
    if generated is not None:
        st.markdown(f"#### {st.session_state.get('v36_candidate_model')} · {st.session_state.get('v36_candidate_window')}")
        st.dataframe(generated.drop(columns=["号码列表"], errors="ignore"), hide_index=True, use_container_width=True)
        if st.session_state.get("v36_candidate_relaxed"):
            st.info("为生成指定组数，程序已明确放宽跨组重号限制；没有切换选号模型。")
        st.download_button(
            "下载候选号码 CSV",
            generated.drop(columns=["号码列表"], errors="ignore").to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v36_candidates.csv", "text/csv",
        )
    selected_window = len(df) if window is None else min(window, len(df))
    scores = number_score_table(df, selected_window)
    st.markdown("### 多模型号码相对评分")
    st.plotly_chart(
        px.bar(scores.sort_values("号码"), x="号码", y="综合相对得分", title="45个号码的统计相对评分（0—100）"),
        use_container_width=True,
    )
    st.dataframe(scores.head(15).round(3), hide_index=True, use_container_width=True)

with page_stats:
    st.subheader("历史数据统计")
    scope = st.selectbox("统计范围", ["最近30期", "最近50期", "最近100期", "最近200期", "最近500期", "全部历史"], index=2, key="v36_stats_scope")
    n = len(df) if scope == "全部历史" else int(scope.replace("最近", "").replace("期", ""))
    sample = df.tail(min(n, len(df)))
    st.caption(f"分析范围：第{int(sample.iloc[0]['draw_no'])}期至第{int(sample.iloc[-1]['draw_no'])}期，共{len(sample)}期")
    freq = frequency_table(sample)
    gap = gap_table(df)
    a, b = st.columns(2)
    with a:
        chart = px.bar(freq.sort_values("号码"), x="号码", y="出现次数", title="号码出现频率")
        chart.add_hline(y=len(sample) * 6 / 45, line_dash="dash", annotation_text="理论期望")
        st.plotly_chart(chart, use_container_width=True)
    with b:
        st.markdown("**出现次数靠前的号码**")
        st.dataframe(freq.head(12), hide_index=True, use_container_width=True)
        st.markdown("**当前遗漏较长的号码（历史描述）**")
        st.dataframe(gap.head(10), hide_index=True, use_container_width=True)
    features = draw_features(sample)
    c, d = st.columns(2)
    with c:
        st.plotly_chart(px.histogram(features, x="和值", nbins=25, title="历史和值分布"), use_container_width=True)
    with d:
        odd = features["奇数个数"].value_counts().reindex(range(7), fill_value=0).rename_axis("奇数个数").reset_index(name="期数")
        st.plotly_chart(px.bar(odd, x="奇数个数", y="期数", title="奇数号码数量分布"), use_container_width=True)
    st.markdown("**两号码历史共现次数**")
    st.dataframe(pair_table(sample, 30), hide_index=True, use_container_width=True)

with page_backtest:
    st.subheader("滚动样本外回测中心")
    st.write("每个测试期只使用该期之前的历史数据。长期频率、贝叶斯平滑和时间衰减策略使用此前全部可用历史；近期频率使用单独指定的近期窗口。")
    mode = st.selectbox("回测模式", ["快速回测", "标准回测", "自定义回测", "多窗口比较"], index=1)
    valid_windows = [w for w in [50, 100, 150, 200, 300, 500] if w < len(df)]
    if not valid_windows:
        st.warning("至少需要51期有效历史数据才能进行回测。")
        st.stop()
    default_lookback = 200 if 200 in valid_windows else valid_windows[min(0, len(valid_windows) - 1)]
    defaults = {
        "快速回测": (default_lookback, 50, 1, 3),
        "标准回测": (default_lookback, 100, 3, 10),
        "自定义回测": (default_lookback, min(100, max(1, len(df) - default_lookback)), 3, 10),
        "多窗口比较": (default_lookback, min(100, max(1, len(df) - max(valid_windows))), 3, 10),
    }
    default_lookback, default_tests, default_seeds, default_tickets = defaults[mode]
    if mode == "多窗口比较":
        default_multi = [w for w in [100, 200, 300, 500] if w in valid_windows]
        windows = st.multiselect("比较训练窗口", valid_windows, default=default_multi or [valid_windows[-1]])
        max_test = max(1, min(200, len(df) - max(windows or [default_lookback])))
        test_count = st.slider("最近测试期数", 1, max_test, min(default_tests, max_test))
        seeds = st.selectbox("随机种子组数", [1, 3, 5], index=1)
        tickets = st.selectbox("每期每模型模拟注数", [3, 5, 10], index=2)
        lookback = max(windows or [200])
    else:
        x, y, z, q = st.columns(4)
        with x:
            lookback = st.selectbox("最低训练历史期数", valid_windows, index=valid_windows.index(default_lookback))
        with y:
            max_test = max(1, min(200, len(df) - lookback))
            test_count = st.slider("最近测试期数", 1, max_test, min(default_tests, max_test))
        with z:
            seeds = st.selectbox("随机种子组数", [1, 3, 5], index=[1, 3, 5].index(default_seeds))
        with q:
            tickets = st.selectbox("每期每模型模拟注数", [1, 3, 5, 10], index=[1, 3, 5, 10].index(default_tickets))
        windows = []
    seed = st.number_input("基础随机种子", min_value=0, max_value=2_000_000_000, value=2026, step=1)
    if st.button("运行回测", type="primary", use_container_width=True):
        progress = st.progress(0, text="准备回测…")
        status = st.empty()
        try:
            def progress_update(done: int, total: int):
                progress.progress(min(1.0, done / max(total, 1)), text=f"回测进度：{done:,}/{total:,} 个策略批次")
            if mode == "多窗口比较":
                if not windows:
                    raise ValueError("请至少选择一个比较窗口。")
                raw, summary, metadata = run_multi_window_backtest(
                    df, windows=windows, test_draws=test_count, tickets_per_draw=tickets,
                    n_seeds=seeds, seed=int(seed), data_source_label=data_source, progress_callback=progress_update,
                )
                st.session_state["v36_bt_metadata"] = metadata
            else:
                raw, summary, metadata = run_backtest(
                    df, lookback=lookback, test_draws=test_count, tickets_per_draw=tickets,
                    n_seeds=seeds, seed=int(seed), recent_window=lookback,
                    data_source_label=data_source, progress_callback=progress_update,
                )
                st.session_state["v36_bt_metadata"] = metadata
            st.session_state["v36_bt_raw"] = raw
            st.session_state["v36_bt_summary"] = summary
            status.success(f"回测完成：{len(raw):,}条明细，耗时 {metadata.get('elapsed_seconds', 0):.2f} 秒。") if isinstance(metadata, dict) else status.success(f"多窗口回测完成：{len(raw):,}条明细。")
        except Exception as exc:
            st.error(f"回测失败：{type(exc).__name__}: {exc}")
    raw = st.session_state.get("v36_bt_raw")
    summary = st.session_state.get("v36_bt_summary")
    metadata = st.session_state.get("v36_bt_metadata")
    if isinstance(raw, pd.DataFrame) and not raw.empty and isinstance(summary, pd.DataFrame) and not summary.empty:
        st.markdown("### 策略汇总")
        view = summary.copy()
        percent_cols = ["3个以上命中率", "4个以上命中率", "5个以上命中率", "6个命中率", "每批至少一注3+比例"]
        for col in percent_cols:
            if col in view.columns:
                view[col] = (view[col] * 100).round(3).astype(str) + "%"
        numeric_cols = [c for c in view.columns if "命中" in c or "覆盖" in c or "重号" in c]
        for col in numeric_cols:
            if col in view.columns and pd.api.types.is_numeric_dtype(view[col]):
                view[col] = view[col].round(4)
        st.dataframe(view, hide_index=True, use_container_width=True)
        chart_df = summary.copy()
        st.plotly_chart(
            px.bar(chart_df, x="策略", y="平均命中数", color="比较训练窗口" if "比较训练窗口" in chart_df.columns else None,
                   barmode="group", title="各策略平均命中数"),
            use_container_width=True,
        )
        st.info("统计单位为开奖期：多个随机种子先在每期内汇总，再进行配对Bootstrap和符号翻转检验；p值另做Holm校正。置信区间跨过0或校正后证据不足时，不应宣称模型优于随机基准。多窗口结果仍属于探索性比较。")
        c1, c2, c3 = st.columns(3)
        c1.download_button("下载逐注明细（含号码）", raw.to_csv(index=False).encode("utf-8-sig"), "lotto645_v36_backtest_detail.csv", "text/csv")
        c2.download_button("下载模型汇总", summary.to_csv(index=False).encode("utf-8-sig"), "lotto645_v36_backtest_summary.csv", "text/csv")
        if isinstance(metadata, list):
            metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2)
        else:
            metadata_text = json.dumps(metadata or {}, ensure_ascii=False, indent=2)
        c3.download_button("下载实验元数据", metadata_text, "lotto645_v36_experiment_metadata.json", "application/json")
        if isinstance(metadata, dict):
            with st.expander("查看实验参数与数据范围"):
                st.json(metadata)
        else:
            with st.expander("查看各窗口实验参数"):
                st.json(metadata)

with page_models:
    st.subheader("模型登记与版本")
    registry = pd.DataFrame(get_model_registry())
    st.dataframe(registry, hide_index=True, use_container_width=True)
    st.info("第一阶段将现有七种策略登记为实验模型或随机对照基准。当前没有任何统计模型被自动标记为已验证优胜模型。")

with page_data:
    st.subheader("手动数据管理")
    st.write("本项目只分析你手动提供的历史开奖文件，不会访问外部开奖网站、API 或数据抓取服务。")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("有效记录", quality["rows"])
    q2.metric("无效/被跳过记录", history_report.get("invalid_rows", quality["invalid_rows"]))
    q3.metric("缺失期号", len(quality["missing_draws"]))
    q4.metric("日期异常", quality["date_errors"])
    st.write({
        "数据来源模式": history_report.get("source_mode", data_source),
        "文件名": history_report.get("source_filename", HISTORY_PATH.name),
        "首期": quality["first_draw"],
        "末期": quality["last_draw"],
        "当前文件覆盖范围内连续": quality["range_complete"],
        "是否从第1期开始": quality["sequence_complete"],
        "文件开始前未提供的期数": quality["missing_before_first"],
    })
    if quality["missing_draws"]:
        st.warning(f"文件覆盖范围内缺失期号示例：{quality['missing_draws'][:40]}")
    st.dataframe(df.tail(20), hide_index=True, use_container_width=True)
    st.download_button("下载当前清洗后的历史数据", df.to_csv(index=False).encode("utf-8-sig"), "lotto645_clean_history.csv", "text/csv")
    st.download_button("下载空白数据模板", pd.DataFrame(columns=COLUMNS).to_csv(index=False).encode("utf-8-sig"), "lotto645_history_template.csv", "text/csv")

with page_review:
    st.subheader("自选号码历史复盘")
    raw_ticket = st.text_input("输入6个主号码（逗号或空格分隔）", placeholder="例如：3, 11, 18, 27, 34, 42")
    if st.button("复盘这组号码"):
        try:
            values = [int(x) for x in raw_ticket.replace("，", ",").replace(" ", ",").split(",") if x.strip()]
            result = compare_ticket_to_history(df, values)
            st.write({"检验期数": result["draws_checked"], "历史最高命中": result["max_hits"], "平均命中": round(result["mean_hits"], 4)})
            dist = pd.DataFrame([{"命中主号码数": k, "历史期数": v} for k, v in result["hit_distribution"].items()])
            st.dataframe(dist, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error(f"复盘失败：{exc}")

metrics = theoretical_random_metrics()
st.divider()
st.caption(
    f"理论随机基准：每注平均命中主号码数 {metrics['mean_hits_per_ticket']:.1f}；"
    f"命中至少3个主号码的概率 {metrics['p3plus']:.3%}；"
    f"单注头奖概率 1/{int(1 / metrics['jackpot_probability']):,}。"
)
st.caption("统计说明：历史频率、遗漏、共现和结构约束仅描述历史样本；公平独立开奖下，任何固定六号码组合的理论概率相同。")
