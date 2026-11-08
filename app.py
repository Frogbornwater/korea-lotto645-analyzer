# -*- coding: utf-8 -*-
"""Lotto 6/45 V5.0 predictive research console with V4.0 governance - offline/manual-data application."""
from __future__ import annotations

import io
import json
import inspect
import sys
from pathlib import Path

# Always prioritize the app's own project directory before importing local modules.
APP_DIR = Path(__file__).resolve().parent
_APP_ROOT = str(APP_DIR)
if _APP_ROOT in sys.path:
    sys.path.remove(_APP_ROOT)
sys.path.insert(0, _APP_ROOT)
for _module_name in (
    "backtest_engine", "joint_optimizer", "robust_optimizer", "parameter_optimizer",
    "portfolio_optimizer", "lotto_core", "model_registry", "model_management",
    "experiment_export", "data_manager", "v40_governance", "v40_store", "v50_predictive_engine", "v50_backtest",
):
    sys.modules.pop(_module_name, None)

import pandas as pd
import plotly.express as px
import streamlit as st

import backtest_engine as _runtime_backtest_engine
import joint_optimizer as _runtime_joint_optimizer
import robust_optimizer as _runtime_robust_optimizer

from backtest_engine import run_backtest, run_multi_window_backtest
from parameter_optimizer import (
    SEARCHABLE_MODELS, evaluate_reserved_holdout, run_parameter_search,
)
from portfolio_optimizer import optimize_portfolio
from joint_optimizer import run_joint_optimization
from robust_optimizer import (
    V38_RETROSPECTIVE_CUTOFF_DRAW,
    build_forward_lock,
    evaluate_forward_locks,
    restore_forward_locks_from_bundle,
    run_robust_research,
)
from model_management import (
    MODEL_STATUSES, add_experiment, change_model_status, export_project_state,
    import_project_state, new_project_state,
)
from experiment_export import build_research_bundle
from data_manager import load_repository_history, read_manual_upload
from lotto_core import (
    COLUMNS, MODEL_NAMES, compare_ticket_to_history, draw_features, frequency_table,
    gap_table, generate_candidates, number_score_table, pair_table, quality_report,
    theoretical_random_metrics, validate_frame,
)
from model_registry import ENGINE_VERSION
from v40_governance import (
    append_draw as v40_append_draw,
    data_fingerprint as v40_data_fingerprint,
    make_event as v40_make_event,
    evaluate_forward_pairs as v40_evaluate_forward_pairs,
)
from v50_predictive_engine import fit_predict_next as v50_fit_predict_next, generate_tickets as v50_generate_tickets, build_v50_forward_lock
from v50_backtest import run_v50_walk_forward
from v40_store import (
    EventStoreError, GitHubEventStore, export_event_archive,
    import_event_archive, latest_model_snapshot, events_to_forward_locks,
)


_RUNTIME_MODULE_DIAGNOSTICS = {
    "app.py": str(Path(__file__).resolve()),
    "APP_DIR": str(APP_DIR),
    "backtest_engine.__file__": str(Path(_runtime_backtest_engine.__file__).resolve()),
    "joint_optimizer.__file__": str(Path(_runtime_joint_optimizer.__file__).resolve()),
    "robust_optimizer.__file__": str(Path(_runtime_robust_optimizer.__file__).resolve()),
    "backtest_engine._make_weights": str(inspect.signature(_runtime_backtest_engine._make_weights)),
    "joint_optimizer._make_weights": str(inspect.signature(_runtime_joint_optimizer._make_weights)),
    "same_function_object": _runtime_joint_optimizer._make_weights is _runtime_backtest_engine._make_weights,
    "11_parameter_contract_ok": len(inspect.signature(_runtime_backtest_engine._make_weights).parameters) == 11,
}
DATA_DIR = APP_DIR / "data"
HISTORY_PATH = DATA_DIR / "lotto645.csv"
EXPERIMENT_LEDGER_PATH = DATA_DIR / "experiment_ledger.json"
try:
    _EXPERIMENT_LEDGER = json.loads(EXPERIMENT_LEDGER_PATH.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError, OSError):
    _EXPERIMENT_LEDGER = {"experiments": [], "known_reviewed_ranges": []}

st.set_page_config(page_title="Lotto 6/45 V5.0 · 预测研究平台", page_icon="🎯", layout="wide")
st.title("🎯 韩国 Lotto 6/45 · V5.0 预测研究平台")
st.caption("V5联合概率预测｜V4研究治理兼容｜手动历史数据导入｜离线滚动回测")

if "v40_events" not in st.session_state:
    st.session_state["v40_events"] = []

def _v40_add_event(event: dict) -> None:
    """Append an auditable event and optionally persist it on the research branch."""
    events = list(st.session_state.get("v40_events", []))
    if any(item.get("event_id") == event.get("event_id") for item in events):
        return
    events.append(event)
    st.session_state["v40_events"] = events
    try:
        token = st.secrets.get("GITHUB_RESEARCH_TOKEN", "")
        if token:
            store = GitHubEventStore(
                "Frogbornwater/korea-lotto645-analyzer", str(token), branch="research-data"
            )
            store.append_event(event)
    except Exception as exc:
        st.warning(f"实验事件暂存在当前会话；GitHub研究档案同步失败：{exc}")


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
    elif isinstance(st.session_state.get("v40_history_override"), pd.DataFrame):
        df = validate_frame(st.session_state["v40_history_override"])
        history_report = quality_report(df)
        history_report.update({"source_mode": "manual_append_session", "source_filename": "V4.0手动追加"})
        data_source = "V4.0 当前会话手动追加数据"
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
st.caption(f"数据来源：{data_source}｜治理底座版本：{ENGINE_VERSION}｜数据仅由用户手动提供")

if "v36_project_state" not in st.session_state:
    st.session_state["v36_project_state"] = new_project_state()

page_overview, page_pick, page_stats, page_backtest, page_optimizer, page_joint, page_robust, page_ensemble, page_portfolio, page_models, page_data, page_review, page_v40 = st.tabs([
    "系统总览", "智能选号", "历史统计", "回测中心", "单模型参数实验", "V3.7联合优化", "V3.8稳健研究", "模型集成", "组合优化", "模型管理", "数据管理", "选号复盘", "V4.0研究治理"
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
    st.info("V3.8 将预测表现与10注组合结构分开评价，并采用空间均衡参数搜索、多窗口嵌套滚动验证和优胜候选多种子复核。")
    with st.expander("运行环境诊断（模块路径与函数签名）"):
        st.json(_RUNTIME_MODULE_DIAGNOSTICS)

with page_pick:
    st.subheader("生成10组候选号码")
    st.warning("任意一组固定六号码的理论头奖概率相同。模型分数是历史统计权重，不是中奖概率。")
    left, right = st.columns(2)
    with left:
        model = st.selectbox("选号模型", MODEL_NAMES, index=4)
        window_label = st.selectbox("近期数据窗口", ["最近30期", "最近50期", "最近100期", "最近200期", "最近500期", "全部历史"], index=2)
        window = None if window_label == "全部历史" else int(window_label.replace("最近", "").replace("期", ""))
    with right:
        count = 10
        st.metric("本次候选组合数量", "10组")
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
            state_now = st.session_state.get("v36_project_state", new_project_state())
            st.session_state["v36_project_state"] = add_experiment(
                state_now, "常规回测", metadata if isinstance(metadata, dict) else {"experiment_type": "multi_window_backtest", "data_first_draw": int(df["draw_no"].min()), "data_last_draw": int(df["draw_no"].max())},
                summary.to_dict(orient="records"),
            )
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

with page_optimizer:
    st.subheader("自动参数优化实验室")
    st.warning("参数排名属于探索性搜索。最新保留区间不会参与参数搜索；搜索结果排名第一不等于显著优于随机。")
    model_target = st.selectbox("要优化的模型", SEARCHABLE_MODELS, index=0, key="v361_param_model")
    valid_train = [w for w in [50, 100, 150, 200, 300, 500] if w + 12 < len(df)]
    if not valid_train:
        st.info("历史期数不足，暂时无法运行参数实验。")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            min_train = st.selectbox("最低训练期数", valid_train, index=valid_train.index(200) if 200 in valid_train else len(valid_train)-1, key="v361_min_train")
        with c2:
            max_holdout = max(10, min(100, len(df) - min_train - 3))
            holdout_n = st.slider("保留验证期数", 10, max_holdout, min(40, max_holdout), key="v361_holdout")
        with c3:
            max_validation = max(2, min(100, len(df) - min_train - holdout_n - 1))
            validation_n = st.slider("参数搜索验证期数", 2, max_validation, min(40, max_validation), key="v361_validation")
        with c4:
            search_seeds = st.selectbox("每期随机种子组", [1, 2, 3], index=1, key="v361_search_seeds")
        s1, s2 = st.columns(2)
        with s1:
            search_tickets = st.selectbox("每期每模型模拟注数", [3, 5, 10], index=1, key="v361_search_tickets")
        with s2:
            search_seed = st.number_input("参数搜索基础种子", min_value=0, max_value=2_000_000_000, value=2026, step=1, key="v361_search_seed")
        if st.button("开始参数搜索", type="primary", use_container_width=True, key="v361_run_search"):
            bar = st.progress(0, text="准备参数实验…")
            try:
                def search_progress(done: int, total: int):
                    bar.progress(min(1.0, done / max(total, 1)), text=f"参数实验进度：{done:,}/{total:,}")
                result_detail, result_table, result_meta = run_parameter_search(
                    df, model_target, min_train=min_train, validation_draws=validation_n,
                    holdout_draws=holdout_n, tickets_per_draw=search_tickets,
                    n_seeds=search_seeds, seed=int(search_seed), progress_callback=search_progress,
                )
                st.session_state["v361_param_detail"] = result_detail
                st.session_state["v361_param_summary"] = result_table
                st.session_state["v361_param_meta"] = result_meta
                st.session_state["v361_param_context"] = {
                    "model": model_target, "min_train": min_train, "holdout": holdout_n,
                    "tickets": search_tickets, "seeds": search_seeds, "seed": int(search_seed),
                }
                compact = result_table.head(10).to_dict(orient="records")
                st.session_state["v36_project_state"] = add_experiment(
                    st.session_state["v36_project_state"], "参数搜索", result_meta, compact,
                )
                st.success(f"参数搜索完成：{len(result_table)}组参数，耗时{result_meta['elapsed_seconds']:.2f}秒。")
            except Exception as exc:
                st.error(f"参数搜索失败：{type(exc).__name__}: {exc}")
        param_summary = st.session_state.get("v361_param_summary")
        param_meta = st.session_state.get("v361_param_meta")
        param_context = st.session_state.get("v361_param_context")
        if isinstance(param_summary, pd.DataFrame) and not param_summary.empty:
            st.markdown("### 参数排名（探索性）")
            st.dataframe(param_summary, hide_index=True, use_container_width=True)
            st.markdown("**当前排名第一的参数组合**")
            st.json(param_meta.get("best_config", {}))
            st.download_button("下载参数搜索排名 CSV", param_summary.to_csv(index=False).encode("utf-8-sig"), "lotto645_v361_parameter_search.csv", "text/csv")
            param_detail = st.session_state.get("v361_param_detail")
            if isinstance(param_detail, pd.DataFrame) and not param_detail.empty:
                st.download_button("下载逐期参数实验明细 CSV", param_detail.to_csv(index=False).encode("utf-8-sig"), "lotto645_v361_parameter_search_detail.csv", "text/csv")
            st.download_button("下载参数实验元数据 JSON", json.dumps(param_meta, ensure_ascii=False, indent=2), "lotto645_v361_parameter_metadata.json", "application/json")
            st.info(f"保留区间：第{param_meta['reserved_holdout_start_draw']}期至第{param_meta['reserved_holdout_end_draw']}期，共{param_meta['reserved_holdout_draws']}期。参数搜索没有读取该区间。")
            st.warning("查看保留区间验证结果后，该区间即被消耗；不要再反复调整参数并把同一批数据当作未触碰测试集。")
            project_experiments = st.session_state.get("v36_project_state", {}).get("experiments", [])
            same_holdout_was_consumed = any(
                item.get("holdout_evaluated")
                and item.get("data_fingerprint") == param_meta.get("data_fingerprint")
                and item.get("reserved_holdout_start_draw") == param_meta.get("reserved_holdout_start_draw")
                and item.get("reserved_holdout_end_draw") == param_meta.get("reserved_holdout_end_draw")
                for item in project_experiments
            )
            holdout_already_used = bool(param_meta.get("holdout_evaluated", False) or same_holdout_was_consumed)
            if holdout_already_used:
                st.success("这组参数的保留区间已经评估过；为避免反复查看同一测试段后继续调参，按钮已锁定。")
            elif st.button("用排名第一的固定参数进行保留区间验证", type="secondary", use_container_width=True, key="v361_evaluate_holdout"):
                try:
                    cfg = param_meta["best_config"]
                    bar = st.progress(0, text="保留区间验证中…")
                    def holdout_progress(done: int, total: int):
                        bar.progress(min(1.0, done / max(total, 1)), text=f"保留区间验证：{done:,}/{total:,}")
                    raw_hold, summary_hold, meta_hold = evaluate_reserved_holdout(
                        df, param_context["model"], cfg, param_context["min_train"],
                        param_context["holdout"], tickets_per_draw=param_context["tickets"],
                        n_seeds=param_context["seeds"], seed=param_context["seed"],
                        progress_callback=holdout_progress,
                    )
                    st.session_state["v361_holdout_raw"] = raw_hold
                    st.session_state["v361_holdout_summary"] = summary_hold
                    st.session_state["v361_holdout_meta"] = meta_hold
                    st.session_state["v361_param_meta"]["holdout_evaluated"] = True
                    st.session_state["v361_param_meta"]["holdout_consumed"] = True
                    st.session_state["v36_project_state"] = add_experiment(
                        st.session_state["v36_project_state"], "固定参数保留区间验证", meta_hold,
                        summary_hold.to_dict(orient="records"),
                    )
                    st.success("固定参数保留区间验证完成。该保留区间已记为已使用。")
                except Exception as exc:
                    st.error(f"保留区间验证失败：{type(exc).__name__}: {exc}")
        hold_sum = st.session_state.get("v361_holdout_summary")
        hold_raw = st.session_state.get("v361_holdout_raw")
        hold_meta = st.session_state.get("v361_holdout_meta")
        if isinstance(hold_sum, pd.DataFrame) and not hold_sum.empty:
            st.markdown("### 固定参数保留区间结果")
            st.dataframe(hold_sum, hide_index=True, use_container_width=True)
            st.download_button("下载保留区间逐注明细", hold_raw.to_csv(index=False).encode("utf-8-sig"), "lotto645_v361_holdout_detail.csv", "text/csv")
            st.download_button("下载保留区间元数据", json.dumps(hold_meta, ensure_ascii=False, indent=2), "lotto645_v361_holdout_metadata.json", "application/json")

with page_joint:
    st.subheader("V3.7 历史联合优化记录")
    st.markdown(
        "一次运行覆盖全部8种策略：7种可调模型共搜索500组唯一参数配置；"
        "均匀随机基准作为所有内层、外层和最终保留测试的共同配对对照，不参与参数调优。"
    )
    st.warning(
        "本实验会消耗最新独立保留区间。最终配置在开发数据上冻结后，保留集只评估一次；"
        "查看结果后不得继续用同一保留集反复调参。公平独立开奖下，模型历史表现不改变固定号码组合的理论概率。"
    )
    import hashlib as _hashlib
    _joint_fingerprint = _hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()
    _project_state = st.session_state.get("v36_project_state", new_project_state())
    _already_consumed = any(
        item.get("experiment_type") == "V3.7全模型联合优化"
        and item.get("data_fingerprint") == _joint_fingerprint
        for item in _project_state.get("experiments", [])
    )
    # The repository ledger survives Streamlit session resets. Protect the
    # exact V3.7 data fingerprint and its previously inspected test range.
    _ledger_v37_records = [
        item for item in _EXPERIMENT_LEDGER.get("experiments", [])
        if item.get("experiment_id") == "lotto645-v37-joint-20261107T083833Z-2026"
        and item.get("holdout_consumed")
    ]
    _ledger_consumed = any(
        item.get("data_fingerprint_sha256") == _joint_fingerprint
        or (
            int(item.get("data_first_draw", -1)) == int(quality["first_draw"])
            and int(item.get("data_last_draw", -1)) == int(quality["last_draw"])
            and int(item.get("data_rows", -1)) == int(quality["rows"])
        )
        for item in _ledger_v37_records
    )
    _existing_result = st.session_state.get("v37_joint_result")
    _same_session_result = (
        isinstance(_existing_result, dict)
        and _existing_result.get("metadata", {}).get("data_fingerprint_sha256") == _joint_fingerprint
    )

    cfg_cols = st.columns(4)
    cfg_cols[0].metric("模型覆盖", "8种（含随机基准）")
    cfg_cols[1].metric("唯一参数预算", "500组")
    cfg_cols[2].metric("每期每模型", "10注")
    cfg_cols[3].metric("随机种子", "3组")
    st.markdown("### 验证结构")
    st.write({
        "优化目标": "balanced（六项目标等权百分位排名）",
        "最低训练历史": "300期",
        "外层滚动验证": "2折 × 每折30期",
        "内层滚动验证": "每个外层训练前缀2折 × 每折5期",
        "独立保留测试集": "最新100期",
        "参数预算口径": "500组唯一参数配置在各折间复用；不是每折额外生成500组。",
    })
    st.caption(
        "Pareto目标：相对随机的平均命中差、单注3+率差、整批3+差、号码覆盖差、组间重号减少，以及内层折间稳定性。"
    )
    _min_required = 300 + 2 * 5 + 2 * 30 + 100
    if len(df) < _min_required:
        st.error(f"当前只有{len(df)}期。按默认嵌套结构至少需要{_min_required}期连续历史记录。请手动补齐 CSV/Excel 后再运行。")
    elif _already_consumed or _same_session_result or _ledger_consumed:
        st.success("这份历史数据的 V3.7 独立保留区间已经使用过。仓库实验台账也已记录本次使用，不能重复运行。")
        st.caption("请先下载本次研究结果并归档；只有加入新的手动开奖数据、形成新的数据指纹后，才可执行下一次联合实验。")
    else:
        st.info("正式运行参数已经按你的要求固定：500组配置、10注、3组随机种子、balanced目标。最低训练300期、外层2×30期、内层2×5期、独立保留100期。")
        _run_joint = st.button("启动全模型联合优化（500组）", type="primary", use_container_width=True, key="v37_run_joint")
        if _run_joint:
            _progress = st.progress(0, text="初始化500组联合参数配置…")
            _status = st.empty()
            try:
                def _joint_progress(done: int, total: int, label: str):
                    _progress.progress(min(1.0, done / max(total, 1)), text=f"{label}：{done:,}/{total:,}")
                    _status.caption(label)

                _result = run_joint_optimization(
                    df, budget=500, tickets_per_draw=10, n_seeds=3,
                    minimum_train=300, holdout_draws=100,
                    outer_folds=2, outer_validation_draws=30,
                    inner_folds=2, inner_validation_draws=5,
                    seed=2026, objective="balanced", progress_callback=_joint_progress,
                )
                st.session_state["v37_joint_result"] = _result
                _metadata_for_archive = dict(_result["metadata"])
                _metadata_for_archive["data_fingerprint"] = _metadata_for_archive.get("data_fingerprint_sha256")
                _metadata_for_archive["holdout_evaluated"] = True
                _metadata_for_archive["reserved_holdout_start_draw"] = _metadata_for_archive.get("holdout_start_draw")
                _metadata_for_archive["reserved_holdout_end_draw"] = _metadata_for_archive.get("holdout_end_draw")
                # It is a multi-model experiment, not evidence that any single
                # model has passed verification; intentionally omit model_name.
                st.session_state["v36_project_state"] = add_experiment(
                    st.session_state.get("v36_project_state", new_project_state()),
                    "V3.7全模型联合优化", _metadata_for_archive,
                    _result["search_summary"].head(20).to_dict(orient="records"),
                )
                st.success(
                    f"联合优化完成：{len(_result['search_summary']) - 1}组可调配置，"
                    f"Pareto点{len(_result['pareto_front'])}个；耗时{_result['metadata']['elapsed_seconds']:.1f}秒。"
                )
            except Exception as exc:
                st.error(f"全模型联合优化失败：{type(exc).__name__}: {exc}")

    _joint_result = st.session_state.get("v37_joint_result")
    if isinstance(_joint_result, dict) and _joint_result.get("metadata", {}).get("data_fingerprint_sha256") == _joint_fingerprint:
        _jm = _joint_result["metadata"]
        _chosen = _joint_result["selected_config"]
        st.divider()
        st.markdown("### 最终冻结配置")
        _kpi = st.columns(4)
        _kpi[0].metric("选中模型", _chosen["model_name"])
        _kpi[1].metric("唯一配置预算", _jm["unique_configuration_budget"])
        _kpi[2].metric("Pareto前沿点", len(_joint_result["pareto_front"]))
        _kpi[3].metric("独立测试期数", _jm["holdout_draws"])
        st.json(_chosen)

        st.markdown("### 独立保留测试：冻结模型与随机基准")
        st.dataframe(_joint_result["holdout_summary"], hide_index=True, use_container_width=True)
        st.dataframe(_joint_result["holdout_detail"].head(20), hide_index=True, use_container_width=True)
        st.markdown("### 外层嵌套滚动验证")
        st.dataframe(_joint_result["outer_selections"], hide_index=True, use_container_width=True)
        st.dataframe(_joint_result["outer_summary"], hide_index=True, use_container_width=True)

        st.markdown("### Pareto候选方案")
        st.dataframe(_joint_result["pareto_front"].head(100), hide_index=True, use_container_width=True)
        _pareto_plot = _joint_result["pareto_front"].copy()
        if not _pareto_plot.empty:
            st.plotly_chart(
                px.scatter(
                    _pareto_plot, x="平均命中差_vs随机", y="整批3+差_vs随机",
                    color="模型", hover_data=["配置编号", "balanced_score", "是否Pareto前沿"],
                    title="Pareto前沿：平均命中差与整批3+差（相对随机）",
                ),
                use_container_width=True,
            )
        st.markdown("### 500组联合搜索排名")
        st.dataframe(_joint_result["search_summary"].head(100), hide_index=True, use_container_width=True)

        st.markdown("### 下载研究结果")
        _d1, _d2, _d3 = st.columns(3)
        _d1.download_button(
            "下载500组参数排名 CSV",
            _joint_result["search_summary"].to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v37_joint_search_ranking.csv", "text/csv",
        )
        _d2.download_button(
            "下载Pareto前沿 CSV",
            _joint_result["pareto_front"].to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v37_pareto_front.csv", "text/csv",
        )
        _d3.download_button(
            "下载内层逐期评估 CSV",
            _joint_result["inner_period_detail"].to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v37_inner_period_detail.csv", "text/csv",
        )
        _d4, _d5, _d6 = st.columns(3)
        _d4.download_button(
            "下载内层折汇总 CSV",
            _joint_result["inner_fold_detail"].to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v37_inner_fold_detail.csv", "text/csv",
        )
        _d5.download_button(
            "下载外层嵌套验证 CSV",
            _joint_result["outer_detail"].to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v37_outer_validation.csv", "text/csv",
        )
        _d6.download_button(
            "下载外层选型记录 CSV",
            _joint_result["outer_selections"].to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v37_outer_selections.csv", "text/csv",
        )
        _d7, _d8, _d9 = st.columns(3)
        _d7.download_button(
            "下载保留集逐注号码 CSV",
            _joint_result["holdout_ticket_detail"].to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v37_holdout_tickets.csv", "text/csv",
        )
        _d8.download_button(
            "下载保留集配对检验 CSV",
            _joint_result["holdout_summary"].to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v37_holdout_statistics.csv", "text/csv",
        )
        _d9.download_button(
            "下载完整实验元数据 JSON",
            json.dumps(_jm, ensure_ascii=False, indent=2),
            "lotto645_v37_joint_metadata.json", "application/json",
        )
        st.download_button(
            "下载模型实验档案（用于恢复保留集使用记录）",
            export_project_state(st.session_state.get("v36_project_state", new_project_state())),
            "lotto645_v37_model_archive.json", "application/json",
        )
        st.warning("最终保留集已经被这次冻结配置使用。不要根据它反复调整参数后再次运行；请保存模型实验档案。若 Streamlit 会话重置，先在‘模型管理’页导入该档案，以继续识别已使用的保留区间。")


with page_robust:
    st.subheader("V3.8 稳健模型研究引擎")
    st.markdown(
        "V3.8 不再把命中表现与组合多样性混为一个总分。系统先对7种可调模型进行500组空间均衡参数初筛，"
        "再对每个模型的优胜候选进行3组随机种子复核。预测目标和组合结构目标分别排名。"
    )
    st.warning(
        "V3.7 已经审阅第1089—1248期的外层验证和独立保留测试。为避免重复使用已看过的结果，"
        "V3.8 的回溯评价默认只使用第1—1088期；这不是新的独立保留测试。真正的前瞻性证据必须在开奖前锁定号码，"
        "随后再手动导入目标期开奖号码。"
    )
    st.write({
        "参数配置预算": "500组（7种可调模型均衡分配）",
        "参数空间设计": "每模型 Latin-hypercube 空间均衡采样",
        "初筛随机种子": "1组（降低计算成本）",
        "每模型优胜候选": "1组进入复核",
        "复核随机种子": "3组",
        "内层滚动窗口": "3个分散窗口 × 每窗口6期",
        "外层滚动验证": "3折 × 每折12期",
        "预测评价": "平均命中差、单注3+率差、整批3+差、跨窗口稳定性",
        "组合结构评价": "号码覆盖差、组间重号减少（独立报告）",
        "独立保留测试": "本次不重复使用已审阅区间；等待未来新期数开展前瞻验证",
    })
    _v38_cutoff = min(int(V38_RETROSPECTIVE_CUTOFF_DRAW), int(quality["last_draw"]))
    _v38_safe = df[df["draw_no"] <= _v38_cutoff]
    _v38_minimum = 300 + max(3 * 4, 3 * 6) + 3 * 12
    st.caption(
        f"本次安全回溯数据：第{quality['first_draw']}—{_v38_cutoff}期；"
        f"共{len(_v38_safe)}期。第1089—1248期不作为本次V3.8回溯验证标签。"
    )
    if len(_v38_safe) < _v38_minimum:
        st.error(f"安全回溯区间至少需要{_v38_minimum}期连续记录；目前只有{len(_v38_safe)}期。")
    else:
        if st.button("启动 V3.8 稳健研究（500组）", type="primary", use_container_width=True, key="v38_run_robust"):
            _v38_progress = st.progress(0, text="生成500组空间均衡参数…")
            _v38_status = st.empty()
            try:
                def _v38_progress_callback(done: int, total: int, label: str):
                    _v38_progress.progress(min(1.0, done / max(total, 1)), text=f"{label}：{done:,}/{total:,}")
                    _v38_status.caption(label)

                _v38_result = run_robust_research(
                    df, budget=500, tickets_per_draw=10, minimum_train=300,
                    outer_folds=3, outer_validation_draws=12,
                    inner_folds=3, inner_validation_draws=6,
                    finalists_per_model=1, screening_folds=3, screening_window_draws=4,
                    screening_seeds=1, finalist_seeds=3,
                    seed=2028, retrospective_cutoff_draw=_v38_cutoff,
                    progress_callback=_v38_progress_callback,
                )
                st.session_state["v38_robust_result"] = _v38_result
                _v38_meta = dict(_v38_result["metadata"])
                _selection_fp = _v38_meta.get("retrospective_data_fingerprint_sha256")
                if _selection_fp:
                    _v40_event = v40_make_event("model_snapshot", _selection_fp, {
                        "selected_config": _v38_result["selected_config"],
                        "selection_data_fingerprint": _selection_fp,
                        "selection_end_draw": _v38_meta.get("retrospective_validation_end_draw"),
                        "research_experiment_id": _v38_meta.get("experiment_id"),
                        "engine_version": _v38_meta.get("engine_version"),
                        "predictive_advantage_status": _v38_meta.get("predictive_advantage_status"),
                        "metadata": _v38_meta,
                    })
                    _v40_add_event(_v40_event)
                _v38_meta["data_fingerprint"] = _v38_meta.get("full_data_fingerprint_sha256")
                _v38_meta["holdout_evaluated"] = False
                st.session_state["v36_project_state"] = add_experiment(
                    st.session_state.get("v36_project_state", new_project_state()),
                    "V3.8多窗口稳健研究", _v38_meta,
                    _v38_result["finalist_summary"].head(20).to_dict(orient="records"),
                )
                st.success(
                    f"V3.8回溯研究完成：500组初筛；每模型优胜候选3种子复核；"
                    f"耗时{_v38_result['metadata']['elapsed_seconds']:.1f}秒。"
                )
            except Exception as exc:
                st.error(f"V3.8稳健研究失败：{type(exc).__name__}: {exc}")

    _v38_result = st.session_state.get("v38_robust_result")
    _v38_selection_history = df[df["draw_no"] <= int(V38_RETROSPECTIVE_CUTOFF_DRAW)]
    _v38_current_selection_fingerprint = (
        v40_data_fingerprint(_v38_selection_history) if not _v38_selection_history.empty else ""
    )
    _v38_saved_selection_fingerprint = (
        _v38_result.get("metadata", {}).get("retrospective_data_fingerprint_sha256")
        if isinstance(_v38_result, dict) else None
    )
    # Freeze against the selection slice, not the ever-growing full history.
    _v38_result_matches_current = (
        isinstance(_v38_result, dict)
        and (
            _v38_saved_selection_fingerprint == _v38_current_selection_fingerprint
            if _v38_saved_selection_fingerprint
            else _v38_result.get("metadata", {}).get("full_data_fingerprint_sha256")
                 == _hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()
        )
    )
    if isinstance(_v38_result, dict) and not _v38_result_matches_current:
        st.warning("当前手动历史数据与本次 V3.8 研究使用的数据版本不同。旧研究结果仅供归档查看，不会用于当前数据生成新的锁定号码。")
    if isinstance(_v38_result, dict):
        _v38_meta = _v38_result["metadata"]
        _v38_selected = _v38_result["selected_config"]
        st.divider()
        st.markdown("### 研究结论与冻结候选")
        st.warning(_v38_meta["predictive_advantage_status"])
        _v38_kpi = st.columns(4)
        _v38_kpi[0].metric("初筛配置", _v38_meta["unique_configuration_budget"])
        _v38_kpi[1].metric("每模型优胜复核", f"{_v38_meta['finalists_per_model']}组")
        _v38_kpi[2].metric("外层验证期数", _v38_meta["outer_folds"] * _v38_meta["outer_validation_draws_per_fold"])
        _v38_kpi[3].metric("是否使用新保留集", "否")
        st.write("预测表现候选（预测综合分不包含号码覆盖与组间重号）：")
        st.json(_v38_selected)
        st.markdown("### 外层嵌套滚动验证")
        st.dataframe(_v38_result["outer_selections"], hide_index=True, use_container_width=True)
        st.dataframe(_v38_result["outer_summary"], hide_index=True, use_container_width=True)
        st.markdown("### 优胜候选：预测表现排名")
        st.dataframe(_v38_result["finalist_summary"].head(50), hide_index=True, use_container_width=True)
        st.markdown("### 预测表现 Pareto 前沿")
        st.dataframe(_v38_result["predictive_pareto_front"], hide_index=True, use_container_width=True)
        st.markdown("### 组合结构 Pareto 前沿（不代表预测优势）")
        st.dataframe(_v38_result["structure_pareto_front"], hide_index=True, use_container_width=True)
        _v38_downloads = st.columns(3)
        _v38_downloads[0].download_button(
            "下载500组初筛排名", _v38_result["screen_summary"].to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v38_screen_ranking.csv", "text/csv",
        )
        _v38_downloads[1].download_button(
            "下载优胜复核排名", _v38_result["finalist_summary"].to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v38_finalist_ranking.csv", "text/csv",
        )
        _v38_downloads[2].download_button(
            "下载外层验证明细", _v38_result["outer_detail"].to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v38_outer_validation.csv", "text/csv",
        )
        _v38_downloads2 = st.columns(3)
        _v38_downloads2[0].download_button(
            "下载预测 Pareto", _v38_result["predictive_pareto_front"].to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v38_predictive_pareto.csv", "text/csv",
        )
        _v38_downloads2[1].download_button(
            "下载结构 Pareto", _v38_result["structure_pareto_front"].to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v38_structure_pareto.csv", "text/csv",
        )
        _v38_downloads2[2].download_button(
            "下载完整研究元数据", json.dumps(_v38_meta, ensure_ascii=False, indent=2),
            "lotto645_v38_metadata.json", "application/json",
        )

        st.divider()
        st.markdown("### 前瞻性预测锁定")
        st.write(
            "冻结上方选出的参数后，可生成下一期模型与均匀随机基准各10注×3组种子的锁定记录。"
            "锁定文件必须在目标期开奖号码公布前保存；导入后续手动历史数据后，系统才会计算命中结果。"
        )
        if "v38_forward_locks" not in st.session_state:
            st.session_state["v38_forward_locks"] = {}
        _v38_locks = st.session_state["v38_forward_locks"]
        _v38_next_draw = int(quality["last_draw"]) + 1
        if str(_v38_next_draw) in _v38_locks:
            st.success(f"第{_v38_next_draw}期已经存在前瞻锁定记录。为避免重复尝试，不能再次生成。")
        elif st.button("锁定下一期模型与随机基准号码", key="v38_lock_next_draw", use_container_width=True, disabled=not _v38_result_matches_current):
            try:
                _lock_detail, _lock_meta = build_forward_lock(
                    df, _v38_selected, tickets_per_draw=10, seed_groups=3,
                    seed=2028,
                )
                st.session_state["v38_forward_locks"][str(_lock_meta["target_draw"])] = {
                    "detail": _lock_detail,
                    "metadata": _lock_meta,
                }
                _lock_fp = _lock_meta.get("history_fingerprint_sha256")
                if _lock_fp:
                    _v40_event = v40_make_event("forward_lock", _lock_fp, {
                        "metadata": _lock_meta,
                        "detail_records": _lock_detail.to_dict(orient="records"),
                    })
                    _v40_add_event(_v40_event)
                st.success(f"第{_lock_meta['target_draw']}期前瞻锁定记录已生成。请立即下载并在开奖前保存。")
            except Exception as exc:
                st.error(f"生成前瞻锁定记录失败：{type(exc).__name__}: {exc}")
        _restore_lock_zip = st.file_uploader(
            "恢复以前保存的完整研究 ZIP（只导入其中的 V3.8 前瞻锁定记录）",
            type=["zip"], key="v38_restore_lock_bundle",
        )
        if _restore_lock_zip is not None and st.button("恢复 ZIP 中的前瞻锁定记录", key="v38_import_lock_bundle"):
            try:
                _restored_locks = restore_forward_locks_from_bundle(_restore_lock_zip.getvalue())
                _merged_locks = dict(st.session_state.get("v38_forward_locks", {}))
                for _target_key, _lock in _restored_locks.items():
                    if _target_key in _merged_locks:
                        raise ValueError(f"当前会话已经存在第{_target_key}期锁定记录，未覆盖现有记录。")
                    _merged_locks[_target_key] = _lock
                st.session_state["v38_forward_locks"] = _merged_locks
                st.success(f"已恢复{len(_restored_locks)}期前瞻锁定记录。")
            except Exception as exc:
                st.error(f"前瞻锁定记录恢复失败：{type(exc).__name__}: {exc}")
        _v38_lock_map = st.session_state.get("v38_forward_locks", {})
        for _target_key, _lock in sorted(_v38_lock_map.items(), key=lambda item: int(item[0])):
            _lock_detail = _lock.get("detail")
            _lock_meta = _lock.get("metadata", {})
            if isinstance(_lock_detail, pd.DataFrame) and not _lock_detail.empty:
                st.caption(f"前瞻锁定：第{_target_key}期；锁定时历史末期：{_lock_meta.get('history_last_draw')}；时间UTC：{_lock_meta.get('created_at_utc')}")
                st.download_button(
                    f"下载第{_target_key}期锁定号码 CSV",
                    _lock_detail.to_csv(index=False).encode("utf-8-sig"),
                    f"lotto645_v38_draw_{_target_key}_locked_tickets.csv", "text/csv",
                    key=f"v38_lock_csv_{_target_key}",
                )
                st.download_button(
                    f"下载第{_target_key}期锁定元数据 JSON",
                    json.dumps(_lock_meta, ensure_ascii=False, indent=2),
                    f"lotto645_v38_draw_{_target_key}_lock_metadata.json", "application/json",
                    key=f"v38_lock_meta_{_target_key}",
                )
        _v38_scored_detail, _v38_scored_summary = evaluate_forward_locks(df, _v38_lock_map)
        st.session_state["v40_scored_forward_detail"] = _v38_scored_detail
        st.session_state["v40_scored_forward_summary"] = _v38_scored_summary
        if not _v38_scored_detail.empty:
            st.markdown("### 已导入开奖结果的前瞻锁定复盘")
            st.warning("系统确认目标期号在锁定时历史末期之后，但不能独立验证你是否在开奖前下载/保存了锁定文件。请保留锁定时间与原始文件，供审计。")
            st.dataframe(_v38_scored_summary, hide_index=True, use_container_width=True)
            st.download_button(
                "下载前瞻锁定复盘明细", _v38_scored_detail.to_csv(index=False).encode("utf-8-sig"),
                "lotto645_v38_forward_validation_detail.csv", "text/csv",
            )

with page_ensemble:
    st.subheader("多模型集成实验")
    st.write("新策略融合近期频率、贝叶斯平滑和指数时间衰减。这里的权重是号码抽样权重，不是开奖概率。")
    recent_share = st.slider("近期频率权重", 0.05, 0.80, 0.40, 0.05, key="v361_ens_recent")
    max_bayes = round(0.95 - recent_share, 2)
    bayes_share = st.slider("贝叶斯平滑权重", 0.05, max(0.05, max_bayes), min(0.35, max_bayes), 0.05, key="v361_ens_bayes")
    decay_share = round(1.0 - recent_share - bayes_share, 2)
    st.metric("指数衰减权重（自动补足）", f"{decay_share:.0%}")
    if decay_share < 0:
        st.error("权重总和不得超过100%。请调整前两项。")
    else:
        st.write({"近期频率": recent_share, "贝叶斯平滑": bayes_share, "指数衰减": decay_share})
        ens_train = st.selectbox("集成模型最低训练期数", [w for w in [50, 100, 150, 200, 300, 500] if w < len(df)], index=0, key="v361_ens_train")
        ens_test_max = max(1, min(100, len(df) - ens_train))
        ens_tests = st.slider("集成模型测试期数", 1, ens_test_max, min(50, ens_test_max), key="v361_ens_tests")
        ens_seeds = st.selectbox("随机种子组数", [1, 3, 5], index=1, key="v361_ens_seeds")
        if st.button("评估集成模型与随机基准", type="primary", use_container_width=True, key="v361_eval_ensemble"):
            try:
                ens_raw, ens_summary, ens_meta = run_backtest(
                    df, lookback=ens_train, test_draws=ens_tests, tickets_per_draw=10,
                    n_seeds=ens_seeds, seed=2026, recent_window=ens_train,
                    multi_model_weights=(recent_share, bayes_share, decay_share),
                    strategies=["多模型集成模型", "均匀随机基准"],
                )
                st.session_state["v361_ensemble_raw"] = ens_raw
                st.session_state["v361_ensemble_summary"] = ens_summary
                st.session_state["v361_ensemble_meta"] = ens_meta
                st.session_state["v36_project_state"] = add_experiment(
                    st.session_state["v36_project_state"], "多模型集成实验", ens_meta,
                    ens_summary.to_dict(orient="records"),
                )
                st.success(f"集成实验完成：{len(ens_raw):,}条明细。")
            except Exception as exc:
                st.error(f"集成实验失败：{type(exc).__name__}: {exc}")
        ens_summary_now = st.session_state.get("v361_ensemble_summary")
        ens_raw_now = st.session_state.get("v361_ensemble_raw")
        ens_meta_now = st.session_state.get("v361_ensemble_meta")
        if isinstance(ens_summary_now, pd.DataFrame) and not ens_summary_now.empty:
            st.dataframe(ens_summary_now, hide_index=True, use_container_width=True)
            st.download_button("下载集成实验明细", ens_raw_now.to_csv(index=False).encode("utf-8-sig"), "lotto645_v361_ensemble_detail.csv", "text/csv")
            st.download_button("下载集成实验元数据", json.dumps(ens_meta_now, ensure_ascii=False, indent=2), "lotto645_v361_ensemble_metadata.json", "application/json")

with page_portfolio:
    st.subheader("10组候选号码组合优化")
    st.info("此功能是在模型产生的号码抽样权重下，用贪心启发式平衡号码评分、覆盖增量和组间重叠。组合多样性本身不提高单注理论中奖概率。")
    portfolio_models = [name for name in MODEL_NAMES if name != "均匀随机基准"]
    p_model = st.selectbox("基础选号模型", portfolio_models, index=portfolio_models.index("多模型集成模型"), key="v361_portfolio_model")
    p_window = st.selectbox("历史窗口", [50, 100, 200, 300, 500], index=1, key="v361_portfolio_window")
    p_overlap = st.slider("任意两组最多共享号码数", 0, 5, 4, key="v361_portfolio_overlap")
    p_pool = st.select_slider("候选池规模", options=[1000, 2000, 5000, 8000, 12000], value=5000, key="v361_portfolio_pool")
    q_weight = st.slider("模型号码评分权重", 0.10, 0.70, 0.30, 0.05, key="v361_portfolio_q")
    c_max = round(0.95 - q_weight, 2)
    c_weight = st.slider("新增号码覆盖权重", 0.10, max(0.10, c_max), min(0.45, c_max), 0.05, key="v361_portfolio_c")
    o_weight = round(1.0 - q_weight - c_weight, 2)
    st.metric("组间重叠惩罚权重（自动补足）", f"{o_weight:.0%}")
    p_seed = st.number_input("组合生成随机种子", min_value=0, max_value=2_000_000_000, value=2026, step=1, key="v361_portfolio_seed")
    if st.button("优化并生成10组号码", type="primary", use_container_width=True, key="v361_run_portfolio"):
        try:
            portfolio, portfolio_report = optimize_portfolio(
                df, model_name=p_model, count=10, window=p_window, max_overlap=p_overlap,
                candidate_pool_size=int(p_pool), quality_weight=q_weight,
                coverage_weight=c_weight, overlap_penalty=o_weight, seed=int(p_seed),
            )
            st.session_state["v361_portfolio"] = portfolio
            st.session_state["v361_portfolio_report"] = portfolio_report
            st.success("10组组合优化完成。")
        except Exception as exc:
            st.error(f"组合优化失败：{type(exc).__name__}: {exc}")
    portfolio_now = st.session_state.get("v361_portfolio")
    portfolio_report_now = st.session_state.get("v361_portfolio_report")
    if isinstance(portfolio_now, pd.DataFrame) and not portfolio_now.empty:
        st.dataframe(portfolio_now, hide_index=True, use_container_width=True)
        if isinstance(portfolio_report_now, dict):
            met = st.columns(4)
            met[0].metric("不同号码数", f"{portfolio_report_now['不同号码数']}/45")
            met[1].metric("号码覆盖率", f"{portfolio_report_now['号码覆盖率']:.1%}")
            met[2].metric("最大组间重号", portfolio_report_now["最大组间重号"])
            met[3].metric("平均组间重号", f"{portfolio_report_now['平均组间重号']:.2f}")
            with st.expander("查看组合质量报告"):
                st.json(portfolio_report_now)
        st.download_button("下载10组优化号码 CSV", portfolio_now.to_csv(index=False).encode("utf-8-sig"), "lotto645_v361_optimized_10_tickets.csv", "text/csv")
        st.download_button("下载组合质量报告 JSON", json.dumps(portfolio_report_now, ensure_ascii=False, indent=2), "lotto645_v361_portfolio_report.json", "application/json")

with page_models:
    st.subheader("模型管理与实验档案")
    st.write("实验状态保存在当前 Streamlit 会话中。为避免云端重启后丢失，请定期下载 JSON 档案并在新会话中重新导入。")
    state_upload = st.file_uploader("导入以前保存的模型档案 JSON", type=["json"], key="v361_state_upload")
    if state_upload is not None and st.button("导入模型档案", key="v361_import_state"):
        try:
            imported_state = import_project_state(state_upload.getvalue().decode("utf-8"))
            st.session_state["v36_project_state"] = imported_state
            st.success("模型档案导入成功。")
        except Exception as exc:
            st.error(f"模型档案导入失败：{exc}")
    current_state = st.session_state.get("v36_project_state", new_project_state())
    model_rows = []
    for name, record in current_state["models"].items():
        model_rows.append({"模型": name, "版本": record.get("version"), "状态": record.get("status"), "状态说明": record.get("status_note"), "证据实验": record.get("evidence_experiment_id")})
    st.markdown("### 当前模型状态")
    st.dataframe(pd.DataFrame(model_rows), hide_index=True, use_container_width=True)
    st.markdown("### 手动调整模型状态")
    editable_models = [name for name in current_state["models"] if name != "均匀随机基准"]
    status_model = st.selectbox("选择模型", editable_models, key="v361_status_model")
    new_status = st.selectbox("新状态", MODEL_STATUSES, key="v361_new_status")
    status_note = st.text_input("调整原因（必填）", key="v361_status_note")
    holdout_experiments = [x for x in current_state["experiments"] if x.get("holdout_evaluated") and x.get("model_name") == status_model]
    evidence_id = None
    if new_status == "验证通过":
        evidence_options = [x["experiment_id"] for x in holdout_experiments]
        evidence_id = st.selectbox("关联该模型的保留区间实验", evidence_options, key="v361_evidence_id") if evidence_options else None
        if not evidence_options:
            st.caption("尚无该模型的保留区间验证证据，暂时不能标记为验证通过。")
    if st.button("保存模型状态", key="v361_save_status"):
        try:
            st.session_state["v36_project_state"] = change_model_status(
                current_state, status_model, new_status, status_note, evidence_id,
            )
            st.success("模型状态已更新。")
        except Exception as exc:
            st.error(f"状态更新失败：{exc}")
    st.markdown("### 实验历史")
    experiment_rows = current_state["experiments"]
    if experiment_rows:
        exp_view = pd.DataFrame([{
            "实验编号": x.get("experiment_id"), "类型": x.get("experiment_type"),
            "模型": x.get("model_name"), "时间": x.get("created_at_utc"),
            "数据末期": x.get("data_last_draw"), "保留区间验证": x.get("holdout_evaluated"),
        } for x in experiment_rows])
        st.dataframe(exp_view, hide_index=True, use_container_width=True)
    else:
        st.caption("当前会话尚无实验记录。执行回测、参数搜索或保留区间验证后会自动记录摘要。")
    st.download_button("下载完整模型与实验档案 JSON", export_project_state(current_state), "lotto645_v361_model_archive.json", "application/json")

    st.divider()
    st.markdown("### 一键导出完整研究实验包")
    st.write("将当前会话中已经生成的回测、参数搜索、保留区间、集成模型、组合优化和模型档案打包为一个 ZIP，方便交给助手进行完整审查。")
    include_history = st.checkbox(
        "同时包含当前清洗后的手动历史开奖数据（建议勾选，便于复核数据与回测范围）",
        value=True, key="v362_export_include_history",
    )
    if st.button("生成完整研究实验 ZIP", type="primary", use_container_width=True, key="v362_build_research_bundle"):
        try:
            bundle_bytes, bundle_manifest = build_research_bundle(
                df, quality, data_source, st.session_state, current_state, include_history=include_history,
            )
            st.session_state["v362_research_bundle_bytes"] = bundle_bytes
            st.session_state["v362_research_bundle_manifest"] = bundle_manifest
            st.success(f"研究实验包生成成功：{len(bundle_bytes):,} 字节；包含 {len(bundle_manifest['included_artifacts'])} 个文件。")
        except Exception as exc:
            st.error(f"研究实验包生成失败：{type(exc).__name__}: {exc}")
    bundle_bytes = st.session_state.get("v362_research_bundle_bytes")
    bundle_manifest = st.session_state.get("v362_research_bundle_manifest")
    if isinstance(bundle_bytes, bytes) and bundle_bytes and isinstance(bundle_manifest, dict):
        created_tag = str(bundle_manifest.get("created_at_utc", ""))[:19].replace(":", "").replace("-", "")
        st.download_button(
            "下载 Lotto 6/45 完整研究实验包 ZIP",
            data=bundle_bytes,
            file_name=f"lotto645_research_bundle_{created_tag or 'latest'}.zip",
            mime="application/zip",
            use_container_width=True,
        )
        with st.expander("查看本次打包内容"):
            st.write({
                "历史数据已包含": bundle_manifest.get("history_included"),
                "历史期数": bundle_manifest.get("history_rows"),
                "最新期号": bundle_manifest.get("history_last_draw"),
                "已包含文件": bundle_manifest.get("included_artifacts"),
                "未生成/不可用项目": bundle_manifest.get("unavailable_or_empty_artifacts"),
            })

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

with page_v50:
    st.subheader("V5.0 联合概率预测实验")
    st.write(
        "V5.0采用严格时间顺序的号码特征、正则化逻辑概率模型、号码共现收缩与固定六号码概率校准。"
        "此页面的历史回测只用于研究筛查；真正的前瞻证据必须在开奖前锁定。"
    )
    st.warning(
        "为避免重复使用已审阅数据，V5.0历史回测默认只使用第1—1088期。"
        "第1089—1248期已在V3.7研究中审阅，不会被标记为新独立测试。"
    )
    safe_v50 = df[df["draw_no"] <= 1088].copy().reset_index(drop=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("安全回溯末期", int(safe_v50["draw_no"].max()) if not safe_v50.empty else "无")
    c2.metric("安全回溯记录", len(safe_v50))
    c3.metric("模型", "V5联合概率模型")
    with st.expander("V5.0模型设置", expanded=True):
        v50_window = st.selectbox("训练窗口", [300, 400, 500, 700, 900], index=2, key="v50_window")
        v50_alpha = st.select_slider("Ridge正则化强度", options=[0.1, 0.3, 1.0, 3.0, 10.0], value=1.0, key="v50_alpha")
        v50_test_draws = st.select_slider("历史回测期数（截至1088期）", options=[20, 30, 50, 80, 100], value=50, key="v50_test_draws")
        v50_tickets = st.select_slider("每期每组注数", options=[5, 10, 15, 20], value=10, key="v50_tickets")
    if st.button("运行 V5.0 安全区间滚动回测", type="primary", use_container_width=True, key="v50_run_backtest"):
        if len(safe_v50) < 300:
            st.error("截至第1088期的安全回溯数据不足300期。")
        else:
            progress = st.progress(0, text="准备V5特征缓存…")
            status = st.empty()
            try:
                def _v50_progress(done, total):
                    progress.progress(min(1.0, done / max(total, 1)), text=f"V5滚动评估：{done}/{total}")
                    status.caption("严格使用目标期开奖前可见的历史数据")
                result = run_v50_walk_forward(
                    safe_v50, test_draws=int(v50_test_draws),
                    training_window=int(v50_window), min_history=180,
                    ridge_alpha=float(v50_alpha), tickets_per_draw=int(v50_tickets),
                    seed=5026, bootstrap_reps=2000, progress_callback=_v50_progress,
                )
                st.session_state["v50_backtest_result"] = result
                st.success(f"V5.0回测完成：{result['test_draws']}期，耗时{result['elapsed_seconds']:.1f}秒。")
            except Exception as exc:
                st.error(f"V5.0回测失败：{type(exc).__name__}: {exc}")
    result = st.session_state.get("v50_backtest_result")
    if result:
        st.markdown("### V5.0 与随机对照结果")
        st.caption(f"回测范围：第{result['test_start_draw']}—{result['test_end_draw']}期；状态：历史回溯，不是前瞻证据。")
        st.dataframe(result["summary"].round(5), hide_index=True, use_container_width=True)
        st.download_button(
            "下载V5.0逐期回测明细CSV",
            result["detail"].to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v50_walk_forward_detail.csv", "text/csv", key="v50_detail_download",
        )
        st.download_button(
            "下载V5.0回测摘要CSV",
            result["summary"].to_csv(index=False).encode("utf-8-sig"),
            "lotto645_v50_walk_forward_summary.csv", "text/csv", key="v50_summary_download",
        )
        st.json({
            "引擎": result["engine_version"],
            "训练窗口": result["training_window"],
            "Ridge alpha": result["ridge_alpha"],
            "数据指纹": result["data_fingerprint_sha256"],
            "证据状态": result["evidence_status"],
        })
    st.markdown("### 下一期V5概率预测与开奖前锁定")
    st.caption("使用当前全部手动历史数据生成下一期研究候选；不会覆盖V4.0冻结快照。")
    if st.button("生成并锁定V5下一期预测（含随机对照）", key="v50_generate_next", use_container_width=True):
        try:
            prediction = v50_fit_predict_next(
                df, training_window=min(int(v50_window), 1000),
                min_history=180, ridge_alpha=float(v50_alpha),
            )
            probs = prediction["number_probabilities"].sort_values("号码")["预测入选概率"].to_numpy()
            lock_detail, lock_metadata = build_v50_forward_lock(df, prediction, tickets_per_group=10, seed=5050)
            tickets = [
                [int(lock_detail.iloc[row][f"n{j}"]) for j in range(1, 7)]
                for row in range(10)
            ]
            st.session_state["v50_next_prediction"] = prediction
            st.session_state["v50_next_tickets"] = tickets
            st.session_state["v50_forward_lock_detail"] = lock_detail
            st.session_state["v50_forward_lock_metadata"] = lock_metadata
        except Exception as exc:
            st.error(f"V5下一期预测失败：{type(exc).__name__}: {exc}")
    prediction = st.session_state.get("v50_next_prediction")
    tickets = st.session_state.get("v50_next_tickets")
    if prediction and tickets:
        st.write({
            "目标期号": prediction["target_draw"],
            "训练结束期": prediction["history_last_draw"],
            "模型": prediction["model_name"],
            "概率总和": round(prediction["probability_sum"], 6),
            "说明": prediction["interpretation"],
        })
        ticket_frame = pd.DataFrame([
            {"注序": idx, "号码": " - ".join(f"{n:02d}" for n in ticket), **{f"n{j+1}": n for j, n in enumerate(ticket)}}
            for idx, ticket in enumerate(tickets, 1)
        ])
        st.dataframe(ticket_frame, hide_index=True, use_container_width=True)
        st.dataframe(prediction["number_probabilities"].head(15).round(5), hide_index=True, use_container_width=True)
        st.download_button(
            "下载V5下一期候选号码CSV",
            ticket_frame.to_csv(index=False).encode("utf-8-sig"),
            f"lotto645_v50_draw_{prediction['target_draw']}_candidates.csv",
            "text/csv", key="v50_next_download",
        )
        lock_detail = st.session_state.get("v50_forward_lock_detail")
        lock_metadata = st.session_state.get("v50_forward_lock_metadata")
        if isinstance(lock_detail, pd.DataFrame) and lock_metadata:
            st.markdown("#### V5 / 随机对照锁定记录")
            st.dataframe(lock_detail, hide_index=True, use_container_width=True)
            st.download_button(
                "下载V5前瞻锁定CSV（含随机对照）",
                lock_detail.to_csv(index=False).encode("utf-8-sig"),
                f"lotto645_v50_lock_draw_{prediction['target_draw']}.csv",
                "text/csv", key="v50_lock_csv_download",
            )
            st.download_button(
                "下载V5前瞻锁定元数据JSON",
                json.dumps(lock_metadata, ensure_ascii=False, indent=2),
                f"lotto645_v50_lock_draw_{prediction['target_draw']}.json",
                "application/json", key="v50_lock_json_download",
            )
            st.warning("必须在目标期开奖前下载并保存CSV与JSON。新会话不会自动保留锁定；请同时备份两个文件。")

with page_v40:
    st.subheader("V4.0 研究治理与数据中心")
    st.write("将模型研究、前瞻锁定、数据版本与实验事件纳入统一的可恢复档案。所有开奖数据仍仅接受手动导入。")

    if isinstance(st.session_state.get("v40_history_override"), pd.DataFrame):
        st.info("检测到当前会话中有手动追加记录。请下载完整 CSV 并提交到 GitHub，才能永久保存。")
        st.download_button(
            "下载当前完整历史 CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            "lotto645.csv", "text/csv", key="v40_current_history_download",
        )

    current_fp = v40_data_fingerprint(df)
    selection_frame = df[df["draw_no"] <= int(V38_RETROSPECTIVE_CUTOFF_DRAW)]
    selection_fp = v40_data_fingerprint(selection_frame) if not selection_frame.empty else ""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("当前历史末期", int(quality["last_draw"]))
    c2.metric("历史记录", int(quality["rows"]))
    c3.metric("完整历史指纹", current_fp[:12] + "…")
    c4.metric("冻结选型指纹", (selection_fp[:12] + "…") if selection_fp else "暂无")

    st.markdown("### 1. 严格追加新期开奖")
    st.caption("只允许追加下一期；重复相同期数会幂等忽略，冲突期号、跳号或不合法号码会被拒绝。")
    with st.form("v40_append_draw_form"):
        expected_draw = int(quality["last_draw"]) + 1
        f1, f2 = st.columns(2)
        with f1:
            new_draw_no = st.number_input("新期号", min_value=expected_draw, max_value=expected_draw, value=expected_draw, step=1)
        with f2:
            new_draw_date = st.text_input("开奖日期（YYYY-MM-DD）", placeholder="2026-11-14")
        ncols = st.columns(7)
        new_numbers = []
        for idx in range(6):
            new_numbers.append(int(ncols[idx].number_input(f"主号 {idx+1}", min_value=1, max_value=45, value=idx + 1, key=f"v40_new_n{idx+1}")))
        new_bonus = int(ncols[6].number_input("Bonus", min_value=1, max_value=45, value=7, key="v40_new_bonus"))
        append_clicked = st.form_submit_button("校验并追加到当前会话", use_container_width=True)
    if append_clicked:
        try:
            updated_history, append_status = v40_append_draw(df, int(new_draw_no), new_draw_date, new_numbers, new_bonus)
            st.session_state["v40_history_override"] = updated_history
            appended_fp = v40_data_fingerprint(updated_history)
            _v40_add_event(v40_make_event("draw_appended", appended_fp, {
                "draw_no": int(new_draw_no), "draw_date": new_draw_date,
                "numbers": sorted(new_numbers), "bonus": new_bonus,
                "append_status": append_status, "history_rows": len(updated_history),
            }))
            st.success(f"第{new_draw_no}期追加成功。请下载完整 CSV 并提交到 GitHub，作为永久历史数据。")
            st.download_button(
                "下载追加后的完整历史 CSV",
                updated_history.to_csv(index=False).encode("utf-8-sig"),
                "lotto645.csv", "text/csv", key=f"v40_append_download_{new_draw_no}",
            )
            st.rerun()
        except Exception as exc:
            st.error(f"追加失败：{type(exc).__name__}: {exc}")

    st.markdown("### 2. 实验档案导出与恢复")
    st.caption("实验事件档案包含模型快照、手动追加事件与前瞻锁定记录，不包含 GitHub Token 或 Streamlit Secret。")
    events_now = list(st.session_state.get("v40_events", []))
    st.download_button(
        "下载 V4.0 实验事件档案 JSON", export_event_archive(events_now),
        "lotto645_v40_event_archive.json", "application/json", use_container_width=True,
    )
    archive_upload = st.file_uploader("恢复 V4.0 实验事件档案 JSON", type=["json"], key="v40_archive_import")
    if archive_upload is not None and st.button("验证并恢复事件档案", key="v40_restore_archive"):
        try:
            imported = import_event_archive(archive_upload.getvalue())
            existing_ids = {item["event_id"] for item in events_now}
            merged = events_now + [item for item in imported if item["event_id"] not in existing_ids]
            restored_locks = events_to_forward_locks(merged)
            current_locks = dict(st.session_state.get("v38_forward_locks", {}))
            for draw_key, lock in restored_locks.items():
                if draw_key in current_locks:
                    existing = current_locks[draw_key]["metadata"]
                    incoming = lock["metadata"]
                    if existing.get("history_fingerprint_sha256") != incoming.get("history_fingerprint_sha256"):
                        raise ValueError(f"第{draw_key}期已存在另一份锁定记录，拒绝覆盖。")
                else:
                    current_locks[draw_key] = lock
            st.session_state["v40_events"] = merged
            st.session_state["v38_forward_locks"] = current_locks
            st.success(f"已恢复 {len(imported)} 条事件，识别到 {len(restored_locks)} 期前瞻锁定。")
            st.rerun()
        except Exception as exc:
            st.error(f"事件档案恢复失败：{type(exc).__name__}: {exc}")

    st.markdown("### 3. 冻结模型快照与下一期锁定")
    snapshot_event = latest_model_snapshot(events_now)
    if snapshot_event:
        snapshot = snapshot_event["payload"]
        frozen_config = snapshot["selected_config"]
        st.json({
            "模型": frozen_config.get("model_name"), "参数": frozen_config.get("params"),
            "选型结束期": snapshot.get("selection_end_draw"),
            "选型数据指纹": snapshot.get("selection_data_fingerprint"),
            "研究ID": snapshot.get("research_experiment_id"),
            "预测证据状态": snapshot.get("predictive_advantage_status"),
        })
        lock_target = int(quality["last_draw"]) + 1
        already_locked = str(lock_target) in st.session_state.get("v38_forward_locks", {})
        if st.button(f"使用冻结快照锁定第{lock_target}期", disabled=already_locked, key="v40_lock_from_snapshot", use_container_width=True):
            try:
                from robust_optimizer import build_forward_lock
                lock_detail, lock_meta = build_forward_lock(df, frozen_config, tickets_per_draw=10, seed_groups=3, seed=2040)
                st.session_state.setdefault("v38_forward_locks", {})[str(lock_target)] = {"detail": lock_detail, "metadata": lock_meta}
                _v40_add_event(v40_make_event("forward_lock", lock_meta["history_fingerprint_sha256"], {
                    "metadata": lock_meta, "detail_records": lock_detail.to_dict(orient="records"),
                }))
                st.success(f"第{lock_target}期模型与随机基准号码已锁定；请立即保存原始锁定文件。")
                st.rerun()
            except Exception as exc:
                st.error(f"锁定失败：{type(exc).__name__}: {exc}")
    else:
        st.warning("尚无可恢复的 V4.0 冻结模型快照。请先在 V3.8 稳健研究页完成一次研究，或导入以前保存的 V4.0 事件档案。")

    st.markdown("### 4. 前瞻效果与模型晋升审查")
    scored_forward = st.session_state.get("v40_scored_forward_detail")
    if isinstance(scored_forward, pd.DataFrame) and not scored_forward.empty:
        candidate_names = [name for name in scored_forward["策略"].dropna().unique().tolist() if name != "均匀随机基准"]
        all_reports, gates = [], []
        for model_idx, model_name in enumerate(candidate_names):
            report, gate = v40_evaluate_forward_pairs(scored_forward, model_name, repetitions=5000, seed=2040 + model_idx * 17)
            if not report.empty:
                report.insert(0, "模型", model_name)
                all_reports.append(report)
            gates.append({"模型": model_name, **gate})
        if all_reports:
            st.dataframe(pd.concat(all_reports, ignore_index=True), hide_index=True, use_container_width=True)
        if gates:
            st.dataframe(pd.DataFrame(gates), hide_index=True, use_container_width=True)
        st.info("正式晋升门槛：至少100个前瞻开奖期；三个主要指标点估计均高于随机；区块Bootstrap 95%CI下限均高于0；Holm校正p值均小于0.05。")
    else:
        st.caption("当前没有可评价的开奖前锁定复盘记录。")

    st.markdown("### 5. 可选 GitHub 持久化（只保存研究事件）")
    st.caption("不配置 Token 时，可使用 JSON 导出与恢复。配置后仅写入独立 research-data 分支，不改动 main 和开奖记录。")
    if st.button("检查并恢复 GitHub 研究档案", key="v40_remote_load"):
        try:
            token = st.secrets.get("GITHUB_RESEARCH_TOKEN", "")
            if not token:
                st.info("尚未配置 GITHUB_RESEARCH_TOKEN。可以继续使用 JSON 导出/恢复。")
            else:
                remote_events = GitHubEventStore("Frogbornwater/korea-lotto645-analyzer", str(token), branch="research-data").load_events()
                merged_ids = {item["event_id"] for item in events_now}
                st.session_state["v40_events"] = events_now + [item for item in remote_events if item["event_id"] not in merged_ids]
                restored = events_to_forward_locks(st.session_state["v40_events"])
                st.session_state["v38_forward_locks"] = {**st.session_state.get("v38_forward_locks", {}), **restored}
                st.success(f"已从 research-data 分支恢复 {len(remote_events)} 条研究事件。")
                st.rerun()
        except EventStoreError as exc:
            st.error(f"GitHub研究档案读取失败：{exc}")
        except Exception as exc:
            st.error(f"GitHub研究档案读取失败：{type(exc).__name__}: {exc}")

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
