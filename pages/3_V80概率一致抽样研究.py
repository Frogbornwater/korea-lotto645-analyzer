# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_manager import read_manual_upload
from lotto_core import quality_report, validate_frame
from v50_predictive_engine import fit_predict_next
from v51_smart_picker import generate_constrained_tickets
from v80_probability_sampling import (
    generate_probability_faithful_tickets,
    summarize_ticket_batch,
)

st.set_page_config(page_title="V8.0概率一致抽样研究", page_icon="🎯", layout="wide")
st.title("🎯 V8.0 概率一致抽样研究")
st.caption("研究专用｜同一组V5概率输入｜旧序贯抽样与固定样本量概率一致抽样对照｜只使用手动历史数据")

st.warning(
    "本页面不改变正式V5选号，也不修改任何已冻结的前瞻档案。"
    "V8.0只修正给定45维边际概率到单注六号码组合分布的映射；"
    "它不能凭抽样算法本身创造开奖预测优势。"
)
st.markdown(
    "本实验将同一组V5预测概率输入两种选号方式：\n"
    "1. 旧V5.1：序贯加权抽样、历史共现调整，并应用10注之间的重号上限。\n"
    "2. V8.0：条件伯努利（rejective）抽样，求解固定抽取6个号码时的权重，"
    "使单注分布的45个边际入选概率与V5输入概率一致；不应用跨注重号过滤。"
)

st.markdown("### 1. 导入手动历史数据")
upload = st.file_uploader(
    "上传完整历史 CSV / XLSX / XLS；不上传时读取仓库内的手动历史CSV",
    type=["csv", "xlsx", "xls"],
    key="v80_history_upload",
)
try:
    if upload is None:
        raw_history = pd.read_csv(ROOT / "data" / "lotto645.csv")
        source_label = "仓库中的手动历史CSV"
        history = validate_frame(raw_history).reset_index(drop=True)
        quality = quality_report(raw_history)
    else:
        history, quality = read_manual_upload(upload, upload.name)
        source_label = f"用户上传：{upload.name}"
        history = history.reset_index(drop=True)
    if history.empty:
        raise ValueError("没有有效开奖记录。")
except Exception as exc:
    st.error(f"历史数据读取失败：{type(exc).__name__}: {exc}")
    st.stop()

first_draw = int(history["draw_no"].iloc[0])
last_draw = int(history["draw_no"].iloc[-1])
contiguous = bool(quality.get("sequence_complete", False)) and first_draw == 1
if not contiguous:
    st.error("历史记录必须从第1期开始且期号连续；请修复或重新上传手动数据。")
else:
    st.success(f"已读取 {len(history):,} 期有效历史，期号 {first_draw}—{last_draw}；预测目标为第 {last_draw + 1} 期。")
st.caption(source_label + "。本页面不访问任何开奖网站或API。")

st.markdown("### 2. 设置同一组V5概率输入")
c1, c2, c3 = st.columns(3)
with c1:
    training_window = st.selectbox("训练窗口", [180, 200, 250, 300, 400, 500, 700, 900, 1000], index=5)
with c2:
    ridge_alpha = st.select_slider("Ridge alpha", options=[0.1, 0.3, 1.0, 3.0, 10.0], value=1.0)
with c3:
    pair_strength = st.slider("旧V5.1共现抽样强度", 0.0, 1.0, 0.5, 0.05)
c4, c5 = st.columns(2)
with c4:
    overlap_limit = st.slider("旧V5.1任意两注最多共享号码", 0, 5, 4)
with c5:
    seed = st.number_input("固定随机种子", min_value=1, max_value=2147483646, value=8032028, step=1)

if contiguous and last_draw <= 1250:
    st.warning(
        "当前手动数据仍停在第1250期。第1251期已有V5.1正式前瞻锁定，"
        "本页面不会生成第二套1251期候选。请在开奖后手动导入并核验第1251期，"
        "届时V8.0才会对第1252期运行独立研究。"
    )
ready = contiguous and len(history) >= 210 and last_draw > 1250
if st.button("运行V5.1与V8.0同概率对照", type="primary", use_container_width=True, disabled=not ready):
    try:
        prediction = fit_predict_next(
            history,
            training_window=int(training_window),
            min_history=180,
            ridge_alpha=float(ridge_alpha),
        )
        probability_frame = prediction["number_probabilities"].sort_values("号码").reset_index(drop=True)
        probabilities = probability_frame["预测入选概率"].to_numpy(dtype=float)
        pair_scores = prediction["pair_scores"]

        legacy_tickets, effective_overlap, relaxed = generate_constrained_tickets(
            probabilities,
            pair_scores,
            count=10,
            seed=int(seed),
            pair_strength=float(pair_strength),
            max_overlap=int(overlap_limit),
        )
        v8_tickets, fitted, v8_batch = generate_probability_faithful_tickets(
            probabilities,
            count=10,
            seed=int(seed),
            sample_size=6,
        )
        legacy_batch = summarize_ticket_batch(legacy_tickets, probabilities)

        ticket_rows = []
        for label, tickets in [("V5.1旧序贯抽样", legacy_tickets), ("V8.0概率一致抽样", v8_tickets)]:
            for index, ticket in enumerate(tickets, start=1):
                ticket_rows.append({
                    "抽样机制": label,
                    "注序": index,
                    "号码": " - ".join(f"{number:02d}" for number in ticket),
                    "和值": int(sum(ticket)),
                    "奇数个数": int(sum(number % 2 for number in ticket)),
                })
        result = {
            "research_only": True,
            "prediction_engine": prediction["engine_version"],
            "target_draw": int(prediction["target_draw"]),
            "history_first_draw": int(prediction["history_first_draw"]),
            "history_last_draw": int(prediction["history_last_draw"]),
            "history_rows": int(prediction["history_rows"]),
            "training_window": int(training_window),
            "ridge_alpha": float(ridge_alpha),
            "legacy_pair_strength": float(pair_strength),
            "legacy_requested_overlap": int(overlap_limit),
            "legacy_effective_overlap": int(effective_overlap),
            "legacy_overlap_relaxed": bool(relaxed),
            "seed": int(seed),
            "number_probabilities": [
                {"number": int(number), "probability": float(probability)}
                for number, probability in zip(probability_frame["号码"], probabilities)
            ],
            "legacy_tickets": legacy_tickets,
            "v8_tickets": v8_tickets,
            "v8_sampler": {
                "version": fitted.version,
                "iterations": int(fitted.iterations),
                "max_abs_analytic_marginal_error": float(fitted.max_abs_error),
                "target_marginals": fitted.target_marginals.tolist(),
                "fitted_marginals": fitted.fitted_marginals.tolist(),
            },
            "legacy_batch_marginal_diagnostics": legacy_batch,
            "v8_batch_marginal_diagnostics": v8_batch,
            "prospective_lock_created": False,
            "production_model_changed": False,
        }
        st.session_state["v80_last_result"] = result
        st.session_state["v80_last_tickets"] = pd.DataFrame(ticket_rows)
    except Exception as exc:
        st.error(f"V8.0实验运行失败：{type(exc).__name__}: {exc}")

result = st.session_state.get("v80_last_result")
tickets_frame = st.session_state.get("v80_last_tickets")
if result:
    st.divider()
    st.markdown("### 3. 对照结果")
    st.caption(f"目标第 {result['target_draw']} 期｜V5输入概率总和：{sum(x['probability'] for x in result['number_probabilities']):.8f}")
    st.dataframe(tickets_frame, hide_index=True, use_container_width=True)
    diagnostics = pd.DataFrame([
        {
            "指标": "分析组合数",
            "V5.1旧序贯抽样": result["legacy_batch_marginal_diagnostics"]["ticket_count"],
            "V8.0概率一致抽样": result["v8_batch_marginal_diagnostics"]["ticket_count"],
        },
        {
            "指标": "45维批次入选频率平均绝对偏差",
            "V5.1旧序贯抽样": result["legacy_batch_marginal_diagnostics"]["mean_absolute_marginal_deviation"],
            "V8.0概率一致抽样": result["v8_batch_marginal_diagnostics"]["mean_absolute_marginal_deviation"],
        },
        {
            "指标": "45维批次入选频率最大绝对偏差",
            "V5.1旧序贯抽样": result["legacy_batch_marginal_diagnostics"]["max_absolute_marginal_deviation"],
            "V8.0概率一致抽样": result["v8_batch_marginal_diagnostics"]["max_absolute_marginal_deviation"],
        },
        {
            "指标": "单注分布解析边际最大误差",
            "V5.1旧序贯抽样": None,
            "V8.0概率一致抽样": result["v8_sampler"]["max_abs_analytic_marginal_error"],
        },
    ])
    st.dataframe(diagnostics, hide_index=True, use_container_width=True)
    st.caption(
        "注意：十注的实际号码频率会有很大的有限样本波动。V8.0的解析误差是对完整单注抽样分布的检查；"
        "十注批次的实际频率不需要、也不可能逐项精确等于模型概率。跨注重号限制会改变组合批次分布。"
    )
    st.markdown("#### V5预测概率与V8解析边际复核")
    probability_audit = pd.DataFrame(result["number_probabilities"])
    probability_audit["V8解析边际"] = result["v8_sampler"]["fitted_marginals"]
    probability_audit["解析误差"] = probability_audit["V8解析边际"] - probability_audit["probability"]
    st.dataframe(probability_audit.sort_values("probability", ascending=False).round(8), hide_index=True, use_container_width=True)
    st.download_button(
        "下载V8.0研究结果JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"lotto645_v80_sampling_research_{result['target_draw']}.json",
        mime="application/json",
    )
    st.info(
        "这次实验只检验抽样分布是否兑现V5提供的45维边际概率，不证明V5概率比随机更准确。"
        "V8.0结果不会写入正式预测、1251期既有锁定、V5.5冻结档案或其他前瞻承诺。"
    )
else:
    st.info("完成设置后运行对照；页面会保留本次研究结果供检查和下载。")
