# -*- coding: utf-8 -*-
"""Read-only Streamlit page for the verified V9.5 common-horizon audit."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "data" / "research_reports" / "v95_common_horizon_summary.json"

st.title("V9.5｜共同区间逐期审计")
st.caption("只读研究报告｜V9.4 vs V5.1 vs 均匀随机基准｜不改变正式选号算法")

if not REPORT_PATH.exists():
    st.error("V9.5审计摘要文件不存在。请检查仓库中的 data/research_reports/v95_common_horizon_summary.json。")
    st.stop()

with REPORT_PATH.open("r", encoding="utf-8") as handle:
    report = json.load(handle)

protocol = report["protocol"]
means = report["mean_metrics"]
comparisons = report["paired_comparisons"]

st.warning(
    "审计结论：尚未建立预测优势。V9.4的各项平均点估计均未优于V5.1；"
    "与V5.1及均匀随机基准的配对95%区间均跨过0。"
)

kpi = st.columns(4)
kpi[0].metric("逐期目标数", f"{protocol['target_draws']}期")
kpi[1].metric("测试范围", f"{protocol['target_start']}—{protocol['target_end']}期")
kpi[2].metric("每模型每期组合", f"{protocol['tickets_per_model_per_draw']}注")
kpi[3].metric("数据末期", f"第{protocol['latest_draw_in_csv']}期")

st.subheader("三种策略的平均表现")
metric_labels = {
    "brier": "Brier（越低越好）",
    "logloss": "LogLoss（越低越好）",
    "top6_hits": "Top-6平均命中数（越高越好）",
    "ticket_mean_hits": "单注平均命中数（越高越好）",
    "ticket_3plus_rate": "单注3+命中率（越高越好）",
    "batch_3plus": "每期10注至少一注命中3+的比例（越高越好）",
}
rows = []
for key, label in metric_labels.items():
    rows.append({
        "指标": label,
        "V9.4": means["v94"][key],
        "V5.1": means["v51"][key],
        "均匀随机": means["uniform"][key],
    })
metric_frame = pd.DataFrame(rows)
st.dataframe(metric_frame, hide_index=True, use_container_width=True)

st.subheader("配对差值与95% Bootstrap区间")
st.caption("差值定义为 V9.4 − 对照策略。Brier、LogLoss越低越好；其余命中指标越高越好。区间跨0表示当前数据不能确认差异方向。")

comparison_rows = []
for baseline, title in [("v51", "V9.4 − V5.1"), ("uniform", "V9.4 − 均匀随机")]:
    result = comparisons[f"v94_minus_{baseline}"]
    for key, label in metric_labels.items():
        item = result[key]
        comparison_rows.append({
            "比较": title,
            "指标": label,
            "平均差值": item["mean_difference"],
            "95%区间下限": item["ci95"][0],
            "95%区间上限": item["ci95"][1],
            "区间是否跨0": item["ci95"][0] <= 0 <= item["ci95"][1],
        })
st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)

st.subheader("审计协议")
st.write({
    "目标期范围": f"{protocol['target_start']}—{protocol['target_end']}",
    "严格前瞻约束": protocol["history_cutoff"],
    "V9.4窗口": protocol["v94_windows"],
    "V9.4 Dirichlet alpha": protocol["v94_alpha"],
    "V5.1训练窗口": protocol["v51_training_window"],
    "V5.1 Ridge alpha": protocol["v51_ridge_alpha"],
    "每模型每期组合数": protocol["tickets_per_model_per_draw"],
    "任意两注最大共享号码": protocol["max_pairwise_ticket_overlap"],
    "不确定性估计": protocol["bootstrap"],
})
st.caption(f"原始CSV SHA-256：{protocol['data_sha256_raw_csv']}")
st.caption(f"GitHub Actions运行：{protocol['workflow_run_id']}｜审计合并提交：{protocol['audit_merge_commit']}")

st.subheader("对当前选号器的影响")
st.info(
    "本页只展示V9.5审计证据。它不自动更改V9.4默认选号器，不改写V5.1参数，"
    "也不把历史回测结果转换成下一期预测优势。当前生产选号器仍按主应用中已配置的V9.4运行。"
)

st.download_button(
    "下载V9.5审计摘要JSON",
    data=json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
    file_name="v95_common_horizon_summary.json",
    mime="application/json",
)
