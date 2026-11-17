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
from v80_probability_sampling import generate_probability_faithful_tickets
from v81_batch_sampler import generate_batch_quota_portfolio, portfolio_diagnostics

st.set_page_config(page_title="V8.1批次配额组合研究", page_icon="🧩", layout="wide")
st.title("🧩 V8.1 批次配额组合研究")
st.caption("研究专用｜相同V5概率输入｜旧V5.1、V8.0独立抽样、V8.1批次配额三种组合对照")

st.warning(
    "本页面不改变正式V5选号，不创建前瞻锁定，也不修改任何已冻结档案。"
    "V8.1控制的是10注组合合计的号码出现次数，不是让每一注都具有相同的边际分布。"
)
st.markdown(
    "本页面对同一组45维V5边际概率比较三种组合方式：\n"
    "1. **V5.1旧机制**：序贯加权抽样、共现调整，并应用跨注重号上限。\n"
    "2. **V8.0单注概率一致**：每注独立使用条件伯努利抽样；单注边际概率与输入概率匹配，但10注之间不做覆盖或重号过滤。\n"
    "3. **V8.1批次配额**：用成对依赖舍入分配整批60个号码位置，每个号码的批次出现次数只取输入期望次数的floor或ceil；跨随机种子的期望出现次数为10倍输入概率，再将配额分配成10组互不重复的组合。"
)

st.markdown("### 1. 导入手动历史数据")
upload = st.file_uploader(
    "上传完整历史 CSV / XLSX / XLS；不上传时读取仓库内的手动历史CSV",
    type=["csv", "xlsx", "xls"],
    key="v81_history_upload",
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
    st.success(
        f"已读取 {len(history):,} 期有效历史，期号 {first_draw}—{last_draw}；"
        f"研究目标为第 {last_draw + 1} 期。"
    )
st.caption(source_label + "。本页面只读取手动提供的数据，不访问任何开奖网站或API。")

st.markdown("### 2. 同一组V5预测输入")
c1, c2, c3 = st.columns(3)
with c1:
    training_window = st.selectbox(
        "训练窗口", [180, 200, 250, 300, 400, 500, 700, 900, 1000],
        index=5, key="v81_training_window",
    )
with c2:
    ridge_alpha = st.select_slider(
        "Ridge alpha", options=[0.1, 0.3, 1.0, 3.0, 10.0],
        value=1.0, key="v81_ridge_alpha",
    )
with c3:
    pair_strength = st.slider(
        "V5.1旧机制共现抽样强度", 0.0, 1.0, 0.5, 0.05,
        key="v81_pair_strength",
    )
c4, c5 = st.columns(2)
with c4:
    overlap_limit = st.slider(
        "V5.1旧机制任意两注最多共享号码", 0, 5, 4,
        key="v81_overlap_limit",
    )
with c5:
    seed = st.number_input(
        "固定随机种子", min_value=1, max_value=2147483646,
        value=81012028, step=1, key="v81_seed",
    )

ready = contiguous and len(history) >= 210
if st.button(
    "运行三种10注组合对照", type="primary",
    use_container_width=True, disabled=not ready,
    key="v81_run_comparison",
):
    try:
        prediction = fit_predict_next(
            history,
            training_window=int(training_window),
            min_history=180,
            ridge_alpha=float(ridge_alpha),
        )
        probability_frame = (
            prediction["number_probabilities"].sort_values("号码").reset_index(drop=True)
        )
        probabilities = probability_frame["预测入选概率"].to_numpy(dtype=float)
        pair_scores = prediction["pair_scores"]

        legacy_tickets, effective_overlap, relaxed = generate_constrained_tickets(
            probabilities, pair_scores, count=10, seed=int(seed),
            pair_strength=float(pair_strength), max_overlap=int(overlap_limit),
        )
        v80_tickets, v80_fit, v80_batch = generate_probability_faithful_tickets(
            probabilities, count=10, seed=int(seed), sample_size=6,
        )
        v81_tickets, v81_batch = generate_batch_quota_portfolio(
            probabilities, ticket_count=10, sample_size=6, seed=int(seed),
        )

        groups = [
            ("V5.1旧机制", legacy_tickets),
            ("V8.0单注概率一致", v80_tickets),
            ("V8.1批次配额", v81_tickets),
        ]
        diagnostics = {
            label: portfolio_diagnostics(tickets, probabilities)
            for label, tickets in groups
        }
        ticket_rows = []
        for label, tickets in groups:
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
            "v80_independent_tickets": v80_tickets,
            "v81_batch_quota_tickets": v81_tickets,
            "v80_sampler": {
                "version": v80_fit.version,
                "max_abs_analytic_marginal_error": float(v80_fit.max_abs_error),
            },
            "legacy_batch_diagnostics": diagnostics["V5.1旧机制"],
            "v80_batch_diagnostics": diagnostics["V8.0单注概率一致"],
            "v81_batch_diagnostics": v81_batch,
            "prospective_lock_created": False,
            "production_model_changed": False,
        }
        st.session_state["v81_last_result"] = result
        st.session_state["v81_last_tickets"] = pd.DataFrame(ticket_rows)
    except Exception as exc:
        st.error(f"V8.1实验运行失败：{type(exc).__name__}: {exc}")

result = st.session_state.get("v81_last_result")
tickets_frame = st.session_state.get("v81_last_tickets")
if result:
    st.divider()
    st.markdown("### 3. 三种机制的10注组合")
    st.caption(
        f"目标第 {result['target_draw']} 期；V5输入概率总和："
        f"{sum(x['probability'] for x in result['number_probabilities']):.8f}"
    )
    st.dataframe(tickets_frame, hide_index=True, use_container_width=True)

    legacy = result["legacy_batch_diagnostics"]
    v80 = result["v80_batch_diagnostics"]
    v81 = result["v81_batch_diagnostics"]
    metric_keys = [
        ("不同组合数", "unique_ticket_count"),
        ("号码覆盖数", "number_coverage"),
        ("最大两注重叠数", "max_pairwise_overlap"),
        ("两注平均重叠数", "mean_pairwise_overlap"),
        ("批次号码频率平均绝对偏差", "mean_absolute_batch_marginal_error"),
        ("批次号码频率最大绝对偏差", "max_absolute_batch_marginal_error"),
    ]
    metrics = []
    for label, key in metric_keys:
        metrics.append({
            "指标": label,
            "V5.1旧机制": legacy.get(key),
            "V8.0单注概率一致": v80.get(key),
            "V8.1批次配额": v81.get(key),
        })
    st.dataframe(pd.DataFrame(metrics), hide_index=True, use_container_width=True)

    st.markdown("#### V8.1号码配额核对")
    quota = pd.DataFrame({
        "号码": [f"{i:02d}" for i in range(1, 46)],
        "输入边际概率": [
            item["probability"] for item in result["number_probabilities"]
        ],
        "期望10注出现次数": v81["expected_appearances"],
        "实际10注出现次数": v81["realized_appearances"],
        "实际出现频率": v81["realized_batch_marginals"],
    })
    quota["实际－输入频率"] = quota["实际出现频率"] - quota["输入边际概率"]
    st.dataframe(
        quota.sort_values("输入边际概率", ascending=False).round(6),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "V8.1的批次配额是随机无偏的：对每个号码，跨大量独立随机种子重复生成时，"
        "平均出现次数回到10倍输入概率。单次10注会受到整数次数限制；"
        "V8.1并不声称每一注都保留V8.0的同一组边际分布。"
    )
    st.download_button(
        "下载V8.1研究结果JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"lotto645_v81_batch_quota_{result['target_draw']}.json",
        mime="application/json",
    )
    st.info(
        "本页只检验组合分配机制，不证明V5预测概率优于随机，也不改变正式选号。"
        "后续如进入前瞻验证，必须先独立锁定策略、号码和时间戳，再在手动导入开奖数据后评分。"
    )
else:
    st.info("完成设置后运行对照；页面会保留本次研究结果供检查和下载。")
