# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lotto_core import validate_frame
from v40_store import export_event_archive, import_event_archive
from v55_forward_temperature import (
    EVENT_TYPE,
    build_v55_temperature_forward_event,
    check_frozen_base_history,
    evaluate_v55_temperature_forward_events,
    load_commitment,
)

st.set_page_config(page_title="V5.5温度校准前瞻研究", page_icon="🧪", layout="wide")
st.title("🧪 V5.5 温度校准前瞻研究")
st.caption("独立研究轨道｜冻结 T=2.875｜原始 V5 与均匀随机对照｜仅使用手动导入的 CSV / Excel")

try:
    commitment = load_commitment()
except Exception as exc:
    st.error(f"V5.5冻结参数承诺无法通过校验：{type(exc).__name__}: {exc}")
    st.stop()

lock = commitment["temperature_lock"]
source = commitment["source_artifact"]
st.markdown("### 已冻结的研究参数")
c1, c2, c3, c4 = st.columns(4)
c1.metric("温度参数", f"{float(lock['temperature']):.3f}")
c2.metric("校准期", f"{lock['calibration_period'][0]}—{lock['calibration_period'][1]}")
c3.metric("基准历史末期", str(lock["base_last_draw"]))
c4.metric("训练窗口", str(lock["training_window"]))
st.caption(
    "来源：GitHub Actions V5.5 归档 "
    f"#{source['artifact_id']}（run {source['workflow_run_id']}）。"
    f"锁文件 SHA-256：{source['forward_temperature_lock_sha256']}"
)
st.warning(
    "T=2.875 只作为研究候选，不参与正式V5选号。"
    "第1251期已有独立V5.1原始锁定，本页面禁止建立第二份1251期锁定，"
    "也不会修改或重算原档案。"
)

st.divider()
st.markdown("### 1. 导入最新手动历史数据")
uploaded_history = st.file_uploader(
    "上传最新历史数据（CSV、XLSX 或 XLS）；不上传时读取仓库内的手动CSV",
    type=["csv", "xlsx", "xls"],
    key="v77_manual_history_upload",
)

try:
    if uploaded_history is None:
        raw_history = pd.read_csv(ROOT / "data" / "lotto645.csv")
        history_source = "仓库手动CSV"
    else:
        suffix = Path(uploaded_history.name).suffix.lower()
        if suffix == ".csv":
            raw_history = pd.read_csv(uploaded_history, encoding="utf-8-sig")
        else:
            raw_history = pd.read_excel(uploaded_history)
        history_source = f"用户上传：{uploaded_history.name}"
    history = validate_frame(raw_history).reset_index(drop=True)
    if history.empty:
        raise ValueError("没有可用的有效开奖记录。")
    history_last = int(history["draw_no"].iloc[-1])
    history_first = int(history["draw_no"].iloc[0])
    is_contiguous = history_first == 1 and bool((history["draw_no"].diff().dropna() == 1).all())
    prefix_audit = check_frozen_base_history(history, commitment)
except Exception as exc:
    st.error(f"历史数据读取或校验失败：{type(exc).__name__}: {exc}")
    st.stop()

h1, h2, h3 = st.columns(3)
h1.metric("有效历史期数", f"{len(history):,}")
h2.metric("历史末期", str(history_last))
h3.metric("下一目标期", str(history_last + 1))
st.caption(history_source)

if not is_contiguous:
    st.error("历史期号必须从第1期开始且连续。请先修复手动数据文件。")
elif prefix_audit.get("ok"):
    st.success(
        "V5.5冻结基准1—1250期指纹一致。"
        "日期文本在本次前缀校验中被忽略，只核对期号、六个主号码与Bonus。"
    )
else:
    st.error(str(prefix_audit.get("status", "冻结历史前缀校验失败")))

if is_contiguous and prefix_audit.get("ok") and history_last <= 1250:
    st.warning(
        "目前数据仍未覆盖第1251期。请先手动导入并验证1251期，"
        "之后本页才会生成1252期或更后续目标的温度校准候选。"
    )

st.divider()
st.markdown("### 2. 建立下一期独立前瞻锁定")
st.write(
    "每个目标期固定生成三组各10注：原始V5概率、冻结T=2.875温度校准概率、"
    "均匀随机基准。原始V5与温度校准组使用相同模型、训练窗口、Ridge及共现强度，"
    "仅改变概率温度；三组种子和45维概率都会写入JSON档案。"
)
st.caption(
    "锁定后请立即下载并保存JSON。必须在目标期开奖前完成锁定；"
    "开奖后只上传手动更新的历史文件进行评分。单期结果仅作描述性记录，"
    "不会据此宣称预测优势。"
)

if "v77_v55_temperature_events" not in st.session_state:
    st.session_state["v77_v55_temperature_events"] = []

events = st.session_state["v77_v55_temperature_events"]
ready = bool(is_contiguous and prefix_audit.get("ok") and history_last > 1250)
if st.button(
    f"建立第{history_last + 1}期 T=2.875 研究锁定",
    type="primary",
    disabled=not ready,
    key="v77_build_temperature_lock",
):
    existing_targets = {
        int(item.get("payload", {}).get("metadata", {}).get("target_draw", -1))
        for item in events
        if item.get("event_type") == EVENT_TYPE
    }
    target = history_last + 1
    if target in existing_targets:
        st.error(f"第{target}期已有锁定事件。为保持前瞻记录不可替换，不允许重复建立。")
    else:
        try:
            event = build_v55_temperature_forward_event(
                history,
                manifest=commitment,
                ticket_count=10,
                master_seed=7755026,
            )
            st.session_state["v77_v55_temperature_events"] = [*events, event]
            st.success(
                f"已建立第{target}期独立研究锁定。"
                "它不改变正式选号，也不影响第1251期原始档案。"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"研究锁定失败：{type(exc).__name__}: {exc}")

events = st.session_state.get("v77_v55_temperature_events", [])
if events:
    st.markdown("#### 已保存的温度研究锁定")
    event_options = [
        item for item in events
        if item.get("event_type") == EVENT_TYPE
    ]
    for event in event_options:
        meta = event["payload"]["metadata"]
        st.write({
            "目标期": meta["target_draw"],
            "锁定历史末期": meta["history_last_draw"],
            "温度": meta["temperature"],
            "事件ID": event["event_id"],
            "每组注数": meta["ticket_count_per_strategy"],
        })
        ticket_frame = pd.DataFrame(event["payload"]["tickets"])
        st.dataframe(ticket_frame, hide_index=True, use_container_width=True)
    st.download_button(
        "下载全部V5.5温度前瞻锁定JSON",
        data=export_event_archive(event_options),
        file_name="lotto645_v55_temperature_forward_locks.json",
        mime="application/json",
        key="v77_download_temperature_locks",
    )

st.divider()
st.markdown("### 3. 开奖后导入锁定档案并评分")
archive_upload = st.file_uploader(
    "上传此前下载的 V5.5 温度前瞻锁定 JSON",
    type=["json"],
    key="v77_temperature_archive_upload",
)
if archive_upload is not None:
    try:
        imported = [
            event for event in import_event_archive(archive_upload.getvalue())
            if event.get("event_type") == EVENT_TYPE
        ]
        current = st.session_state.get("v77_v55_temperature_events", [])
        by_id = {event["event_id"]: event for event in current}
        for event in imported:
            by_id[event["event_id"]] = event
        st.session_state["v77_v55_temperature_events"] = list(by_id.values())
        st.success(f"已恢复 {len(imported)} 条V5.5温度研究锁定事件。")
    except Exception as exc:
        st.error(f"锁定档案导入失败：{type(exc).__name__}: {exc}")

events = st.session_state.get("v77_v55_temperature_events", [])
if events:
    try:
        summary, detail, status = evaluate_v55_temperature_forward_events(history, events)
        if not status.empty:
            st.markdown("#### 前瞻锁定状态")
            st.dataframe(status, hide_index=True, use_container_width=True)
        if not summary.empty:
            st.markdown("#### 已开奖逐期评分")
            st.dataframe(summary.round(6), hide_index=True, use_container_width=True)
            st.dataframe(detail, hide_index=True, use_container_width=True)
            st.download_button(
                "下载V5.5温度校准逐期评分CSV",
                data=summary.to_csv(index=False).encode("utf-8-sig"),
                file_name="lotto645_v55_temperature_forward_summary.csv",
                mime="text/csv",
                key="v77_download_temperature_summary",
            )
            st.download_button(
                "下载V5.5温度校准命中明细CSV",
                data=detail.to_csv(index=False).encode("utf-8-sig"),
                file_name="lotto645_v55_temperature_forward_detail.csv",
                mime="text/csv",
                key="v77_download_temperature_detail",
            )
        elif status.empty:
            st.info("尚未导入V5.5温度前瞻锁定档案。")
    except Exception as exc:
        st.error(f"前瞻评分失败：{type(exc).__name__}: {exc}")
else:
    st.info("导入以前保存的V5.5研究档案，或在1251期手动导入后建立下一期锁定。")

st.caption(
    "研究边界：校准候选仍未经独立前瞻验证；本页不修改V5正式选号，"
    "不联网获取开奖结果，不访问任何彩票官网或API。"
)
