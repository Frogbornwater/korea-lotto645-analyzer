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
    aggregate_v55_temperature_forward_evidence,
    load_commitment,
    persist_hash_only_commitment,
    verify_persisted_hash_only_commitment,
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
if "v77_v55_commit_anchors" not in st.session_state:
    st.session_state["v77_v55_commit_anchors"] = {}
if "v77_v55_anchor_verification" not in st.session_state:
    st.session_state["v77_v55_anchor_verification"] = {}

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
            # Commit only a SHA-256 manifest to GitHub. Ticket numbers and
            # probability vectors stay in the downloadable private JSON file.
            anchor_result = None
            try:
                token = str(st.secrets.get("GITHUB_RESEARCH_TOKEN", "")).strip()
            except Exception:
                token = ""
            if token:
                try:
                    anchor_result = persist_hash_only_commitment(event, token)
                    verification_result = verify_persisted_hash_only_commitment(event, token)
                    if verification_result.get("verified") and not verification_result.get("commit_time_utc"):
                        verification_result["commit_time_utc"] = anchor_result.get("commit_time_utc")
                        verification_result["commit_sha"] = anchor_result.get("commit_sha")
                        verification_result["manifest_url"] = anchor_result.get("url")
                    st.session_state["v77_v55_commit_anchors"][str(target)] = anchor_result
                    st.session_state["v77_v55_anchor_verification"][event["event_id"]] = verification_result
                    if not verification_result.get("verified"):
                        st.warning(
                            "GitHub返回了写入结果，但重新读取哈希承诺时没有通过核验；"
                            "该事件不会进入前瞻评分，需先查明原因。"
                        )
                except Exception as anchor_exc:
                    st.warning(
                        "研究锁已生成，但GitHub哈希时间锚创建失败；"
                        "请立即下载保存，当前记录不能独立证明早于开奖。"
                        f"原因：{type(anchor_exc).__name__}: {anchor_exc}"
                    )
            else:
                st.warning(
                    "研究锁已生成，但未配置GITHUB_RESEARCH_TOKEN；"
                    "本次只有本地下载档案，没有外部提交时间锚，不能作为严格的开奖前证据。"
                )
            st.session_state["v77_v55_temperature_events"] = [*events, event]
            if anchor_result:
                st.success(
                    f"第{target}期研究锁已生成，并已建立GitHub哈希承诺。"
                    "公开仓库只保存哈希，不包含号码与概率明细。"
                )
            else:
                st.info(f"第{target}期研究锁已生成，待下载保存并核实其前瞻时间。")
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
        anchor = st.session_state.get("v77_v55_commit_anchors", {}).get(str(meta["target_draw"]))
        if anchor:
            st.markdown(f"[查看第{meta['target_draw']}期公开哈希承诺]({anchor['url']})")
            if anchor.get("commit_time_utc"):
                st.caption(f"GitHub提交时间（UTC）：{anchor['commit_time_utc']}")
        ticket_frame = pd.DataFrame(event["payload"]["tickets"])
        st.dataframe(ticket_frame, hide_index=True, use_container_width=True)
        st.download_button(
            f"下载第{meta['target_draw']}期单期锁定JSON",
            data=export_event_archive([event]),
            file_name=f"lotto645_v55_temperature_draw_{meta['target_draw']}.json",
            mime="application/json",
            key=f"v77_download_temperature_lock_{meta['target_draw']}",
        )
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
        try:
            verifier_token = str(st.secrets.get("GITHUB_RESEARCH_TOKEN", "")).strip()
        except Exception:
            verifier_token = ""
        verification_cache = st.session_state.setdefault("v77_v55_anchor_verification", {})
        verified_events = []
        verification_rows = []
        for event in events:
            if event.get("event_type") != EVENT_TYPE:
                continue
            event_id = str(event.get("event_id", ""))
            verification = verification_cache.get(event_id)
            if not isinstance(verification, dict) or "verified" not in verification:
                verification = verify_persisted_hash_only_commitment(event, verifier_token)
                verification_cache[event_id] = verification
            meta = event.get("payload", {}).get("metadata", {})
            target_draw = int(meta.get("target_draw", -1))
            temporal_status = "哈希未通过：不纳入评分"
            if verification.get("verified"):
                verified_events.append(event)
                commit_time = verification.get("commit_time_utc")
                target_rows = history[history["draw_no"] == target_draw]
                if not commit_time:
                    temporal_status = "缺少GitHub提交时间，无法证明开奖前锁定"
                elif target_rows.empty:
                    temporal_status = "等待目标期开奖日期导入后核验时间先后"
                else:
                    draw_date = str(target_rows.iloc[0]["date"]).strip()
                    if not draw_date or draw_date.lower() in {"nan", "none"}:
                        temporal_status = "开奖日期缺失，无法确认锁定早于开奖"
                    else:
                        commit_timestamp = pd.Timestamp(commit_time)
                        if commit_timestamp.tzinfo is None:
                            commit_timestamp = commit_timestamp.tz_localize("UTC")
                        else:
                            commit_timestamp = commit_timestamp.tz_convert("UTC")
                        draw_day_start_utc = (
                            pd.Timestamp(draw_date)
                            .tz_localize("Asia/Seoul")
                            .tz_convert("UTC")
                        )
                        temporal_status = (
                            "确认GitHub提交早于开奖日开始"
                            if commit_timestamp < draw_day_start_utc
                            else "未能证明GitHub提交早于开奖日开始"
                        )
            verification_rows.append({
                "目标期": target_draw,
                "公开哈希承诺": "匹配" if verification.get("verified") else "未匹配",
                "GitHub提交时间UTC": verification.get("commit_time_utc"),
                "开奖时间先后": temporal_status,
                "承诺链接": verification.get("manifest_url"),
                "核验说明": verification.get("status"),
            })
        if verification_rows:
            st.markdown("#### 外部哈希承诺与时间顺序核验")
            st.dataframe(pd.DataFrame(verification_rows), hide_index=True, use_container_width=True)
        if len(verified_events) != len([x for x in events if x.get("event_type") == EVENT_TYPE]):
            st.warning("未通过公开GitHub哈希承诺的事件不会参与前瞻评分。")
        summary, detail, status = evaluate_v55_temperature_forward_events(history, verified_events)
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
            st.markdown("#### 多期配对Bootstrap与Holm校正")
            aggregate_report, aggregate_gate = aggregate_v55_temperature_forward_evidence(summary)
            st.write(aggregate_gate)
            if not aggregate_report.empty:
                st.dataframe(aggregate_report.round(8), hide_index=True, use_container_width=True)
                st.download_button(
                    "下载V5.5多期统计审计CSV",
                    data=aggregate_report.to_csv(index=False).encode("utf-8-sig"),
                    file_name="lotto645_v55_temperature_forward_evidence.csv",
                    mime="text/csv",
                    key="v77_download_temperature_evidence",
                )
            else:
                st.info(aggregate_gate.get("status", "多期前瞻样本尚未达到统计要求。"))
        elif status.empty and not verified_events:
            st.info("没有通过公开哈希承诺核验的锁定事件，因此暂不计算前瞻评分。")
        elif status.empty:
            st.info("尚无已开奖且通过历史指纹核验的V5.5温度锁定记录。")
    except Exception as exc:
        st.error(f"前瞻评分失败：{type(exc).__name__}: {exc}")
else:
    st.info("导入以前保存的V5.5研究档案，或在1251期手动导入后建立下一期锁定。")

st.caption(
    "研究边界：校准候选仍未经独立前瞻验证；本页不修改V5正式选号，"
    "不联网获取开奖结果，不访问任何彩票官网或API。"
)
