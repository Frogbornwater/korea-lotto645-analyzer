# -*- coding: utf-8 -*-
"""Research-only forward lock and scoring for the frozen V5.5 temperature candidate.

The frozen T=2.875 setting is anchored to the V5.5 artifact generated from
manual draws 1-1250. This module never fetches draw data or changes production
prediction settings. It deliberately refuses to create a second lock for draw
1251, which already has an immutable V5.1 lock.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lotto_core import COLUMNS, NUMBER_COLS, validate_frame
from v40_governance import data_fingerprint, make_event
from v40_store import validate_event
from v50_predictive_engine import fit_predict_next, generate_tickets
from scripts.v55_prequential_temperature_calibration import transform


ROOT = Path(__file__).resolve().parent
DEFAULT_COMMITMENT_PATH = ROOT / "data" / "forward_lock_commitments" / "v55_temperature_1250.json"
EVENT_TYPE = "v55_temperature_forward_lock"
RAW_STRATEGY = "V5原始概率（对照）"
TEMPERATURE_STRATEGY = "V5.5温度校准（T=2.875）"
RANDOM_STRATEGY = "均匀随机基准"
EXPECTED_TEMPERATURE = 2.875
EXPECTED_BASE_DRAW = 1250
EXPECTED_BASE_ROWS = 1250
EXPECTED_BASE_FINGERPRINT = "5529086e3d3401d20b778d94a62e41f3ee2bf0a886018e986ee1ecbe46ef1d11"
EXPECTED_LOCK_FILE_SHA256 = "f9937e49e72fdcb2627392e526e2176c932226122085e8adba4721f9edc0683c"
EXPECTED_ARTIFACT_SHA256 = "3616629c886aa166844a874c182f64e86e0ab4e1bf022f59a4ed4c0811d6404e"


def load_commitment(path: str | Path = DEFAULT_COMMITMENT_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    verify_commitment(manifest)
    return manifest


def verify_commitment(manifest: dict[str, Any]) -> dict[str, Any]:
    """Fail closed if the committed V5.5 lock provenance or frozen values drift."""
    if not isinstance(manifest, dict):
        raise ValueError("V5.5温度锁定承诺必须是JSON对象。")
    lock = manifest.get("temperature_lock")
    source = manifest.get("source_artifact")
    if not isinstance(lock, dict) or not isinstance(source, dict):
        raise ValueError("V5.5温度锁定承诺缺少参数或来源信息。")
    checks = {
        "lock_id": manifest.get("lock_id") == "lotto645-v55-temperature-forward-base-1250",
        "lock_version": lock.get("lock_version") == "v55-temperature-lock-2",
        "temperature": float(lock.get("temperature", -1)) == EXPECTED_TEMPERATURE,
        "base_last_draw": int(lock.get("base_last_draw", -1)) == EXPECTED_BASE_DRAW,
        "base_row_count": int(lock.get("base_row_count", -1)) == EXPECTED_BASE_ROWS,
        "history_fingerprint": lock.get("history_sha256") == EXPECTED_BASE_FINGERPRINT,
        "calibration_period": lock.get("calibration_period") == [1201, 1250],
        "calibration_draws": int(lock.get("calibration_draws", -1)) == 50,
        "training_window": int(lock.get("training_window", -1)) == 500,
        "ridge_alpha": float(lock.get("ridge_alpha", -1)) == 1.0,
        "lock_file_sha256": source.get("forward_temperature_lock_sha256") == EXPECTED_LOCK_FILE_SHA256,
        "artifact_sha256": source.get("artifact_zip_sha256") == EXPECTED_ARTIFACT_SHA256,
        "artifact_id": int(source.get("artifact_id", -1)) == 14349776279,
        "workflow_run_id": int(source.get("workflow_run_id", -1)) == 44129772806,
    }
    failed = [name for name, matched in checks.items() if not matched]
    if failed:
        raise ValueError("V5.5冻结温度承诺校验失败：" + ", ".join(failed))
    return {"valid": True, "checks": checks}


def numeric_history_fingerprint(history: pd.DataFrame) -> str:
    """Fingerprint draw IDs and results while ignoring optional/variable date text."""
    clean = validate_frame(history).reset_index(drop=True)
    if clean.empty:
        raise ValueError("历史数据为空，无法计算数值历史指纹。")
    numeric = clean[["draw_no", *NUMBER_COLS, "bonus"]].copy()
    encoded = numeric.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def check_frozen_base_history(
    history: pd.DataFrame,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify draws 1-1250 against the V5.5 base lock, ignoring date formatting."""
    commitment = manifest if manifest is not None else load_commitment()
    verify_commitment(commitment)
    try:
        clean = validate_frame(history).reset_index(drop=True)
        if clean.empty:
            return {"ok": False, "status": "历史数据为空", "history_last_draw": None}
        draws = clean["draw_no"].to_numpy(dtype=int)
        if int(draws[0]) != 1 or np.any(np.diff(draws) != 1):
            return {
                "ok": False,
                "status": "历史期号必须从第1期开始且连续，拒绝建立前瞻锁定",
                "history_last_draw": int(draws[-1]),
            }
        prefix = clean[clean["draw_no"] <= EXPECTED_BASE_DRAW].copy()
        if len(prefix) != EXPECTED_BASE_ROWS or int(prefix["draw_no"].iloc[-1]) != EXPECTED_BASE_DRAW:
            return {
                "ok": False,
                "status": "历史数据未完整覆盖1—1250期",
                "history_last_draw": int(draws[-1]),
                "prefix_rows": int(len(prefix)),
            }
        # The original V5.5 build read blank date cells with pandas, which
        # canonicalized them to the literal string "nan". Recreate that exact
        # canonical representation while ignoring any real dates in uploads.
        prefix.loc[:, "date"] = np.nan
        observed = data_fingerprint(prefix)
        expected = commitment["temperature_lock"]["history_sha256"]
        if observed != expected:
            return {
                "ok": False,
                "status": "1—1250期历史主号码/Bonus与V5.5冻结基准不一致",
                "history_last_draw": int(draws[-1]),
                "expected_fingerprint": expected,
                "actual_fingerprint": observed,
            }
        return {
            "ok": True,
            "status": "V5.5冻结基准1—1250期指纹一致",
            "history_last_draw": int(draws[-1]),
            "history_rows": int(len(clean)),
            "base_prefix_rows": int(len(prefix)),
            "base_fingerprint": observed,
        }
    except (ValueError, TypeError, KeyError) as exc:
        return {"ok": False, "status": f"历史数据校验失败：{exc}", "history_last_draw": None}


def _derived_seed(master_seed: int, target_draw: int, stream_id: int) -> int:
    return int(np.random.SeedSequence([int(master_seed), int(target_draw), int(stream_id)]).generate_state(1)[0])


def build_v55_temperature_forward_event(
    history: pd.DataFrame,
    *,
    manifest: dict[str, Any] | None = None,
    ticket_count: int = 10,
    master_seed: int = 7755026,
) -> dict[str, Any]:
    """Create a research-only paired lock for the next target draw after 1251."""
    commitment = manifest if manifest is not None else load_commitment()
    verify_commitment(commitment)
    prefix_audit = check_frozen_base_history(history, commitment)
    if not prefix_audit.get("ok"):
        raise ValueError(str(prefix_audit.get("status", "冻结历史校验失败")))
    clean = validate_frame(history).reset_index(drop=True)
    draws = clean["draw_no"].to_numpy(dtype=int)
    history_last = int(draws[-1])
    if history_last <= EXPECTED_BASE_DRAW:
        raise ValueError(
            "当前历史仍止于第1250期。必须先手动导入并核验第1251期，"
            "本研究轨道才可锁定第1252期或更后续目标；不会创建第二份1251期锁定。"
        )
    if not 1 <= int(ticket_count) <= 20:
        raise ValueError("每种策略的注数必须位于1—20。")

    prediction = fit_predict_next(
        clean,
        training_window=int(commitment["temperature_lock"]["training_window"]),
        min_history=180,
        ridge_alpha=float(commitment["temperature_lock"]["ridge_alpha"]),
    )
    raw_frame = prediction["number_probabilities"].sort_values("号码").reset_index(drop=True)
    number_ids = pd.to_numeric(raw_frame["号码"], errors="raise").to_numpy(dtype=int)
    raw_probabilities = pd.to_numeric(raw_frame["预测入选概率"], errors="raise").to_numpy(dtype=float)
    if not np.array_equal(number_ids, np.arange(1, 46)):
        raise ValueError("V5输出必须完整包含1—45号概率。")
    if not np.all(np.isfinite(raw_probabilities)) or np.any(raw_probabilities <= 0):
        raise ValueError("V5原始概率包含无效值。")
    if not np.isclose(float(raw_probabilities.sum()), 6.0, atol=1e-5):
        raise ValueError("V5原始概率总和不等于6。")

    temperature = float(commitment["temperature_lock"]["temperature"])
    calibrated_probabilities = transform(raw_probabilities, temperature)
    if not np.isclose(float(calibrated_probabilities.sum()), 6.0, atol=1e-5):
        raise ValueError("温度校准概率总和不等于6。")
    pair_scores = np.asarray(prediction["pair_scores"], dtype=float)
    target_draw = history_last + 1
    # Use common random numbers for raw and calibrated sampling so differences
    # are driven by the probability transform, not independent RNG streams.
    paired_model_seed = _derived_seed(master_seed, target_draw, 7701)
    random_seed = _derived_seed(master_seed, target_draw, 7703)

    raw_tickets = generate_tickets(
        raw_probabilities, pair_scores, int(ticket_count), paired_model_seed, pair_strength=0.5
    )
    calibrated_tickets = generate_tickets(
        calibrated_probabilities, pair_scores, int(ticket_count), paired_model_seed, pair_strength=0.5
    )
    random_tickets = generate_tickets(
        np.full(45, 6.0 / 45.0, dtype=float), None, int(ticket_count), random_seed, pair_strength=0.0
    )

    strategy_tickets = (
        (RAW_STRATEGY, raw_tickets, paired_model_seed),
        (TEMPERATURE_STRATEGY, calibrated_tickets, paired_model_seed),
        (RANDOM_STRATEGY, random_tickets, random_seed),
    )
    ticket_records: list[dict[str, Any]] = []
    for strategy, tickets, stream_seed in strategy_tickets:
        for index, ticket in enumerate(tickets, 1):
            record = {
                "目标期号": target_draw,
                "策略": strategy,
                "注序": index,
                "号码": " · ".join(f"{number:02d}" for number in ticket),
                "随机种子组": 1,
                "seed": int(stream_seed),
            }
            record.update({f"n{position}": int(number) for position, number in enumerate(ticket, 1)})
            ticket_records.append(record)

    probability_records = [
        {
            "号码": number,
            "raw_v5_probability": float(raw_probabilities[number - 1]),
            "temperature_probability": float(calibrated_probabilities[number - 1]),
            "uniform_probability": float(6.0 / 45.0),
        }
        for number in range(1, 46)
    ]
    history_fingerprint = numeric_history_fingerprint(clean)
    temperature_source = commitment["source_artifact"]
    payload = {
        "metadata": {
            "target_draw": target_draw,
            "history_first_draw": int(draws[0]),
            "history_last_draw": history_last,
            "history_rows": int(len(clean)),
            "history_numeric_fingerprint_sha256": history_fingerprint,
            "base_temperature_draw": EXPECTED_BASE_DRAW,
            "base_temperature_history_sha256": EXPECTED_BASE_FINGERPRINT,
            "temperature_lock_id": commitment["lock_id"],
            "temperature": temperature,
            "temperature_lock_file_sha256": temperature_source["forward_temperature_lock_sha256"],
            "source_artifact_zip_sha256": temperature_source["artifact_zip_sha256"],
            "engine_version": prediction["engine_version"],
            "training_window": int(prediction["training_window"]),
            "ridge_alpha": float(prediction["ridge_alpha"]),
            "pair_strength": 0.5,
            "ticket_count_per_strategy": int(ticket_count),
            "master_seed": int(master_seed),
            "raw_seed": paired_model_seed,
            "temperature_seed": paired_model_seed,
            "random_seed": random_seed,
        },
        "configuration": {
            "raw_strategy": RAW_STRATEGY,
            "temperature_strategy": TEMPERATURE_STRATEGY,
            "random_strategy": RANDOM_STRATEGY,
            "temperature": temperature,
            "temperature_base_draw": EXPECTED_BASE_DRAW,
            "training_window": int(prediction["training_window"]),
            "ridge_alpha": float(prediction["ridge_alpha"]),
            "pair_strength": 0.5,
            "ticket_count_per_strategy": int(ticket_count),
            "seed_derivation": "SeedSequence([master_seed, target_draw, stream_id])",
            "stream_ids": {"raw_v5_and_temperature_2_875_paired": 7701, "uniform_random": 7703},
            "paired_model_seed": paired_model_seed,
        },
        "number_probabilities": probability_records,
        "tickets": ticket_records,
        "research_only": True,
        "production_model_changed": False,
    }
    return make_event(EVENT_TYPE, history_fingerprint, payload)


def evaluate_v55_temperature_forward_events(
    history: pd.DataFrame,
    events: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score matching locks only after their target result is manually present."""
    clean = validate_frame(history).reset_index(drop=True)
    if clean.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    draws = clean["draw_no"].to_numpy(dtype=int)
    if int(draws[0]) != 1 or np.any(np.diff(draws) != 1):
        raise ValueError("前瞻评分要求从第1期开始的连续手动历史。")
    current_last = int(draws[-1])
    draw_map = {
        int(row["draw_no"]): {int(row[column]) for column in NUMBER_COLS}
        for _, row in clean.iterrows()
    }
    status_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    frozen_commitment = load_commitment()
    frozen_lock = frozen_commitment["temperature_lock"]
    frozen_source = frozen_commitment["source_artifact"]

    for raw_event in events:
        if not isinstance(raw_event, dict) or raw_event.get("event_type") != EVENT_TYPE:
            continue
        try:
            event = validate_event(raw_event)
            payload = event["payload"]
            metadata = payload["metadata"]
            target_draw = int(metadata["target_draw"])
            history_last = int(metadata["history_last_draw"])
            tickets = payload["tickets"]
        except (ValueError, KeyError, TypeError):
            status_rows.append({"目标期号": None, "状态": "锁定档案格式无效"})
            continue
        base = {
            "目标期号": target_draw,
            "锁定时历史末期": history_last,
            "温度参数": metadata.get("temperature"),
            "事件ID": event["event_id"],
        }
        expected_fingerprint = str(metadata.get("history_numeric_fingerprint_sha256", ""))
        provenance_matches = (
            float(metadata.get("temperature", -1)) == EXPECTED_TEMPERATURE
            and int(metadata.get("base_temperature_draw", -1)) == EXPECTED_BASE_DRAW
            and metadata.get("base_temperature_history_sha256") == EXPECTED_BASE_FINGERPRINT
            and metadata.get("temperature_lock_id") == frozen_commitment["lock_id"]
            and metadata.get("temperature_lock_file_sha256") == frozen_source["forward_temperature_lock_sha256"]
            and metadata.get("source_artifact_zip_sha256") == frozen_source["artifact_zip_sha256"]
            and int(metadata.get("training_window", -1)) == int(frozen_lock["training_window"])
            and float(metadata.get("ridge_alpha", -1)) == float(frozen_lock["ridge_alpha"])
            and float(metadata.get("pair_strength", -1)) == 0.5
            and payload.get("research_only") is True
            and payload.get("production_model_changed") is False
        )
        if not provenance_matches:
            status_rows.append({**base, "状态": "拒绝评分：冻结温度来源或配置被修改"})
            continue
        if target_draw != history_last + 1 or event.get("data_fingerprint_sha256") != expected_fingerprint:
            status_rows.append({**base, "状态": "拒绝评分：目标期或锁定事件指纹无效"})
            continue
        if current_last < history_last:
            status_rows.append({**base, "状态": "等待锁定历史数据"})
            continue
        prefix = clean[clean["draw_no"] <= history_last].copy()
        if len(prefix) != int(metadata.get("history_rows", -1)):
            status_rows.append({**base, "状态": "拒绝评分：锁定历史行数不匹配"})
            continue
        try:
            prefix_fingerprint = numeric_history_fingerprint(prefix)
        except ValueError:
            prefix_fingerprint = ""
        if prefix_fingerprint != expected_fingerprint:
            status_rows.append({**base, "状态": "拒绝评分：锁定历史号码前缀指纹不匹配"})
            continue
        if target_draw not in draw_map:
            status = "拒绝评分：当前历史跳过目标期号" if current_last >= target_draw else "等待开奖数据（锁定历史指纹已核验）"
            status_rows.append({**base, "状态": status})
            continue

        actual = draw_map[target_draw]
        if len(actual) != 6:
            status_rows.append({**base, "状态": "拒绝评分：目标期主号码无效"})
            continue
        probability_rows = payload.get("number_probabilities", [])
        try:
            probability_frame = pd.DataFrame(probability_rows).sort_values("号码")
            numbers = pd.to_numeric(probability_frame["号码"], errors="raise").to_numpy(dtype=int)
            raw_p = pd.to_numeric(probability_frame["raw_v5_probability"], errors="raise").to_numpy(dtype=float)
            temp_p = pd.to_numeric(probability_frame["temperature_probability"], errors="raise").to_numpy(dtype=float)
            uniform_p = pd.to_numeric(probability_frame["uniform_probability"], errors="raise").to_numpy(dtype=float)
            if len(numbers) != 45 or not np.array_equal(numbers, np.arange(1, 46)):
                raise ValueError("概率号码不完整")
            for vector in (raw_p, temp_p, uniform_p):
                if not np.all(np.isfinite(vector)) or np.any(vector <= 0) or not np.isclose(vector.sum(), 6.0, atol=1e-5):
                    raise ValueError("概率向量无效")
            expected_temp_p = transform(raw_p, EXPECTED_TEMPERATURE)
            expected_uniform_p = np.full(45, 6.0 / 45.0, dtype=float)
            if not np.allclose(temp_p, expected_temp_p, rtol=1e-9, atol=1e-11):
                raise ValueError("温度校准概率与冻结T=2.875不一致")
            if not np.allclose(uniform_p, expected_uniform_p, rtol=0.0, atol=1e-12):
                raise ValueError("均匀随机概率向量被修改")
        except (KeyError, TypeError, ValueError):
            status_rows.append({**base, "状态": "拒绝评分：45维概率记录无效"})
            continue

        groups: dict[str, list[int]] = {}
        ticket_seeds: dict[str, set[int]] = {}
        ticket_numbers: dict[str, set[tuple[int, ...]]] = {}
        valid = True
        for record in tickets:
            try:
                strategy = str(record["策略"])
                ticket = [int(record[f"n{i}"]) for i in range(1, 7)]
                ticket_no = int(record["注序"])
                ticket_seed = int(record["seed"])
                if len(set(ticket)) != 6 or any(value < 1 or value > 45 for value in ticket):
                    raise ValueError("号码无效")
                ticket_seeds.setdefault(strategy, set()).add(ticket_seed)
                ticket_numbers.setdefault(strategy, set()).add(tuple(sorted(ticket)))
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            hits = sorted(set(ticket) & actual)
            groups.setdefault(strategy, []).append(len(hits))
            detail_rows.append({
                "目标期号": target_draw,
                "策略": strategy,
                "注序": ticket_no,
                "号码": " · ".join(f"{value:02d}" for value in sorted(ticket)),
                "命中号码": " · ".join(f"{value:02d}" for value in hits),
                "命中数": len(hits),
                "命中3个以上": len(hits) >= 3,
            })
        expected_groups = {RAW_STRATEGY, TEMPERATURE_STRATEGY, RANDOM_STRATEGY}
        count = int(metadata.get("ticket_count_per_strategy", 0))
        paired_seeds_match = (
            len(ticket_seeds.get(RAW_STRATEGY, set())) == 1
            and ticket_seeds.get(RAW_STRATEGY) == ticket_seeds.get(TEMPERATURE_STRATEGY)
            and len(ticket_seeds.get(RANDOM_STRATEGY, set())) == 1
        )
        no_duplicate_tickets = all(
            len(ticket_numbers.get(name, set())) == count for name in expected_groups
        )
        if (
            not valid
            or set(groups) != expected_groups
            or any(len(groups[name]) != count for name in expected_groups)
            or not paired_seeds_match
            or not no_duplicate_tickets
        ):
            status_rows.append({**base, "状态": "拒绝评分：三组锁定号码、配对随机种子或去重完整性无效"})
            continue

        actual_vector = np.zeros(45, dtype=float)
        actual_vector[np.asarray(sorted(actual), dtype=int) - 1] = 1.0
        def probability_metrics(vector: np.ndarray) -> tuple[float, float]:
            clipped = np.clip(vector, 1e-9, 1.0 - 1e-9)
            brier = float(np.mean((clipped - actual_vector) ** 2))
            logloss = float(-np.mean(actual_vector * np.log(clipped) + (1.0 - actual_vector) * np.log(1.0 - clipped)))
            return brier, logloss
        raw_brier, raw_logloss = probability_metrics(raw_p)
        temp_brier, temp_logloss = probability_metrics(temp_p)
        uniform_brier, uniform_logloss = probability_metrics(uniform_p)

        def ticket_metrics(values: list[int]) -> dict[str, float]:
            arr = np.asarray(values, dtype=float)
            return {
                "mean_hits": float(arr.mean()),
                "ticket_3plus": float(np.mean(arr >= 3)),
                "ticket_4plus": float(np.mean(arr >= 4)),
                "batch_3plus": float(np.any(arr >= 3)),
                "batch_4plus": float(np.any(arr >= 4)),
                "best_hits": float(arr.max()),
            }
        raw_m = ticket_metrics(groups[RAW_STRATEGY])
        temp_m = ticket_metrics(groups[TEMPERATURE_STRATEGY])
        random_m = ticket_metrics(groups[RANDOM_STRATEGY])
        summary_rows.append({
            "目标期号": target_draw,
            "锁定历史末期": history_last,
            "温度参数": float(metadata["temperature"]),
            "每组注数": count,
            "V5原始概率平均命中": raw_m["mean_hits"],
            "V5.5校准平均命中": temp_m["mean_hits"],
            "随机平均命中": random_m["mean_hits"],
            "V5原始概率单注3+率": raw_m["ticket_3plus"],
            "V5.5校准单注3+率": temp_m["ticket_3plus"],
            "随机单注3+率": random_m["ticket_3plus"],
            "V5原始概率整批3+": raw_m["batch_3plus"],
            "V5.5校准整批3+": temp_m["batch_3plus"],
            "随机整批3+": random_m["batch_3plus"],
            "V5原始Brier": raw_brier,
            "V5.5校准Brier": temp_brier,
            "随机Brier": uniform_brier,
            "V5原始LogLoss": raw_logloss,
            "V5.5校准LogLoss": temp_logloss,
            "随机LogLoss": uniform_logloss,
            "校准减原始Brier": temp_brier - raw_brier,
            "校准减原始LogLoss": temp_logloss - raw_logloss,
        })
        status_rows.append({**base, "状态": "已评分"})

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows), pd.DataFrame(status_rows)


def build_hash_only_commitment(event: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Build a public manifest that anchors an archive hash without exposing tickets."""
    from v40_store import export_event_archive

    validated = validate_event(event)
    if validated.get("event_type") != EVENT_TYPE:
        raise ValueError("只允许为V5.5温度前瞻事件建立哈希承诺。")
    payload = validated["payload"]
    metadata = payload["metadata"]
    archive_bytes = export_event_archive([validated])
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    manifest = {
        "commitment_schema_version": 1,
        "commitment_id": f"lotto645-v55-temperature-draw-{int(metadata['target_draw'])}",
        "event_id": validated["event_id"],
        "event_type": EVENT_TYPE,
        "event_created_at_utc": validated["created_at_utc"],
        "target_draw": int(metadata["target_draw"]),
        "history": {
            "first_draw": int(metadata["history_first_draw"]),
            "last_draw": int(metadata["history_last_draw"]),
            "rows": int(metadata["history_rows"]),
            "numeric_fingerprint_sha256": str(metadata["history_numeric_fingerprint_sha256"]),
        },
        "temperature_lock": {
            "lock_id": str(metadata["temperature_lock_id"]),
            "temperature": float(metadata["temperature"]),
            "base_last_draw": EXPECTED_BASE_DRAW,
            "base_history_sha256": EXPECTED_BASE_FINGERPRINT,
            "source_lock_file_sha256": str(metadata["temperature_lock_file_sha256"]),
            "source_artifact_zip_sha256": str(metadata["source_artifact_zip_sha256"]),
        },
        "archive_commitment": {
            "file_name": f"lotto645_v55_temperature_draw_{int(metadata['target_draw'])}.json",
            "sha256": archive_sha,
            "ticket_numbers_embedded": False,
            "probability_vectors_embedded": False,
        },
        "provenance_limit": (
            "This GitHub commit anchors the exact lock archive bytes no later than the commit time. "
            "To establish that the lock predates the actual draw, compare the GitHub commit time "
            "with the target draw date in the manually imported history."
        ),
    }
    return manifest, archive_sha


def _github_json_request(
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    import urllib.error
    import urllib.request

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "lotto645-v55-hash-only-lock",
    }
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read()
            return int(response.status), json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            body = {}
        return int(exc.code), body
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"GitHub哈希承诺连接失败：{exc}") from exc


def persist_hash_only_commitment(
    event: dict[str, Any],
    token: str,
    *,
    repository: str = "Frogbornwater/korea-lotto645-analyzer",
    branch: str = "forward-lock-commitments",
) -> dict[str, Any]:
    """Commit only the hash manifest to a separate public branch; never upload tickets."""
    import base64
    import urllib.parse

    if not token:
        raise ValueError("缺少GITHUB_RESEARCH_TOKEN，无法创建外部时间锚。")
    manifest, archive_sha = build_hash_only_commitment(event)
    owner, name = repository.split("/", 1)
    api_root = f"https://api.github.com/repos/{owner}/{name}"
    branch_encoded = urllib.parse.quote(branch, safe="")
    branch_status, branch_data = _github_json_request(
        "GET", f"{api_root}/git/ref/heads/{branch_encoded}", token
    )
    if branch_status == 404:
        main_status, main_data = _github_json_request(
            "GET", f"{api_root}/git/ref/heads/main", token
        )
        if main_status != 200:
            raise RuntimeError("无法读取main分支，不能创建哈希承诺分支。")
        create_status, create_data = _github_json_request(
            "POST",
            f"{api_root}/git/refs",
            token,
            {"ref": f"refs/heads/{branch}", "sha": main_data["object"]["sha"]},
        )
        if create_status not in (201, 422):
            raise RuntimeError(f"创建哈希承诺分支失败：HTTP {create_status}")
        branch_status, branch_data = _github_json_request(
            "GET", f"{api_root}/git/ref/heads/{branch_encoded}", token
        )
    if branch_status != 200:
        raise RuntimeError(f"读取哈希承诺分支失败：HTTP {branch_status}")

    target_draw = int(event["payload"]["metadata"]["target_draw"])
    path = f"forward-lock-commitments/draw_{target_draw}.json"
    path_encoded = urllib.parse.quote(path, safe="/")
    existing_status, existing_data = _github_json_request(
        "GET", f"{api_root}/contents/{path_encoded}?ref={branch_encoded}", token
    )
    if existing_status == 200:
        try:
            existing_bytes = base64.b64decode(existing_data.get("content", ""))
            existing_manifest = json.loads(existing_bytes.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("目标期哈希承诺文件已存在且无法验证，拒绝覆盖。") from exc
        if existing_manifest.get("archive_commitment", {}).get("sha256") == archive_sha:
            commit_sha = existing_data.get("sha", "")
            return {
                "status": "already_committed",
                "branch": branch,
                "path": path,
                "commit_sha": commit_sha,
                "archive_sha256": archive_sha,
                "url": f"https://github.com/{repository}/blob/{branch}/{path}",
            }
        raise RuntimeError(f"第{target_draw}期已有不同哈希承诺，拒绝覆盖。")
    if existing_status != 404:
        raise RuntimeError(f"检查目标期承诺文件失败：HTTP {existing_status}")

    content = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    put_status, put_data = _github_json_request(
        "PUT",
        f"{api_root}/contents/{path_encoded}",
        token,
        {
            "message": f"Commit hash-only V5.5 temperature lock for draw {target_draw}",
            "content": base64.b64encode(content).decode("ascii"),
            "branch": branch,
        },
    )
    if put_status not in (200, 201):
        raise RuntimeError(f"写入哈希承诺失败：HTTP {put_status} {put_data.get('message', '')}")
    commit_sha = str(put_data.get("commit", {}).get("sha", ""))
    commit_date = None
    if commit_sha:
        commit_status, commit_data = _github_json_request(
            "GET", f"{api_root}/commits/{commit_sha}", token
        )
        if commit_status == 200:
            commit_date = (
                commit_data.get("commit", {}).get("committer", {}).get("date")
                or commit_data.get("commit", {}).get("author", {}).get("date")
            )
    return {
        "status": "committed",
        "branch": branch,
        "path": path,
        "commit_sha": commit_sha,
        "commit_time_utc": commit_date,
        "archive_sha256": archive_sha,
        "url": f"https://github.com/{repository}/blob/{branch}/{path}",
    }
