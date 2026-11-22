# -*- coding: utf-8 -*-
"""V8.7 immutable forward-lock and single-draw scoring protocol.

This module is offline. It only consumes manually supplied CSV/Excel history and
never accesses the lottery website or any lottery API.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from lotto_core import COLUMNS, NUMBER_COLS, validate_frame
from v50_predictive_engine import BASE_RATE, V50_ENGINE_VERSION, fit_predict_next, generate_tickets
from v81_batch_sampler import generate_batch_quota_portfolio
from v85_hybrid_portfolio import generate_probability_weighted_portfolio

TICKET_COUNT = 10
TICKET_SIZE = 6
PROTOCOL_VERSION = "8.7.0-immutable-forward-validation"


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalized_history_hash(frame: pd.DataFrame) -> str:
    clean = validate_frame(frame).reset_index(drop=True)
    records = clean[COLUMNS].to_dict(orient="records")
    return _sha256(_canonical_bytes({"rows": records}))


def _utc_timestamp(value: str | datetime | None = None) -> str:
    if value is None:
        stamp = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        stamp = value
    else:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("时间戳必须包含时区信息。")
    return stamp.astimezone(timezone.utc).isoformat(timespec="seconds")


def _derive_seed(master_seed: int, target_draw: int, stream: int) -> int:
    return int(np.random.SeedSequence(
        [int(master_seed), int(target_draw), int(stream)]
    ).generate_state(1)[0])


def _validate_tickets(tickets: list[list[int]], *, expected_count: int = TICKET_COUNT) -> list[list[int]]:
    if len(tickets) != int(expected_count):
        raise ValueError(f"每个策略必须恰好包含{expected_count}注。")
    normalized: list[list[int]] = []
    for ticket in tickets:
        values = [int(x) for x in ticket]
        if len(values) != TICKET_SIZE or len(set(values)) != TICKET_SIZE:
            raise ValueError("每注必须由6个互不重复号码组成。")
        if min(values) < 1 or max(values) > 45:
            raise ValueError("号码必须位于1—45。")
        normalized.append(sorted(values))
    if len({tuple(ticket) for ticket in normalized}) != len(normalized):
        raise ValueError("同一策略内存在重复组合。")
    return normalized


def _assert_target_not_committed(target_draw: int, commitment_dir: str | Path) -> None:
    root = Path(commitment_dir)
    if not root.exists():
        return
    existing = sorted(root.glob(f"draw_{int(target_draw)}*.json"))
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"第{target_draw}期已经存在锁定/研究档案（{names}）；V8.7拒绝覆盖或建立重复锁。"
        )


def prepare_v87_lock(
    history: pd.DataFrame,
    *,
    commitment_dir: str | Path,
    output_dir: str | Path,
    raw_csv_bytes: bytes | None = None,
    training_window: int = 500,
    min_history: int = 180,
    ridge_alpha: float = 1.0,
    master_seed: int = 87_2026,
    created_at_utc: str | datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, str]]:
    """Generate four frozen ten-ticket portfolios and a hash manifest.

    The function does not write to the commitment directory. Publication is a
    separate workflow action so that it can create a unique immutable Git ref.
    """
    clean = validate_frame(history).reset_index(drop=True)
    if clean.empty or int(clean["draw_no"].iloc[0]) != 1:
        raise ValueError("V8.7要求从第1期开始的手动历史数据。")
    draws = clean["draw_no"].to_numpy(dtype=int)
    if np.any(np.diff(draws) != 1):
        raise ValueError("历史期号必须连续；存在缺期时拒绝建立锁定。")
    if not 30 <= int(training_window) <= 1000:
        raise ValueError("training_window必须位于30—1000。")
    if len(clean) < int(min_history) + 30:
        raise ValueError(f"至少需要{int(min_history) + 30}期历史数据。")

    target_draw = int(draws[-1]) + 1
    _assert_target_not_committed(target_draw, commitment_dir)

    stamp = _utc_timestamp(created_at_utc)
    prediction = fit_predict_next(
        clean, training_window=int(training_window), min_history=int(min_history),
        ridge_alpha=float(ridge_alpha),
    )
    probability_frame = prediction["number_probabilities"].sort_values("号码")
    probabilities = probability_frame["预测入选概率"].to_numpy(dtype=float)
    pair_scores = np.asarray(prediction["pair_scores"], dtype=float)
    if probabilities.shape != (45,) or not np.isclose(probabilities.sum(), 6.0, atol=1e-6):
        raise RuntimeError("V5输出的45维概率无效或总和不为6。")

    seeds = {
        "V5": _derive_seed(master_seed, target_draw, 5001),
        "V8.1": _derive_seed(master_seed, target_draw, 8101),
        "V8.5": _derive_seed(master_seed, target_draw, 8501),
        "uniform": _derive_seed(master_seed, target_draw, 8700),
    }
    v5_tickets = _validate_tickets(generate_tickets(
        probabilities, pair_scores, TICKET_COUNT, seeds["V5"], pair_strength=0.5
    ))
    v81_tickets, v81_report = generate_batch_quota_portfolio(
        probabilities, ticket_count=TICKET_COUNT, sample_size=TICKET_SIZE, seed=seeds["V8.1"]
    )
    v81_tickets = _validate_tickets(v81_tickets)
    v85_tickets, v85_report = generate_probability_weighted_portfolio(
        probabilities, seed=seeds["V8.5"]
    )
    v85_tickets = _validate_tickets(v85_tickets)
    uniform_tickets = _validate_tickets(generate_tickets(
        np.full(45, BASE_RATE), None, TICKET_COUNT, seeds["uniform"], pair_strength=0.0
    ))

    groups = {
        "V5_raw_probability": {
            "tickets": v5_tickets,
            "seed": seeds["V5"],
            "sampler": "V5 sequential weighted sampling; pair_strength=0.5",
        },
        "V8.1_batch_quota": {
            "tickets": v81_tickets,
            "seed": seeds["V8.1"],
            "sampler": "dependent-rounded appearance quotas",
            "structure_diagnostics": v81_report,
        },
        "V8.5_low_overlap": {
            "tickets": v85_tickets,
            "seed": seeds["V8.5"],
            "sampler": "V8.1 randomized quotas with pairwise overlap <= 1",
            "structure_diagnostics": v85_report,
        },
        "uniform_random_control": {
            "tickets": uniform_tickets,
            "seed": seeds["uniform"],
            "sampler": "uniform random distinct six-number tickets",
            "theoretical_probability_per_ticket": 1.0 / 8_145_060.0,
        },
    }
    raw_bytes = raw_csv_bytes if raw_csv_bytes is not None else clean.to_csv(index=False).encode("utf-8")
    history_meta = {
        "first_draw": int(draws[0]),
        "last_draw": int(draws[-1]),
        "rows": int(len(clean)),
        "raw_csv_sha256": _sha256(raw_bytes),
        "normalized_history_sha256": _normalized_history_hash(clean),
        "last_draw_main_numbers": sorted(int(x) for x in clean.iloc[-1][NUMBER_COLS].tolist()),
        "last_draw_bonus": int(clean.iloc[-1]["bonus"]),
        "last_draw_date": str(clean.iloc[-1]["date"] or ""),
        "source": "manually supplied repository CSV only",
    }
    bundle: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "commitment_id": f"lotto645-v87-forward-lock-draw-{target_draw}",
        "event_type": "v87_forward_prediction_lock",
        "event_created_at_utc": stamp,
        "target_draw": target_draw,
        "history": history_meta,
        "model_configuration": {
            "marginal_model": V50_ENGINE_VERSION,
            "training_window": int(training_window),
            "min_history": int(min_history),
            "ridge_alpha": float(ridge_alpha),
            "probability_sum": float(probabilities.sum()),
            "ticket_count_per_group": TICKET_COUNT,
            "ticket_size": TICKET_SIZE,
            "master_seed": int(master_seed),
            "strategy_seeds": seeds,
        },
        "probabilities": {
            "number_order": list(range(1, 46)),
            "marginal_inclusion_probabilities": [float(x) for x in probabilities],
            "ranked_numbers_descending": [
                int(x) for x in probability_frame.sort_values(
                    ["预测入选概率", "号码"], ascending=[False, True]
                )["号码"].tolist()
            ],
        },
        "candidate_groups": groups,
        "evidence_boundary": {
            "research_only": True,
            "prediction_superiority_proven": False,
            "temperature_calibration_promoted": False,
            "production_model_changed": False,
            "random_control_included": True,
            "manual_history_only": True,
            "official_lottery_site_or_api_accessed": False,
            "notes": [
                "All strategies are locked before the target result is manually imported.",
                "One future draw is descriptive validation, not a statistically powered superiority test.",
                "The uniform control is generated with a fixed seed for auditability.",
            ],
        },
        "data_policy": "manual_csv_only_no_external_lottery_site_or_api_access",
    }
    bundle["canonical_payload_sha256"] = _sha256(_canonical_bytes(bundle))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidate_name = f"draw_{target_draw}_v87_candidates.json"
    manifest_name = f"draw_{target_draw}_v87_manifest.json"
    candidate_path = out / candidate_name
    manifest_path = out / manifest_name
    if candidate_path.exists() or manifest_path.exists():
        raise FileExistsError("输出目录已存在同一期V8.7文件，拒绝覆盖。")
    candidate_bytes = json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    manifest = {
        "manifest_schema_version": 1,
        "commitment_id": bundle["commitment_id"],
        "event_type": bundle["event_type"],
        "target_draw": target_draw,
        "event_created_at_utc": stamp,
        "candidate_file_name": candidate_name,
        "candidate_file_sha256": _sha256(candidate_bytes),
        "canonical_payload_sha256": bundle["canonical_payload_sha256"],
        "history": history_meta,
        "timestamp_provenance": {
            "event_timestamp_provenance": "generated_by_GitHub_Actions_workflow",
            "public_commitment_provenance": "candidate and manifest are committed together on a unique public forward-lock Git branch",
            "limitation": "The Git commit timestamp must be compared with the manually supplied target draw date; same-day ordering cannot be established from a date-only draw record.",
        },
        "data_policy": "manual_csv_only_no_external_lottery_site_or_api_access",
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    # Open in exclusive-create mode. A concurrent or repeated invocation fails closed.
    created_candidate = False
    created_manifest = False
    try:
        with candidate_path.open("xb") as stream:
            created_candidate = True
            stream.write(candidate_bytes)
        with manifest_path.open("xb") as stream:
            created_manifest = True
            stream.write(manifest_bytes)
    except Exception:
        # Remove only files created by this invocation; never delete a concurrent
        # lock that appeared after the initial existence check.
        if created_candidate:
            candidate_path.unlink(missing_ok=True)
        if created_manifest:
            manifest_path.unlink(missing_ok=True)
        raise
    return bundle, manifest, (str(candidate_path), str(manifest_path))


def _load_json(path: str | Path) -> tuple[dict[str, Any], bytes]:
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("JSON根节点必须是对象。")
    return payload, raw


def verify_v87_bundle(
    candidate_path: str | Path,
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle, raw = _load_json(candidate_path)
    manifest, _ = _load_json(manifest_path)
    if _sha256(raw) != manifest.get("candidate_file_sha256"):
        raise ValueError("候选文件SHA-256与锁定清单不一致。")
    claimed = bundle.get("canonical_payload_sha256")
    unsigned = dict(bundle)
    unsigned.pop("canonical_payload_sha256", None)
    actual = _sha256(_canonical_bytes(unsigned))
    if claimed != actual or claimed != manifest.get("canonical_payload_sha256"):
        raise ValueError("候选包规范化内容哈希校验失败。")
    if int(bundle.get("target_draw", -1)) != int(manifest.get("target_draw", -2)):
        raise ValueError("候选包与锁定清单的目标期号不一致。")
    if bundle.get("commitment_id") != manifest.get("commitment_id"):
        raise ValueError("候选包与锁定清单的commitment_id不一致。")
    return bundle, manifest


def load_legacy_v8x_archive(
    manifest_path: str | Path,
    archive_zip_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify and load the immutable 1252 V8.x archive without rewriting it."""
    manifest, _ = _load_json(manifest_path)
    zip_bytes = Path(archive_zip_path).read_bytes()
    commitment = manifest.get("archive_commitment", {})
    if _sha256(zip_bytes) != commitment.get("artifact_zip_sha256"):
        raise ValueError("V8.x归档ZIP SHA-256与既有1252清单不一致。")
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
        names = [name for name in archive.namelist() if name.endswith(".json")]
        if len(names) != 1:
            raise ValueError("V8.x归档必须且只能包含一个JSON候选包。")
        raw = archive.read(names[0])
    if _sha256(raw) != commitment.get("raw_file_sha256"):
        raise ValueError("V8.x候选JSON的原始SHA-256与既有清单不一致。")
    bundle = json.loads(raw.decode("utf-8-sig"))
    claimed = bundle.get("canonical_payload_sha256")
    unsigned = dict(bundle)
    unsigned.pop("canonical_payload_sha256", None)
    if _sha256(_canonical_bytes(unsigned)) != claimed:
        raise ValueError("V8.x候选包规范化哈希校验失败。")
    if claimed != commitment.get("canonical_payload_sha256"):
        raise ValueError("V8.x候选包规范化哈希与既有1252清单不一致。")
    if int(bundle.get("target_draw", -1)) != int(manifest.get("target_draw", -2)):
        raise ValueError("V8.x归档与既有清单的目标期号不一致。")
    if bundle.get("history", {}).get("last_draw") != manifest.get("history", {}).get("last_draw"):
        raise ValueError("V8.x归档训练截止期与既有清单不一致。")
    return bundle, manifest


def _timing_status(
    draw_date: str,
    commitment_timestamp_utc: str | None,
) -> dict[str, Any]:
    if not commitment_timestamp_utc:
        return {
            "status": "not_verified_no_public_commit_timestamp_supplied",
            "commitment_timestamp_utc": None,
            "target_draw_date": draw_date or None,
        }
    if not draw_date:
        return {
            "status": "unverifiable_target_draw_date_missing",
            "commitment_timestamp_utc": _utc_timestamp(commitment_timestamp_utc),
            "target_draw_date": None,
        }
    commit = datetime.fromisoformat(_utc_timestamp(commitment_timestamp_utc)).astimezone(
        ZoneInfo("Asia/Seoul")
    )
    target = pd.to_datetime(draw_date, errors="coerce")
    if pd.isna(target):
        return {
            "status": "unverifiable_target_draw_date_invalid",
            "commitment_timestamp_utc": _utc_timestamp(commitment_timestamp_utc),
            "target_draw_date": draw_date,
        }
    target_day = target.date()
    if commit.date() < target_day:
        status = "verified_commit_date_precedes_draw_date"
    elif commit.date() == target_day:
        status = "same_calendar_day_order_unverified"
    else:
        status = "commit_date_after_draw_date"
    return {
        "status": status,
        "commitment_timestamp_utc": _utc_timestamp(commitment_timestamp_utc),
        "commitment_date_kst": commit.date().isoformat(),
        "target_draw_date": target_day.isoformat(),
    }


def _score_groups(
    groups: dict[str, Any],
    actual_numbers: set[int],
    *,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, group in groups.items():
        tickets_raw = group.get("tickets", []) if isinstance(group, dict) else group
        tickets = _validate_tickets([[int(x) for x in ticket] for ticket in tickets_raw])
        hits = [len(set(ticket) & actual_numbers) for ticket in tickets]
        metrics[name] = {
            "ticket_count": len(tickets),
            "ticket_hits": hits,
            "hit_count_distribution": {str(k): int(hits.count(k)) for k in range(7)},
            "mean_hits_per_ticket": float(np.mean(hits)),
            "ticket_3plus_rate": float(np.mean(np.asarray(hits) >= 3)),
            "ticket_4plus_rate": float(np.mean(np.asarray(hits) >= 4)),
            "batch_at_least_3_hits": bool(any(value >= 3 for value in hits)),
            "batch_at_least_4_hits": bool(any(value >= 4 for value in hits)),
            "best_ticket_hits": int(max(hits)),
            "total_hits_across_tickets": int(sum(hits)),
            "number_coverage": int(len({n for ticket in tickets for n in ticket})),
        }
    if probabilities is not None:
        y = np.zeros(45, dtype=float)
        y[np.asarray(sorted(actual_numbers), dtype=int) - 1] = 1.0
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
        v5_prob = {
            "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
            "top6_hits": int(sum(y[i] > 0 for i in np.argsort(-p)[:6])),
            "top10_hits": int(sum(y[i] > 0 for i in np.argsort(-p)[:10])),
            "top15_hits": int(sum(y[i] > 0 for i in np.argsort(-p)[:15])),
            "uniform_expected_top6_hits": 0.8,
            "uniform_expected_top10_hits": 6.0 * 10.0 / 45.0,
            "uniform_expected_top15_hits": 2.0,
        }
    else:
        v5_prob = None
    uniform_key = next(
        (name for name in metrics if name.lower() == "uniform_random_control"),
        None,
    )
    uniform = metrics.get(uniform_key) if uniform_key else None
    comparisons = {}
    if uniform:
        for name, result in metrics.items():
            if name == uniform_key:
                continue
            comparisons[name] = {
                "mean_hits_difference_vs_uniform": float(
                    result["mean_hits_per_ticket"] - uniform["mean_hits_per_ticket"]
                ),
                "best_ticket_hits_difference_vs_uniform": int(
                    result["best_ticket_hits"] - uniform["best_ticket_hits"]
                ),
                "batch_3plus_difference_vs_uniform": int(
                    result["batch_at_least_3_hits"] - uniform["batch_at_least_3_hits"]
                ),
                "batch_4plus_difference_vs_uniform": int(
                    result["batch_at_least_4_hits"] - uniform["batch_at_least_4_hits"]
                ),
            }
    return {
        "strategy_metrics": metrics,
        "single_draw_differences_vs_uniform": comparisons,
        "v5_probability_metrics": v5_prob,
        "inference_boundary": (
            "单期表现仅为描述性观察；不能用一期开奖结果证明或否定预测优势。"
        ),
    }


def score_v87_lock(
    history_with_result: pd.DataFrame,
    candidate_path: str | Path,
    manifest_path: str | Path,
    *,
    commitment_timestamp_utc: str | None = None,
    scored_at_utc: str | datetime | None = None,
) -> dict[str, Any]:
    bundle, manifest = verify_v87_bundle(candidate_path, manifest_path)
    clean = validate_frame(history_with_result).reset_index(drop=True)
    target_draw = int(bundle["target_draw"])
    draws = clean["draw_no"].to_numpy(dtype=int)
    if clean.empty or int(draws[-1]) != target_draw:
        raise ValueError(
            f"评分要求手动数据最新期恰好为目标期{target_draw}；当前最新期为{int(draws[-1]) if len(draws) else '无'}。"
        )
    if np.any(np.diff(draws) != 1):
        raise ValueError("开奖后数据期号不连续，拒绝评分。")
    prefix = clean.iloc[:-1].reset_index(drop=True)
    expected_history = bundle["history"]
    if len(prefix) != int(expected_history["rows"]):
        raise ValueError("目标期开奖前的历史记录数与预测锁定不匹配。")
    if int(prefix["draw_no"].iloc[-1]) != int(expected_history["last_draw"]):
        raise ValueError("目标期开奖前的历史截止期与预测锁定不匹配。")
    if _normalized_history_hash(prefix) != expected_history["normalized_history_sha256"]:
        raise ValueError("目标期开奖前的完整历史指纹与预测锁定不匹配。")

    actual = set(int(x) for x in clean.iloc[-1][NUMBER_COLS].tolist())
    target_date = str(clean.iloc[-1]["date"] or "")
    probability_values = bundle["probabilities"]["marginal_inclusion_probabilities"]
    probabilities = np.asarray(probability_values, dtype=float)
    if probabilities.shape != (45,) or not np.isclose(probabilities.sum(), 6.0, atol=1e-6):
        raise ValueError("锁定包中的45维概率向量无效。")
    scored_at = _utc_timestamp(scored_at_utc)
    report = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "score_id": f"lotto645-v87-forward-score-draw-{target_draw}",
        "target_draw": target_draw,
        "commitment_id": bundle["commitment_id"],
        "candidate_file_sha256": manifest["candidate_file_sha256"],
        "canonical_payload_sha256": bundle["canonical_payload_sha256"],
        "score_created_at_utc": scored_at,
        "target_draw_date": target_date,
        "actual_main_numbers": sorted(actual),
        "actual_bonus_number": int(clean.iloc[-1]["bonus"]),
        "pre_draw_history_fingerprint_verified": True,
        "commitment_timing": _timing_status(target_date, commitment_timestamp_utc),
        "probability_forecast": {
            "source": "immutable_pre_draw_lock_bundle",
            "vector_length": 45,
            "marginal_inclusion_probabilities": [float(x) for x in probabilities.tolist()],
            "vector_sha256": _sha256(_canonical_bytes([float(x) for x in probabilities.tolist()])),
            "sum": float(probabilities.sum()),
        },
        **_score_groups(bundle["candidate_groups"], actual, probabilities=probabilities),
        "evidence_boundary": {
            "single_draw_result_is_not_statistical_superiority_evidence": True,
            "production_model_changed": False,
            "manual_history_only": True,
            "official_lottery_site_or_api_accessed": False,
        },
    }
    return report


def score_legacy_v8x_lock(
    history_with_result: pd.DataFrame,
    manifest_path: str | Path,
    archive_zip_path: str | Path,
    *,
    commitment_timestamp_utc: str | None = None,
    scored_at_utc: str | datetime | None = None,
) -> dict[str, Any]:
    bundle, manifest = load_legacy_v8x_archive(manifest_path, archive_zip_path)
    clean = validate_frame(history_with_result).reset_index(drop=True)
    target_draw = int(bundle["target_draw"])
    draws = clean["draw_no"].to_numpy(dtype=int)
    if clean.empty or int(draws[-1]) != target_draw:
        raise ValueError(
            f"评分要求手动数据最新期恰好为目标期{target_draw}；当前最新期为{int(draws[-1]) if len(draws) else '无'}。"
        )
    if np.any(np.diff(draws) != 1):
        raise ValueError("开奖后数据期号不连续，拒绝评分。")
    if int(bundle["history"]["last_draw"]) != target_draw - 1:
        raise ValueError("旧锁定档案的训练截止期与目标期不连续。")
    # The legacy manifest includes the final pre-target draw's manually supplied
    # numbers. Verify those fields even though its historical hash serialization
    # predates the V8.7 canonical history scheme.
    prior = clean.iloc[-2]
    old_last = manifest.get("history", {}).get(f"draw_{target_draw - 1}", {})
    if old_last:
        main = sorted(int(prior[column]) for column in NUMBER_COLS)
        if main != sorted(int(x) for x in old_last.get("main_numbers", [])):
            raise ValueError("旧锁定档案中的最后一期主号码与当前手动历史不一致。")
        if int(prior["bonus"]) != int(old_last.get("bonus", -1)):
            raise ValueError("旧锁定档案中的最后一期特别号与当前手动历史不一致。")

    actual = set(int(x) for x in clean.iloc[-1][NUMBER_COLS].tolist())
    target_date = str(clean.iloc[-1]["date"] or "")
    scored_at = _utc_timestamp(scored_at_utc)
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "score_id": f"lotto645-legacy-v8x-forward-score-draw-{target_draw}",
        "target_draw": target_draw,
        "commitment_id": bundle["commitment_id"],
        "candidate_archive_zip_sha256": manifest["archive_commitment"]["artifact_zip_sha256"],
        "canonical_payload_sha256": bundle["canonical_payload_sha256"],
        "score_created_at_utc": scored_at,
        "target_draw_date": target_date,
        "actual_main_numbers": sorted(actual),
        "actual_bonus_number": int(clean.iloc[-1]["bonus"]),
        "pre_draw_history_endpoint_verified": int(prior["draw_no"]) == target_draw - 1,
        "commitment_timing": _timing_status(target_date, commitment_timestamp_utc),
        **_score_groups(bundle["candidate_groups"], actual),
        "evidence_boundary": {
            "single_draw_result_is_not_statistical_superiority_evidence": True,
            "production_model_changed": False,
            "manual_history_only": True,
            "official_lottery_site_or_api_accessed": False,
            "legacy_history_fingerprint_full_recomputation": False,
        },
    }


def write_score_report(report: dict[str, Any], output_path: str | Path) -> str:
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"评分报告已存在，拒绝覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
    return str(path)
