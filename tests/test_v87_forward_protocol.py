# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lotto_core import COLUMNS
from v87_forward_protocol import (
    _canonical_bytes,
    _timing_status,
    load_legacy_v8x_archive,
    prepare_v87_lock,
    score_legacy_v8x_lock,
    score_v87_lock,
    verify_v87_bundle,
)


def synthetic_history(n: int = 215, seed: int = 8701) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = datetime(2020, 1, 1)
    for draw in range(1, n + 1):
        nums = sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False))
        rows.append({
            "draw_no": draw,
            "date": (start + timedelta(days=7 * (draw - 1))).strftime("%Y-%m-%d"),
            "n1": nums[0], "n2": nums[1], "n3": nums[2],
            "n4": nums[3], "n5": nums[4], "n6": nums[5],
            "bonus": next(x for x in range(1, 46) if x not in nums),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def append_target(history: pd.DataFrame, draw_no: int = 216) -> pd.DataFrame:
    numbers = [1, 7, 14, 22, 33, 45]
    row = {
        "draw_no": draw_no,
        "date": "2024-02-01",
        **{f"n{i}": numbers[i - 1] for i in range(1, 7)},
        "bonus": 2,
    }
    return pd.concat([history, pd.DataFrame([row])], ignore_index=True)


class V87ForwardProtocolTests(unittest.TestCase):
    def _prepare(self, root: Path):
        history = synthetic_history()
        commit_dir = root / "commitments"
        out_dir = root / "out"
        bundle, manifest, paths = prepare_v87_lock(
            history,
            commitment_dir=commit_dir,
            output_dir=out_dir,
            created_at_utc="2024-01-01T00:00:00+00:00",
            master_seed=8702,
        )
        return history, bundle, manifest, paths, commit_dir, out_dir

    def test_prepare_creates_four_unique_ten_ticket_groups_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            history, bundle, manifest, paths, _, _ = self._prepare(Path(temp))
            self.assertEqual(bundle["target_draw"], 216)
            self.assertEqual(set(bundle["candidate_groups"]), {
                "V5_raw_probability", "V8.1_batch_quota",
                "V8.5_low_overlap", "uniform_random_control",
            })
            self.assertAlmostEqual(bundle["model_configuration"]["probability_sum"], 6.0, places=7)
            self.assertEqual(manifest["candidate_file_sha256"], hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest())
            for group in bundle["candidate_groups"].values():
                tickets = group["tickets"]
                self.assertEqual(len(tickets), 10)
                self.assertEqual(len({tuple(ticket) for ticket in tickets}), 10)
                self.assertTrue(all(len(ticket) == 6 and len(set(ticket)) == 6 for ticket in tickets))
            loaded, loaded_manifest = verify_v87_bundle(paths[0], paths[1])
            self.assertEqual(loaded["canonical_payload_sha256"], bundle["canonical_payload_sha256"])
            self.assertEqual(loaded_manifest["target_draw"], 216)

    def test_existing_target_lock_is_never_overwritten_or_duplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            commitment_dir = root / "commitments"
            commitment_dir.mkdir()
            (commitment_dir / "draw_216_research.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "已经存在锁定"):
                prepare_v87_lock(
                    synthetic_history(), commitment_dir=commitment_dir,
                    output_dir=root / "out", created_at_utc="2024-01-01T00:00:00+00:00",
                )

    def test_score_requires_exact_prefix_and_records_single_draw_descriptive_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            history, bundle, manifest, paths, _, _ = self._prepare(root)
            after = append_target(history)
            score = score_v87_lock(
                after, paths[0], paths[1],
                commitment_timestamp_utc="2024-01-15T00:00:00+00:00",
                scored_at_utc="2024-02-02T00:00:00+00:00",
            )
            self.assertEqual(score["target_draw"], 216)
            self.assertTrue(score["pre_draw_history_fingerprint_verified"])
            self.assertEqual(score["commitment_timing"]["status"], "verified_commit_date_precedes_draw_date")
            self.assertEqual(score["actual_main_numbers"], [1, 7, 14, 22, 33, 45])
            self.assertEqual(len(score["strategy_metrics"]["V5_raw_probability"]["ticket_hits"]), 10)
            forecast = score["probability_forecast"]
            self.assertEqual(forecast["source"], "immutable_pre_draw_lock_bundle")
            self.assertEqual(forecast["vector_length"], 45)
            self.assertEqual(len(forecast["marginal_inclusion_probabilities"]), 45)
            self.assertAlmostEqual(forecast["sum"], 6.0, places=6)
            self.assertEqual(
                forecast["vector_sha256"],
                hashlib.sha256(_canonical_bytes(forecast["marginal_inclusion_probabilities"])).hexdigest(),
            )
            self.assertTrue(score["evidence_boundary"]["single_draw_result_is_not_statistical_superiority_evidence"])

    def test_tampered_candidate_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, paths, _, _ = self._prepare(root)
            with open(paths[0], "ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_v87_bundle(paths[0], paths[1])

    def test_commitment_on_same_draw_date_is_not_claimed_as_pre_draw(self):
        same_day = _timing_status(
            "2024-02-01", "2024-02-01T00:00:00+00:00"
        )
        self.assertEqual(same_day["status"], "same_calendar_day_order_unverified")
        absent = _timing_status("2024-02-01", None)
        self.assertEqual(absent["status"], "not_verified_no_public_commit_timestamp_supplied")

    def test_legacy_1252_archive_is_hash_verified_and_scoreable_without_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            history = synthetic_history()
            actual = append_target(history)
            tickets = [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [19, 20, 21, 22, 23, 24],
                [25, 26, 27, 28, 29, 30],
                [31, 32, 33, 34, 35, 36],
                [37, 38, 39, 40, 41, 42],
                [1, 8, 15, 22, 29, 36],
                [2, 9, 16, 23, 30, 37],
                [3, 10, 17, 24, 31, 38],
            ]
            payload = {
                "schema_version": 1,
                "commitment_id": "legacy-test-draw-216",
                "event_type": "legacy_forward_archive",
                "created_at_utc": "2024-01-01T00:00:00+00:00",
                "target_draw": 216,
                "history": {
                    "first_draw": 1, "last_draw": 215, "rows": 215,
                    "draw_215": {
                        "main_numbers": sorted(int(x) for x in history.iloc[-1][["n1","n2","n3","n4","n5","n6"]]),
                        "bonus": int(history.iloc[-1]["bonus"]),
                    },
                },
                "candidate_groups": {
                    "V5_baseline": {"tickets": [[f"{n:02d}" for n in t] for t in tickets]},
                    "Uniform_random_control": {"tickets": [[f"{n:02d}" for n in t] for t in reversed(tickets)]},
                },
            }
            payload["canonical_payload_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
            raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            archive_path = root / "archive.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("legacy_candidates.json", raw)
            zip_bytes = archive_path.read_bytes()
            manifest = {
                "target_draw": 216,
                "commitment_id": payload["commitment_id"],
                "history": {
                    "first_draw": 1, "last_draw": 215, "rows": 215,
                    "draw_215": payload["history"]["draw_215"],
                },
                "archive_commitment": {
                    "raw_file_sha256": hashlib.sha256(raw).hexdigest(),
                    "canonical_payload_sha256": payload["canonical_payload_sha256"],
                    "artifact_zip_sha256": hashlib.sha256(zip_bytes).hexdigest(),
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded, _ = load_legacy_v8x_archive(manifest_path, archive_path)
            self.assertEqual(loaded["target_draw"], 216)
            score = score_legacy_v8x_lock(actual, manifest_path, archive_path)
            self.assertEqual(score["target_draw"], 216)
            self.assertTrue(score["pre_draw_history_endpoint_verified"])
            self.assertIn("V5_baseline", score["strategy_metrics"])

    def test_legacy_archive_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            zip_path = root / "tampered.zip"
            zip_path.write_bytes(b"not the original archive")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({
                "archive_commitment": {"artifact_zip_sha256": "0" * 64}
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ZIP SHA-256"):
                load_legacy_v8x_archive(manifest_path, zip_path)


if __name__ == "__main__":
    unittest.main()
