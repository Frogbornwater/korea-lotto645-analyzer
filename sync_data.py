# -*- coding: utf-8 -*-
"""Sync Lotto 6/45 draw history from Donghaeng's official result JSON.

The endpoints below are used by the official website frontend and are not a
versioned public API. This script deliberately throttles requests, validates
all rounds, and never replaces a valid CSV with a partial history.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "lotto645.csv"
STATUS_PATH = DATA_DIR / "sync_status.json"

# New official frontend endpoint: a center query returns an approximately
# 10-round window. Do not use selectPstLt645Info.do with a round parameter for
# historical lookup; that endpoint is for the latest draw, not arbitrary rounds.
BASE_URL = "https://www.dhlottery.co.kr/lt645"
LATEST_URL = f"{BASE_URL}/selectPstLt645Info.do"
ROUND_URL = f"{BASE_URL}/selectPstLt645InfoNew.do"
MAIN_INFO_URL = "https://www.dhlottery.co.kr/selectMainInfo.do"
OFFICIAL_RESULT_URL = f"{BASE_URL}/result"

REQUEST_DELAY_SECONDS = 1.25
CONNECT_TIMEOUT_SECONDS = 12
READ_TIMEOUT_SECONDS = 35
RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": OFFICIAL_RESULT_URL,
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
# Keep-alive lets successive round requests reuse a TCP/TLS connection.
SESSION.mount(
    "https://",
    HTTPAdapter(
        pool_connections=2,
        pool_maxsize=2,
        max_retries=Retry(total=0, connect=0, read=0, redirect=3, status=0),
    ),
)

sys.path.insert(0, str(ROOT))
from lotto_core import COLUMNS, extract_records, quality_report, validate_frame  # noqa: E402


def request_json(
    url: str,
    params: dict[str, Any] | None = None,
    retries: int = RETRIES,
) -> dict:
    """GET JSON with conservative backoff; distinguish HTTP and network errors."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = SESSION.get(
                url,
                params=params,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "json" not in content_type:
                excerpt = " ".join(response.text[:160].split())
                raise RuntimeError(
                    "官方接口返回的不是 JSON（可能是访问限制/排队页面）；"
                    f"Content-Type={content_type!r}，响应开头={excerpt!r}"
                )
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("官方返回内容不是 JSON 对象")
            return payload
        except requests.HTTPError as exc:
            last_error = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            # 4xx other than 429 are generally not fixed by retrying.
            if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                break
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
            # Parsing/shape errors are not transient network failures.
            break

        if attempt + 1 < retries:
            delay = (8, 20, 40)[min(attempt, 2)]
            print(
                f"官方接口暂时不可用，第 {attempt + 1}/{retries} 次失败；"
                f"{delay} 秒后重试：{last_error}",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(f"请求官方数据失败：{last_error}")


def get_latest_round() -> int:
    """Read the latest published draw from the official latest-result endpoint."""
    errors: list[str] = []
    try:
        records = extract_records(request_json(LATEST_URL))
        if records:
            return max(int(row["draw_no"]) for row in records)
        errors.append("selectPstLt645Info.do 返回 JSON，但未识别到开奖记录")
    except Exception as exc:
        errors.append(f"selectPstLt645Info.do: {exc}")

    # Secondary official frontend data endpoint, used only to identify the
    # latest draw. It is not used for historical per-round retrieval.
    try:
        records = extract_records(request_json(MAIN_INFO_URL))
        if records:
            return max(int(row["draw_no"]) for row in records)
        errors.append("selectMainInfo.do 返回 JSON，但未识别到开奖记录")
    except Exception as exc:
        errors.append(f"selectMainInfo.do: {exc}")

    raise RuntimeError("无法从 동행복권 官方接口识别最新期次。 " + " | ".join(errors))


def fetch_round_window(center: int) -> list[dict]:
    """Fetch the official ten-round history window centered on `center`."""
    params = {"srchDir": "center", "srchLtEpsd": int(center)}
    payload = request_json(ROUND_URL, params)
    records = extract_records(payload)
    if not records:
        raise RuntimeError(
            f"官方第 {center} 期附近接口返回 JSON，但 data.list 中没有可识别记录"
        )
    return records


def _merge_record(by_round: dict[int, dict], rec: dict, start_round: int, end_round: int) -> None:
    draw_no = int(rec["draw_no"])
    if not start_round <= draw_no <= end_round:
        return
    old = by_round.get(draw_no)
    if old is not None and old != rec:
        raise RuntimeError(f"官方接口对第 {draw_no} 期返回了冲突记录")
    by_round[draw_no] = rec


def _window_centers(start_round: int, end_round: int) -> list[int]:
    """The frontend returns 10 rounds around center; query non-overlapping windows.

    For example, center 6 covers approximately rounds 1-10, center 16 covers
    11-20, etc. Missing rounds are repaired individually after the main pass.
    """
    first_center = start_round + 5
    return list(range(first_center, end_round + 6, 10))


def fetch_official_range(start_round: int, end_round: int) -> pd.DataFrame:
    """Fetch, repair, and validate a complete official range with throttling."""
    if start_round < 1 or end_round < start_round:
        raise ValueError("无效的官方期次范围")

    by_round: dict[int, dict] = {}
    centers = _window_centers(start_round, end_round)
    total = len(centers)

    for idx, center in enumerate(centers, start=1):
        records = fetch_round_window(center)
        for rec in records:
            _merge_record(by_round, rec, start_round, end_round)
        print(
            f"官方历史同步进度：{idx}/{total} 个窗口；"
            f"已收集 {len(by_round)}/{end_round - start_round + 1} 期",
            flush=True,
        )
        if idx < total:
            time.sleep(REQUEST_DELAY_SECONDS)

    missing = [n for n in range(start_round, end_round + 1) if n not in by_round]
    if missing:
        print(f"发现 {len(missing)} 个缺失期次，开始逐期修复。", flush=True)
        for idx, draw_no in enumerate(missing, start=1):
            records = fetch_round_window(draw_no)
            for rec in records:
                _merge_record(by_round, rec, start_round, end_round)
            print(f"缺失期修复进度：{idx}/{len(missing)}，当前缺失 {sum(n not in by_round for n in missing)} 期", flush=True)
            if idx < len(missing):
                time.sleep(REQUEST_DELAY_SECONDS)

    ordered = [by_round[k] for k in sorted(by_round)]
    frame = validate_frame(pd.DataFrame(ordered, columns=COLUMNS))
    q = quality_report(frame)
    expected_count = end_round - start_round + 1
    if (
        len(frame) != expected_count
        or q["first_draw"] != start_round
        or q["last_draw"] != end_round
        or q["missing_draws"]
    ):
        raise RuntimeError(
            f"官方数据范围 {start_round}—{end_round} 不完整：实际 {len(frame)}/{expected_count} 期，"
            f"缺失示例 {q['missing_draws'][:30]}。"
        )
    return frame


def load_existing() -> tuple[pd.DataFrame, dict]:
    existing = pd.DataFrame(columns=COLUMNS)
    status: dict = {}
    if CSV_PATH.exists():
        existing = validate_frame(pd.read_csv(CSV_PATH))
    if STATUS_PATH.exists():
        try:
            status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            status = {}
    return existing, status


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("开始执行韩国 Lotto 6/45 官方历史数据同步。", flush=True)
    old_csv_exists = CSV_PATH.exists()
    try:
        print("正在查询官方最新已开奖期次……", flush=True)
        latest = get_latest_round()
        print(f"官方最新已开奖期次：第 {latest} 期。", flush=True)

        existing, old_status = load_existing()
        old_quality = quality_report(existing)
        previously_verified_official = (
            old_status.get("status") == "success"
            and old_status.get("source_type") == "official_site_frontend_endpoint"
            and old_quality["sequence_complete"]
        )

        if not previously_verified_official or old_quality["last_draw"] is None:
            print("首次同步或历史来源未验证：完整抓取第 1 期至最新期。", flush=True)
            merged = fetch_official_range(1, latest)
            mode = "full_official_backfill"
        else:
            start = max(1, min(int(old_quality["last_draw"]) - 10, latest))
            print(f"已有官方数据，增量核验第 {start} 期至第 {latest} 期。", flush=True)
            delta = fetch_official_range(start, latest)
            merged = validate_frame(pd.concat([existing, delta], ignore_index=True))
            q = quality_report(merged)
            if not q["sequence_complete"] or q["last_draw"] != latest:
                print("增量核验发现缺口，转为完整官方历史重抓。", flush=True)
                merged = fetch_official_range(1, latest)
                mode = "full_official_repair"
            else:
                mode = "incremental_official_refresh"

        quality = quality_report(merged)
        if not quality["sequence_complete"] or quality["last_draw"] != latest:
            raise RuntimeError("完整性最终核验失败，拒绝覆盖已有 CSV")

        temp_csv = CSV_PATH.with_suffix(".csv.tmp")
        merged.to_csv(temp_csv, index=False, encoding="utf-8-sig")
        temp_csv.replace(CSV_PATH)

        status = {
            "status": "success",
            "source": "Donghaeng Lottery official result-page JSON",
            "source_type": "official_site_frontend_endpoint",
            "source_url": OFFICIAL_RESULT_URL,
            "sync_mode": mode,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": quality["rows"],
            "first_draw": quality["first_draw"],
            "last_draw": quality["last_draw"],
            "missing_draw_count": len(quality["missing_draws"]),
            "duplicate_count": quality["duplicates"],
            "date_error_count": quality["date_errors"],
            "complete_from_draw_1": True,
            "note": "仅官方来源；未使用第三方数据补齐。仅完整连续数据会写入 CSV。",
        }
        STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        status = {
            "status": "failed",
            "source": "Donghaeng official result-page JSON",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error": str(exc),
            "previous_csv_preserved": bool(old_csv_exists and CSV_PATH.exists()),
            "note": "同步失败时保留上一份历史 CSV，不使用第三方来源静默替代。",
        }
        STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
        raise


if __name__ == "__main__":
    main()
