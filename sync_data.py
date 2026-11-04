# -*- coding: utf-8 -*-
"""Fetch Lotto 6/45 history from Donghaeng's official result-page JSON.

These URLs are used by the official site frontend; they are not a guaranteed,
versioned public API. On failure this script preserves the old CSV and reports
failure; it never silently substitutes third-party records.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "lotto645.csv"
STATUS_PATH = DATA_DIR / "sync_status.json"
LATEST_URL = "https://dhlottery.co.kr/selectMainInfo.do"
ROUND_URL = "https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do"
OFFICIAL_RESULT_URL = "https://www.dhlottery.co.kr/lt645/result"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "X-Requested-With": "XMLHttpRequest",
    "AJAX": "true",
    "Referer": OFFICIAL_RESULT_URL,
}

sys.path.insert(0, str(ROOT))
from lotto_core import COLUMNS, extract_records, quality_report, validate_frame  # noqa: E402


def request_json(url: str, params: dict | None = None, retries: int = 4) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=25)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("官方返回内容不是 JSON 对象")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"请求官方数据失败：{last_error}")


def get_latest_round() -> int:
    records = extract_records(request_json(LATEST_URL))
    if not records:
        raise RuntimeError("官方最新期次接口未返回可识别的 Lotto 6/45 数据")
    return max(row["draw_no"] for row in records)


def fetch_round_window(center: int) -> list[dict]:
    payload = request_json(ROUND_URL, {"srchDir": "center", "srchLtEpsd": int(center)})
    return extract_records(payload)


def fetch_official_range(start_round: int, end_round: int) -> pd.DataFrame:
    """Fetch a range with overlapping windows and targeted repair for gaps."""
    if start_round < 1 or end_round < start_round:
        raise ValueError("无效的官方期次范围")
    by_round: dict[int, dict] = {}
    # Query centers every five rounds; overlap mitigates uncertain window edges.
    centers = list(range(start_round, end_round + 1, 5))
    for idx, center in enumerate(centers, start=1):
        for rec in fetch_round_window(center):
            if start_round <= rec["draw_no"] <= end_round:
                by_round[rec["draw_no"]] = rec
        if idx % 25 == 0:
            print(f"官方查询进度：{idx}/{len(centers)}，已收集 {len(by_round)} 期")
        time.sleep(0.12)

    missing = [n for n in range(start_round, end_round + 1) if n not in by_round]
    for draw_no in missing:
        for rec in fetch_round_window(draw_no):
            if start_round <= rec["draw_no"] <= end_round:
                by_round[rec["draw_no"]] = rec
        time.sleep(0.12)

    frame = validate_frame(pd.DataFrame([by_round[k] for k in sorted(by_round)], columns=COLUMNS))
    q = quality_report(frame)
    expected_count = end_round - start_round + 1
    if len(frame) != expected_count or q["first_draw"] != start_round or q["last_draw"] != end_round or q["missing_draws"]:
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
    try:
        latest = get_latest_round()
        existing, old_status = load_existing()
        old_quality = quality_report(existing)
        previously_verified_official = (
            old_status.get("status") == "success"
            and old_status.get("source_type") == "official_site_frontend_endpoint"
            and old_quality["sequence_complete"]
        )

        if not previously_verified_official or old_quality["last_draw"] is None:
            print("首次同步或历史来源未验证：完整抓取第 1 期至最新期。")
            official = fetch_official_range(1, latest)
            merged = official
            mode = "full_official_backfill"
        else:
            # Re-check a small overlap even when already current, and fetch all new issues.
            start = max(1, min(int(old_quality["last_draw"]) - 10, latest))
            delta = fetch_official_range(start, latest)
            merged = validate_frame(pd.concat([existing, delta], ignore_index=True))
            q = quality_report(merged)
            if not q["sequence_complete"] or q["last_draw"] != latest:
                # A gap means do not trust a partial incremental merge; retry full official backfill.
                print("增量核验发现缺口，转为完整官方历史重抓。")
                merged = fetch_official_range(1, latest)
                mode = "full_official_repair"
            else:
                mode = "incremental_official_refresh"

        quality = quality_report(merged)
        if not quality["sequence_complete"] or quality["last_draw"] != latest:
            raise RuntimeError("完整性最终核验失败，拒绝覆盖已有 CSV")
        # Only write data after all official-source validation has passed.
        merged.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        status = {
            "status": "success",
            "source": "Donghaeng Lottery official result-page JSON",
            "source_type": "official_site_frontend_endpoint",
            "source_url": OFFICIAL_RESULT_URL,
            "sync_mode": mode,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": quality["rows"], "first_draw": quality["first_draw"],
            "last_draw": quality["last_draw"], "missing_draw_count": len(quality["missing_draws"]),
            "complete_from_draw_1": True,
            "note": "仅官方来源；未使用第三方数据补齐。仅完整连续数据会写入 CSV。",
        }
        STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False, indent=2))
    except Exception as exc:
        status = {
            "status": "failed", "source": "Donghaeng official result-page JSON",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error": str(exc),
            "note": "同步失败时保留上一份历史 CSV，不使用第三方来源静默替代。",
        }
        STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False, indent=2))
        raise


if __name__ == "__main__":
    main()
