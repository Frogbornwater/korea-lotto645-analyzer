# -*- coding: utf-8 -*-
"""Sync Lotto 6/45 draw history for GitHub Actions.

The Donghaeng JSON endpoints below are public response endpoints used by the
official result site, not a guaranteed/stable public API. A third-party mirror
is used only as a bootstrap/fallback source.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "lotto645.csv"
STATUS_PATH = DATA_DIR / "sync_status.json"
BASE = "https://www.dhlottery.co.kr/lt645"
LATEST_URL = f"{BASE}/selectPstLt645Info.do"
WINDOW_URL = f"{BASE}/selectPstLt645InfoNew.do"
MIRROR_ALL_URL = "https://smok95.github.io/lotto/results/all.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Lotto645Research/2.0)",
           "Accept": "application/json,text/plain,*/*"}

sys.path.insert(0, str(ROOT))
from lotto_core import COLUMNS, extract_records, merge_histories, quality_report, validate_frame  # noqa: E402


def get_json(url: str, params: dict | None = None) -> dict:
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=25)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("响应不是 JSON 对象")
            return data
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"请求失败（重试 3 次）：{last_error}")


def fetch_official_recent() -> tuple[pd.DataFrame, str]:
    latest_payload = get_json(LATEST_URL)
    latest_records = extract_records(latest_payload)
    if not latest_records:
        raise ValueError("官方最新期接口未返回可识别开奖记录")
    latest_no = max(r["draw_no"] for r in latest_records)
    window_payload = get_json(WINDOW_URL, {"srchDir": "center", "srchLtEpsd": latest_no})
    window_records = extract_records(window_payload)
    merged = merge_histories(pd.DataFrame(latest_records), pd.DataFrame(window_records))
    if merged.empty or int(merged["draw_no"].max()) != latest_no:
        raise ValueError("官方最新期与指定期次查询结果不一致")
    return merged, "Donghaeng Lotto result-page JSON"


def fetch_mirror_all() -> pd.DataFrame:
    payload = get_json(MIRROR_ALL_URL)
    records = extract_records(payload)
    frame = validate_frame(pd.DataFrame(records))
    if frame.empty:
        raise ValueError("第三方历史镜像没有返回有效记录")
    return frame


def fetch_official_full(latest_no: int) -> pd.DataFrame:
    """Fallback bootstrap: query public result-page windows in 10-round chunks."""
    records: dict[int, dict] = {}
    for center in range(6, latest_no + 1, 10):
        payload = get_json(WINDOW_URL, {"srchDir": "center", "srchLtEpsd": center})
        for item in extract_records(payload):
            records[item["draw_no"]] = item
        time.sleep(0.15)
    frame = validate_frame(pd.DataFrame([records[n] for n in sorted(records)]))
    if frame.empty:
        raise ValueError("官方历史分段查询没有返回有效记录")
    return frame


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame(columns=COLUMNS)
    existing = validate_frame(existing)
    errors: list[str] = []
    official = pd.DataFrame(columns=COLUMNS)
    mirror = pd.DataFrame(columns=COLUMNS)
    try:
        official, official_source = fetch_official_recent()
    except Exception as exc:
        official_source = "unavailable"
        errors.append(f"official_recent: {exc}")
    if existing.empty:
        try:
            mirror = fetch_mirror_all()
        except Exception as exc:
            errors.append(f"mirror_bootstrap: {exc}")
        # A recent official window is not enough for a first install; bootstrap the full history.
        if mirror.empty:
            try:
                if official.empty:
                    latest_payload = get_json(LATEST_URL)
                    latest_records = extract_records(latest_payload)
                    if not latest_records:
                        raise ValueError("官方最新期接口未返回有效期次")
                    latest_no = max(r["draw_no"] for r in latest_records)
                else:
                    latest_no = int(official["draw_no"].max())
                official = fetch_official_full(latest_no)
                official_source = "Donghaeng result-page JSON (paged bootstrap)"
            except Exception as full_exc:
                errors.append(f"official_full: {full_exc}")
        if mirror.empty and official.empty:
            raise RuntimeError("无法取得初始历史数据；" + "；".join(errors))
    elif official.empty:
        # On an ordinary scheduled run, fall back to the full mirror rather than
        # reporting success while leaving the tracked CSV silently unchanged.
        try:
            mirror = fetch_mirror_all()
        except Exception as exc:
            errors.append(f"mirror_fallback: {exc}")
            raise RuntimeError("官方最新数据源不可用，备用镜像也无法取得；保留旧 CSV，不报告同步成功。" + "；".join(errors))
    merged = merge_histories(existing, mirror, official)
    if merged.empty:
        raise RuntimeError("合并后没有有效开奖记录")
    report = quality_report(merged)
    # Require a complete round sequence after bootstrap; do not silently commit holes.
    if existing.empty and report["first_draw"] != 1:
        raise RuntimeError(f"初始历史数据未从第 1 期开始（当前起始期：{report['first_draw']}）；拒绝将不完整历史标记为成功")
    if existing.empty and report["missing_draws"]:
        raise RuntimeError(f"初始数据存在缺失期数（前20项）：{report['missing_draws'][:20]}")
    merged.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    status = {
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": official_source if not official.empty else "third-party mirror fallback",
        "row_count": int(len(merged)), "first_draw": report["first_draw"],
        "last_draw": report["last_draw"], "missing_draw_count": len(report["missing_draws"]),
        "warnings": errors,
    }
    STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
