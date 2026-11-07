# -*- coding: utf-8 -*-
"""V4.0 append-only research event archive with optional GitHub branch persistence.

GitHub is used only to persist research metadata and lock artifacts. It is never
used to fetch lottery results. Draw history remains manual CSV/Excel only.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd

ARCHIVE_SCHEMA_VERSION = 1
EVENTS_PATH = "v40_store/events.jsonl"


class EventStoreError(RuntimeError):
    pass


def validate_event(event: dict[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "event_id", "event_type", "created_at_utc", "data_fingerprint_sha256", "payload"}
    if not isinstance(event, dict) or not required.issubset(event):
        raise ValueError("实验事件格式无效或缺少必要字段。")
    if event["schema_version"] != 1:
        raise ValueError("不支持的实验事件版本。")
    if not isinstance(event["payload"], dict):
        raise ValueError("实验事件 payload 必须是对象。")
    fingerprint = str(event["data_fingerprint_sha256"])
    if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint.lower()):
        raise ValueError("实验事件的数据指纹不是有效的 SHA-256。")
    return event


def export_event_archive(events: list[dict[str, Any]]) -> bytes:
    validated = [validate_event(event) for event in events]
    ids = [event["event_id"] for event in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("事件档案中存在重复 event_id。")
    body = {"archive_schema_version": ARCHIVE_SCHEMA_VERSION, "events": validated}
    return json.dumps(body, ensure_ascii=False, indent=2).encode("utf-8")


def import_event_archive(content: bytes) -> list[dict[str, Any]]:
    try:
        raw = json.loads(content.decode("utf-8"))
    except Exception as exc:
        raise ValueError("无法解析 V4.0 事件档案 JSON。") from exc
    if not isinstance(raw, dict) or raw.get("archive_schema_version") != ARCHIVE_SCHEMA_VERSION:
        raise ValueError("事件档案版本不兼容。")
    events = raw.get("events")
    if not isinstance(events, list):
        raise ValueError("事件档案缺少 events 列表。")
    validated = [validate_event(event) for event in events]
    ids = [event["event_id"] for event in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("导入的事件档案中有重复 event_id。")
    return validated


def latest_model_snapshot(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        event for event in events
        if event.get("event_type") == "model_snapshot"
        and isinstance(event.get("payload"), dict)
        and isinstance(event["payload"].get("selected_config"), dict)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda event: event.get("created_at_utc", ""))


def events_to_forward_locks(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    locks: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event_type") != "forward_lock":
            continue
        payload = event.get("payload", {})
        metadata = payload.get("metadata")
        records = payload.get("detail_records")
        if not isinstance(metadata, dict) or not isinstance(records, list):
            continue
        try:
            target_draw = int(metadata["target_draw"])
            history_last = int(metadata["history_last_draw"])
        except (KeyError, TypeError, ValueError):
            continue
        if target_draw <= history_last or not records:
            continue
        detail = pd.DataFrame(records)
        required = {"目标期号", "策略", "随机种子组", "配置编号", "注序", "号码", *{f"n{i}" for i in range(1, 7)}}
        if not required.issubset(detail.columns):
            continue
        if not (pd.to_numeric(detail["目标期号"], errors="coerce") == target_draw).all():
            continue
        valid = True
        for _, row in detail.iterrows():
            try:
                numbers = [int(row[f"n{i}"]) for i in range(1, 7)]
            except (TypeError, ValueError):
                valid = False
                break
            if len(set(numbers)) != 6 or any(number < 1 or number > 45 for number in numbers):
                valid = False
                break
        if valid:
            locks[str(target_draw)] = {"detail": detail, "metadata": metadata}
    return locks


class GitHubEventStore:
    """Persist append-only research records on a separate GitHub branch.

    Required token permission: repository Contents read/write. The configured
    branch is separate from the Streamlit deployment branch to avoid app redeploys.
    """

    def __init__(self, repository: str, token: str, branch: str = "research-data", timeout: int = 15):
        if not repository or "/" not in repository:
            raise ValueError("repository 必须是 owner/name。")
        if not token:
            raise ValueError("缺少 GitHub token。")
        self.repository = repository
        self.token = token
        self.branch = branch
        self.timeout = timeout
        self.api_root = f"https://api.github.com/repos/{repository}"

    def _request(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        url = self.api_root + endpoint
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "lotto645-v40-research-store",
        }
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                return response.status, json.loads(raw.decode("utf-8")) if raw else {}
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read().decode("utf-8"))
            except Exception:
                body = {}
            return exc.code, body
        except urllib.error.URLError as exc:
            raise EventStoreError(f"GitHub研究档案连接失败：{exc.reason}") from exc
        except TimeoutError as exc:
            raise EventStoreError("GitHub研究档案请求超时。") from exc

    def _ensure_branch(self) -> None:
        status, _ = self._request("GET", f"/branches/{urllib.parse.quote(self.branch, safe='')}")
        if status == 200:
            return
        if status != 404:
            raise EventStoreError(f"无法检查研究数据分支（HTTP {status}）。")
        status, main_ref = self._request("GET", "/git/ref/heads/main")
        if status != 200:
            raise EventStoreError(f"无法读取 main 基线（HTTP {status}）。")
        base_sha = main_ref.get("object", {}).get("sha")
        if not base_sha:
            raise EventStoreError("GitHub 未返回 main 基线 SHA。")
        status, body = self._request("POST", "/git/refs", {"ref": f"refs/heads/{self.branch}", "sha": base_sha})
        if status not in (201, 422):
            raise EventStoreError(f"无法创建独立研究数据分支（HTTP {status}）：{body.get('message', '')}")

    def _read_raw(self) -> tuple[list[dict[str, Any]], str | None, str]:
        self._ensure_branch()
        endpoint = f"/contents/{EVENTS_PATH}?ref={urllib.parse.quote(self.branch, safe='')}"
        status, body = self._request("GET", endpoint)
        if status == 404:
            return [], None, ""
        if status != 200:
            raise EventStoreError(f"无法读取研究事件档案（HTTP {status}）。")
        try:
            encoded = body.get("content", "").replace("\n", "")
            raw = base64.b64decode(encoded).decode("utf-8")
            events = [json.loads(line) for line in raw.splitlines() if line.strip()]
            for event in events:
                validate_event(event)
        except Exception as exc:
            raise EventStoreError("GitHub研究事件档案损坏或格式不兼容。") from exc
        return events, body.get("sha"), raw

    def load_events(self) -> list[dict[str, Any]]:
        events, _, _ = self._read_raw()
        return events

    def append_event(self, event: dict[str, Any]) -> bool:
        validate_event(event)
        for attempt in range(3):
            events, sha, old_text = self._read_raw()
            if any(item["event_id"] == event["event_id"] for item in events):
                return False
            new_text = old_text
            if new_text and not new_text.endswith("\n"):
                new_text += "\n"
            new_text += json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
            payload: dict[str, Any] = {
                "message": f"V4.0 research event: {event['event_type']}",
                "content": base64.b64encode(new_text.encode("utf-8")).decode("ascii"),
                "branch": self.branch,
            }
            if sha:
                payload["sha"] = sha
            endpoint = f"/contents/{EVENTS_PATH}"
            status, body = self._request("PUT", endpoint, payload)
            if status in (200, 201):
                return True
            if status in (409, 422) and attempt < 2:
                continue
            raise EventStoreError(f"保存研究档案失败（HTTP {status}）：{body.get('message', '')}")
        raise EventStoreError("保存研究档案时发生并发冲突，请重新尝试。")
