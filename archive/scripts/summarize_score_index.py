#!/usr/bin/env python3
"""Summarize ScoreVision public/private result shards from an index URL."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.parse import urlparse


DEFAULT_INDEX_URL = "https://turbo.scoredata.me/manako/index.json"


def load_loose_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes().decode("utf-8", "replace")
    raw = re.sub(r"[\x00-\x1f]", "", raw)
    return json.loads(raw)


def get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def key_to_url(index_url: str, key: str) -> str:
    if key.startswith("http://") or key.startswith("https://"):
        return key
    if key.startswith("manako/"):
        parsed = urlparse(index_url)
        return f"{parsed.scheme}://{parsed.netloc}/{key}"
    base = index_url.rsplit("/", 1)[0] + "/"
    return urljoin(base, key)


def build_hotkey_uid_map(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    data = load_loose_json(path)
    out = {}
    for row in data.get("uids", []):
        if isinstance(row, dict) and row.get("hotkey") is not None and row.get("uid") is not None:
            out[str(row["hotkey"])] = int(row["uid"])
    return out


def iter_records(blob: Any) -> list[dict[str, Any]]:
    if isinstance(blob, list):
        return [item for item in blob if isinstance(item, dict)]
    if isinstance(blob, dict):
        return [blob]
    return []


def extract_record(record: dict[str, Any]) -> dict[str, Any] | None:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else record
    if not isinstance(payload, dict):
        return None
    telemetry = payload.get("telemetry") or {}
    miner = telemetry.get("miner") or {}
    metrics = payload.get("metrics") or {}
    hotkey = miner.get("hotkey") or record.get("miner_hotkey") or payload.get("miner_hotkey")
    score = metrics.get("composite_score", payload.get("composite_score", record.get("score")))
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = None
    return {
        "element_id": payload.get("element_id") or record.get("element_id"),
        "window_id": payload.get("window_id") or record.get("window_id"),
        "hotkey": hotkey,
        "score": score_f,
        "latency_pass": payload.get("latency_pass"),
        "p95_latency_ms": payload.get("p95_latency_ms"),
        "task_id": telemetry.get("task_id") or payload.get("task_id") or record.get("task_id"),
        "slug": miner.get("slug"),
        "chute_id": miner.get("chute_id"),
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL)
    parser.add_argument("--element-contains", default="", help="Substring filter for index keys")
    parser.add_argument("--last", type=int, default=300, help="Number of matching keys from the end")
    parser.add_argument("--metagraph-json", type=Path, default=None)
    args = parser.parse_args()

    hotkey_to_uid = build_hotkey_uid_map(args.metagraph_json)
    index = get_json(args.index_url)
    if not isinstance(index, list):
        raise SystemExit("index is not a JSON list")
    keys = [key for key in index if isinstance(key, str)]
    if args.element_contains:
        keys = [key for key in keys if args.element_contains in key]
    keys = keys[-args.last :]

    rows = []
    failures = 0
    for key in keys:
        try:
            blob = get_json(key_to_url(args.index_url, key))
            for record in iter_records(blob):
                parsed = extract_record(record)
                if parsed:
                    parsed["key"] = key
                    rows.append(parsed)
        except Exception:
            failures += 1

    by_element: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_hotkey: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_element[str(row.get("element_id") or "")].append(row)
        if row.get("hotkey"):
            by_hotkey[str(row["hotkey"])].append(row)

    print("== SCORE INDEX SUMMARY ==")
    print("index_url:", args.index_url)
    print("filter:", args.element_contains or "<none>")
    print("keys_sampled:", len(keys))
    print("records:", len(rows))
    print("fetch_failures:", failures)
    print()

    print("== BY ELEMENT ==")
    for element_id, vals in sorted(by_element.items()):
        scores = [row["score"] for row in vals if row.get("score") is not None]
        nonzero = [score for score in scores if score > 0]
        print(
            f"{element_id or '<missing>'}: records={len(vals)} "
            f"mean={mean(scores):.5f} nonzero={len(nonzero)} max={max(scores or [0]):.5f}"
        )
    print()

    print("== TOP HOTKEYS BY MEAN SCORE ==")
    ranked = []
    for hotkey, vals in by_hotkey.items():
        scores = [row["score"] for row in vals if row.get("score") is not None]
        if not scores:
            continue
        ranked.append((mean(scores), max(scores), len(scores), hotkey, vals[-1]))
    ranked.sort(reverse=True)
    for avg, best, count, hotkey, last_row in ranked[:25]:
        uid = hotkey_to_uid.get(hotkey)
        uid_s = str(uid) if uid is not None else "?"
        print(
            f"uid={uid_s:>3} avg={avg:.5f} best={best:.5f} n={count:>3} "
            f"element={last_row.get('element_id')} hotkey={hotkey} slug={last_row.get('slug')}"
        )


if __name__ == "__main__":
    main()
