#!/usr/bin/env python3
"""ScoreVision competition radar.

Tracks element lifecycle from the public manifest and, when the console API is
healthy, prints current leaderboards. Re-running this file gives a change diff.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


MANIFEST_INDEX = "https://turbo.scoredata.me/manifest/index.json"
CONSOLE_BASE = "https://console.scorevision.io/api/v2"


def get_json(url: str, retries: int = 5) -> Any:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:250]
            last_error = f"HTTP {exc.code}: {body}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(20, attempt * 3))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def get_text(url: str, retries: int = 5) -> str:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read().decode("utf-8", "replace")
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(20, attempt * 3))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def latest_manifest() -> tuple[str, dict[str, Any]]:
    index = get_json(MANIFEST_INDEX)
    if not isinstance(index, list) or not index:
        raise RuntimeError("manifest index is empty or invalid")
    key = sorted(str(item) for item in index if isinstance(item, str))[-1]
    url = urllib.parse.urljoin("https://turbo.scoredata.me/", key)
    return url, yaml.safe_load(get_text(url))


def norm_element(element: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": element.get("id"),
        "track": element.get("track"),
        "weight": element.get("weight"),
        "first_block": element.get("first_block"),
        "expiry_block": element.get("expiry_block"),
        "window_block": element.get("window_block"),
        "max_model_size_mb": element.get("max_model_size_mb"),
        "groundtruth_type": element.get("groundtruth_type"),
        "objects": element.get("objects") or [],
    }


def load_previous(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def print_manifest_diff(previous: dict[str, Any] | None, current: dict[str, Any]) -> None:
    current_by_id = {e["id"]: e for e in current["elements"] if e.get("id")}
    previous_by_id = {}
    if previous:
        previous_by_id = {e["id"]: e for e in previous.get("elements", []) if e.get("id")}

    new_ids = sorted(set(current_by_id) - set(previous_by_id))
    ended_ids = sorted(set(previous_by_id) - set(current_by_id))
    maybe_changed = sorted(
        eid for eid in set(current_by_id) & set(previous_by_id)
        if current_by_id[eid] != previous_by_id[eid]
    )

    print("== ELEMENT LIFECYCLE ==")
    if not previous:
        print("No previous snapshot. This run establishes the baseline.")
    if new_ids:
        print("NEW:", ", ".join(new_ids))
    if ended_ids:
        print("ENDED:", ", ".join(ended_ids))
    if maybe_changed:
        print("CHANGED:", ", ".join(maybe_changed))
    if not new_ids and not ended_ids and not maybe_changed and previous:
        print("No manifest element changes since previous run.")
    print()


def print_elements(elements: list[dict[str, Any]]) -> None:
    print("== ACTIVE MANIFEST ELEMENTS ==")
    for e in sorted(elements, key=lambda item: (str(item.get("track")), -float(item.get("weight") or 0), str(item.get("id")))):
        cap = e.get("max_model_size_mb")
        gt = e.get("groundtruth_type")
        objects = e.get("objects") or []
        print(
            f"{e.get('id')} track={e.get('track')} weight={e.get('weight')} "
            f"cap_mb={cap if cap is not None else '-'} gt={gt or '-'} "
            f"objects={','.join(map(str, objects[:8]))}"
        )
    print()


def console_element_detail(element_id: str, lookback_days: int) -> dict[str, Any] | None:
    encoded = urllib.parse.quote(element_id, safe="")
    url = f"{CONSOLE_BASE}/elements/{encoded}?lookback_days={lookback_days}"
    try:
        data = get_json(url, retries=3)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        print(f"[warn] console element API failed for {element_id}: {exc}")
        return None


def console_overview(lookback_days: int) -> dict[str, Any] | None:
    url = f"{CONSOLE_BASE}/overview?lookback_days={lookback_days}"
    try:
        data = get_json(url, retries=3)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        print(f"[warn] console overview API failed: {exc}")
        return None


def print_console_overview(overview: dict[str, Any] | None) -> None:
    if not overview:
        return
    print("== CONSOLE ACTIVE / END SIGNALS ==")
    for row in overview.get("activeElements") or []:
        try:
            current = float(row.get("currentScore") or 0)
            target = float(row.get("targetScore") or 0)
        except (TypeError, ValueError):
            current = 0.0
            target = 0.0
        gap = max(0.0, target - current)
        print(
            f"ACTIVE {row.get('id')} track={row.get('track')} "
            f"score={current*100:.1f}% target={target*100:.1f}% gap={gap*100:.1f}% "
            f"participants={row.get('participants')} winner={row.get('winnerHotkey')} "
            f"nextEval={row.get('nextEval') or '-'} timeToTarget={row.get('timeToTarget') or '-'}"
        )
    upcoming = overview.get("upcomingElements") or []
    if upcoming:
        print()
        for row in upcoming:
            print(
                f"UPCOMING {row.get('id')} track={row.get('track')} "
                f"objects={','.join(map(str, row.get('objects') or []))} "
                f"launchDate={row.get('launchDate') or '-'} nextEval={row.get('nextEval') or '-'}"
            )
    else:
        print("UPCOMING none published by API")
    completed = overview.get("completedElements") or []
    if completed:
        print()
        for row in completed[:8]:
            paid = row.get("totalPaid") or {}
            print(
                f"COMPLETED {row.get('id')} score={float(row.get('currentScore') or 0)*100:.1f}% "
                f"participants={row.get('participants')} timeToTarget={row.get('timeToTarget') or '-'} "
                f"paid_usd=${float(paid.get('usd') or 0):.2f} winner={row.get('winnerHotkey')}"
            )
    print()


def print_leaderboard(element_id: str, detail: dict[str, Any], top: int) -> None:
    rows = detail.get("leaderboard") or []
    print(f"== LEADERBOARD {element_id} ==")
    if not rows:
        print("No leaderboard data.")
        print()
        return
    for rank, row in enumerate(rows[:top], start=1):
        print(
            f"rank={rank:>2} score={float(row.get('score') or 0) * 100:>5.1f}% "
            f"latest={float(row.get('latestScore') or 0) * 100:>5.1f}% "
            f"usd=${float(row.get('usdEarned') or 0):>9.2f} "
            f"alpha={float(row.get('alphaEarned') or 0):>8.2f} "
            f"hotkey={row.get('hotkey')}"
        )
    print()


def print_win_read(element: dict[str, Any], detail: dict[str, Any] | None) -> None:
    if not detail:
        return
    rows = detail.get("leaderboard") or []
    if not rows:
        return
    scores = [float(row.get("score") or 0) for row in rows]
    best = max(scores)
    fifth = sorted(scores, reverse=True)[4] if len(scores) >= 5 else min(scores)
    weak_top = [row for row in rows[:10] if float(row.get("score") or 0) < 0.5]
    print(f"== WIN READ {element.get('id')} ==")
    print(f"best_score={best*100:.1f}% top5_cutoff={fifth*100:.1f}% competitors={len(rows)}")
    if weak_top:
        print(f"weak_top10_count={len(weak_top)}: top ranks exist with score < 50%, beatable if current scoring stays similar")
    if element.get("track") == "public":
        print("public strategy: fit under cap_mb, optimize threshold for map50 + false-positive, deploy first on new elements")
    else:
        print("private strategy: uptime + latency + correct action schema; no 30MB manifest cap")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=Path("score_miner_project/artifacts/competition_radar/latest.json"))
    parser.add_argument("--lookback-days", type=int, default=4)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--leaderboards", action="store_true")
    parser.add_argument("--overview", action="store_true")
    args = parser.parse_args()

    manifest_url, manifest = latest_manifest()
    elements = [norm_element(e) for e in manifest.get("elements", []) if isinstance(e, dict)]
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_url": manifest_url,
        "manifest_hash": manifest.get("manifest_hash") or manifest.get("hash"),
        "elements": elements,
    }

    previous = load_previous(args.state)
    print("manifest_url:", manifest_url)
    print("timestamp_utc:", snapshot["timestamp_utc"])
    print()
    print_manifest_diff(previous, snapshot)
    print_elements(elements)

    if args.overview or args.leaderboards:
        print_console_overview(console_overview(args.lookback_days))

    if args.leaderboards:
        for element in elements:
            detail = console_element_detail(str(element.get("id")), args.lookback_days)
            print_leaderboard(str(element.get("id")), detail or {}, args.top)
            print_win_read(element, detail)

    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    print("saved_state:", args.state)


if __name__ == "__main__":
    main()
