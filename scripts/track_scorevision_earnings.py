#!/usr/bin/env python3
"""Snapshot and compare ScoreVision miner earnings from the console V2 API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_URL = "https://console.scorevision.io/api/v2/miners?lookback_days={lookback_days}&limit={limit}"


def fetch_miners(lookback_days: int, limit: int, retries: int) -> list[dict[str, Any]]:
    url = API_URL.format(lookback_days=lookback_days, limit=limit)
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8", "replace"))
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            last_error = f"HTTP {exc.code}: {body}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(15, attempt * 3))
    else:
        raise RuntimeError(f"failed to fetch {url}: {last_error}")
    if not isinstance(data, list):
        raise RuntimeError("ScoreVision miners API did not return a list")
    return [row for row in data if isinstance(row, dict)]


def number(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_previous(out_dir: Path, current_name: str) -> dict[str, Any] | None:
    files = sorted(p for p in out_dir.glob("miners_*.json") if p.name != current_name)
    if not files:
        return None
    return json.loads(files[-1].read_text())


def by_hotkey(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in snapshot.get("miners", []):
        hotkey = row.get("hotkey")
        if isinstance(hotkey, str) and hotkey:
            out[hotkey] = row
    return out


def print_snapshot(snapshot: dict[str, Any], top: int) -> None:
    print("== CURRENT SCOREVISION API SNAPSHOT ==")
    print("timestamp_utc:", snapshot["timestamp_utc"])
    print("lookback_days:", snapshot["lookback_days"])
    print("miners:", len(snapshot["miners"]))
    print()
    for rank, row in enumerate(snapshot["miners"][:top], start=1):
        print(
            f"rank={rank:>2} usd=${number(row, 'usdEarned'):>10.2f} "
            f"alpha={number(row, 'alphaEarned'):>9.2f} tao={number(row, 'taoEarned'):>8.4f} "
            f"wins={int(number(row, 'wins')):>4} score={number(row, 'rollingScore') * 100:>5.1f}% "
            f"hotkey={row.get('hotkey')}"
        )


def print_delta(previous: dict[str, Any], current: dict[str, Any], top: int) -> None:
    prev_rows = by_hotkey(previous)
    current_ts = datetime.fromisoformat(current["timestamp_utc"])
    prev_ts = datetime.fromisoformat(previous["timestamp_utc"])
    hours = max((current_ts - prev_ts).total_seconds() / 3600.0, 1e-9)

    print()
    print("== DELTA VS PREVIOUS SNAPSHOT ==")
    print("previous_utc:", previous["timestamp_utc"])
    print("hours:", f"{hours:.3f}")
    print()

    deltas = []
    for row in current["miners"]:
        hotkey = row.get("hotkey")
        prev = prev_rows.get(hotkey)
        if not prev:
            continue
        usd_delta = number(row, "usdEarned") - number(prev, "usdEarned")
        alpha_delta = number(row, "alphaEarned") - number(prev, "alphaEarned")
        tao_delta = number(row, "taoEarned") - number(prev, "taoEarned")
        wins_delta = number(row, "wins") - number(prev, "wins")
        monthly_usd = usd_delta / hours * 24.0 * 30.0
        deltas.append((usd_delta, monthly_usd, alpha_delta, tao_delta, wins_delta, row))

    deltas.sort(reverse=True, key=lambda item: item[0])
    if not deltas:
        print("No shared hotkeys with previous snapshot yet. Run again later.")
        return
    for usd_delta, monthly_usd, alpha_delta, tao_delta, wins_delta, row in deltas[:top]:
        print(
            f"delta=${usd_delta:>9.2f} projected_month=${monthly_usd:>10.2f} "
            f"alpha_delta={alpha_delta:>8.2f} tao_delta={tao_delta:>8.4f} "
            f"wins_delta={wins_delta:>5.0f} hotkey={row.get('hotkey')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("score_miner_project/artifacts/scorevision_earnings"))
    parser.add_argument("--lookback-days", type=int, default=4)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    filename = f"miners_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    miners = fetch_miners(args.lookback_days, args.limit, args.retries)
    snapshot = {
        "timestamp_utc": now.isoformat(),
        "unix": time.time(),
        "lookback_days": args.lookback_days,
        "limit": args.limit,
        "miners": miners,
    }
    previous = load_previous(args.out_dir, filename)
    out = args.out_dir / filename
    out.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    print("saved:", out)
    print_snapshot(snapshot, args.top)
    if previous is not None:
        print_delta(previous, snapshot, args.top)


if __name__ == "__main__":
    main()
