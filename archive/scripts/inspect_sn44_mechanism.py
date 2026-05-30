#!/usr/bin/env python3
"""Inspect SN44 mechanism rewards and miner element registrations.

Run this from the TurboVision repo with:

    SCOREVISION_MECHID=1 uv run --python python3.13 \
      python ../score_miner_project/scripts/inspect_sn44_mechanism.py \
      --json sn44_m1.json --online
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any


PRIVATE_ELEMENTS = (
    "manako/DetectFootballEvent",
    "manako/DetectCricketDelivery",
)

PUBLIC_ELEMENTS = (
    "manak0/Detect-crime",
    "manak0/Detect-fire",
    "manak0/Detect-beverage-detect",
    "manak0/Detect-car-wash",
)


def load_loose_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes().decode("utf-8", "replace")
    raw = re.sub(r"[\x00-\x1f]", "", raw)
    return json.loads(raw)


def as_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def parse_local_metagraph(path: Path, top_n: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = load_loose_json(path)
    owner_coldkey = data.get("owner")
    rows = [
        row
        for row in data.get("uids", [])
        if isinstance(row, dict) and row.get("coldkey") != owner_coldkey
    ]
    rows.sort(key=lambda row: as_float(row.get("incentive")), reverse=True)

    print("== LOCAL METAGRAPH ==")
    print("netuid:", data.get("netuid"))
    print("mechanism_id:", data.get("mechanism_id"))
    print("mechanism_count:", data.get("mechanism_count"))
    print("rate:", data.get("rate"))
    print("tempo:", data.get("tempo"))
    print("registration_cost:", data.get("registration_cost"))
    print("non_owner_uids:", len(rows))
    print()
    print(f"== TOP {top_n} BY INCENTIVE ==")
    for row in rows[:top_n]:
        print(
            "uid={uid:>3} incentive={inc:.6f} emissions={emis:.6f} "
            "stake={stake:.2f} hotkey={hotkey} identity={identity}".format(
                uid=row.get("uid"),
                inc=as_float(row.get("incentive")),
                emis=as_float(row.get("emissions")),
                stake=as_float(row.get("stake")),
                hotkey=row.get("hotkey"),
                identity=row.get("identity") or "~",
            )
        )
    print()
    return data, rows


def parse_commit_payload(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, str):
        return None
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def inspect_online(top_rows: list[dict[str, Any]], include_public: bool) -> None:
    from scorevision.utils.bittensor_helpers import get_subtensor
    from scorevision.utils.miner_registry import get_miners_from_registry
    from scorevision.validator.central.private_track.registry import get_registered_miners

    os.environ.setdefault("SCOREVISION_MECHID", "1")
    top_uids = {int(row["uid"]) for row in top_rows if row.get("uid") is not None}
    top_hotkeys = {str(row["hotkey"]) for row in top_rows if row.get("hotkey")}

    st = await get_subtensor()
    meta = await st.metagraph(44, mechid=int(os.environ.get("SCOREVISION_MECHID", "1")))
    commits = await st.get_all_revealed_commitments(44)

    print("== VALIDATOR INDEX COMMITMENTS ==")
    for uid, hotkey in enumerate(meta.hotkeys):
        arr = commits.get(hotkey) or []
        for block, payload in arr[-12:]:
            obj = parse_commit_payload(payload)
            if not obj:
                continue
            role = obj.get("role")
            if role not in ("central_validator", "audit_validator"):
                continue
            interesting = {
                key: obj.get(key)
                for key in (
                    "role",
                    "index_url",
                    "hotkey",
                    "chute_name",
                    "version",
                )
                if key in obj
            }
            print(f"uid={uid:>3} hotkey={hotkey} block={block} {json.dumps(interesting, sort_keys=True)}")
    print()

    print("== TOP UID REVEALED COMMITMENTS ==")
    for uid in sorted(top_uids):
        hotkey = meta.hotkeys[uid] if uid < len(meta.hotkeys) else None
        print(f"\n-- uid={uid} hotkey={hotkey} --")
        arr = commits.get(hotkey) if hotkey else None
        if not arr:
            print("no revealed commitments")
            continue
        for block, payload in arr[-8:]:
            obj = parse_commit_payload(payload)
            if obj is None:
                print(f"block={block} unparsable_commit={payload!r}")
                continue
            interesting = {
                key: obj.get(key)
                for key in (
                    "role",
                    "track",
                    "element_id",
                    "model",
                    "revision",
                    "slug",
                    "chute_id",
                    "image_repo",
                    "image_tag",
                    "image_digest",
                )
                if key in obj
            }
            print(f"block={block} {json.dumps(interesting, sort_keys=True)}")

    print("\n== PRIVATE TRACK REGISTRATIONS ==")
    for element_id in PRIVATE_ELEMENTS:
        miners = await get_registered_miners(st, meta, set(), element_id)
        print(f"\n{element_id}: {len(miners)} registered")
        for miner in miners:
            marker = " TOP_UID" if miner.uid in top_uids or miner.hotkey in top_hotkeys else ""
            print(
                f"uid={miner.uid:>3}{marker} hotkey={miner.hotkey} "
                f"ip={miner.ip} port={miner.port} "
                f"image={miner.image_repo}:{miner.image_tag} block={miner.commit_block}"
            )

    if not include_public:
        print("\n(public Detect registry skipped; add --public to include it)")
        return

    print("\n== PUBLIC DETECT REGISTRATIONS ==")
    for element_id in PUBLIC_ELEMENTS:
        miners, skipped = await get_miners_from_registry(
            44,
            element_id=element_id,
            max_model_size_mb=30,
        )
        print(f"\n{element_id}: eligible={len(miners)} skipped={len(skipped)}")
        for uid, miner in sorted(miners.items()):
            marker = " TOP_UID" if uid in top_uids or miner.hotkey in top_hotkeys else ""
            print(
                f"uid={uid:>3}{marker} hotkey={miner.hotkey} "
                f"model={miner.model} revision={miner.revision} "
                f"slug={miner.slug} block={miner.block}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True, help="btcli metagraph JSON dump")
    parser.add_argument("--top", type=int, default=12, help="Top incentive rows to inspect")
    parser.add_argument("--online", action="store_true", help="Inspect on-chain commitments")
    parser.add_argument("--public", action="store_true", help="Also inspect public Detect registry")
    args = parser.parse_args()

    _data, rows = parse_local_metagraph(args.json, args.top)
    if args.online:
        asyncio.run(inspect_online(rows[: args.top], include_public=args.public))


if __name__ == "__main__":
    main()
