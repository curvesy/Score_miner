#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from score_miner_core.public_detect.score_api import (
    element_objects,
    fetch_element_detail,
    starter_assets,
)
from score_miner_core.public_detect.yolo_dataset import convert_starter_assets_to_yolo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--element-id", required=True)
    parser.add_argument("--out", type=Path, default=Path("score_miner_project/data/public_detect"))
    parser.add_argument("--lookback-days", type=int, default=4)
    args = parser.parse_args()

    detail = fetch_element_detail(args.element_id, lookback_days=args.lookback_days)
    objects = element_objects(detail)
    assets = starter_assets(detail)
    if not objects:
        raise SystemExit(f"No objects found for {args.element_id}")
    if not assets:
        raise SystemExit(f"No starter assets found for {args.element_id}")

    summary = convert_starter_assets_to_yolo(
        assets=assets,
        objects=objects,
        output_dir=args.out,
        element_id=args.element_id,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

