#!/usr/bin/env python3
"""Render starter-pack labels for visual sanity checking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from public_detect.elements import load_element_spec
from public_detect.render import render_asset
from public_detect.score_api import parse_starter_assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--element-config", required=True, type=Path)
    parser.add_argument("--starter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    spec = load_element_spec(args.element_config)
    detail = json.loads((args.starter_dir / "raw" / "element_detail.json").read_text())
    assets = parse_starter_assets(detail)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for asset in assets:
        image_path = next((args.starter_dir / "images").glob(f"{asset.asset_id}.*"))
        render_asset(
            asset=asset,
            spec=spec,
            image_path=image_path,
            output_path=args.output_dir / f"{asset.asset_id}.jpg",
        )
        count += 1
    print(f"rendered {count} label previews to {args.output_dir}")


if __name__ == "__main__":
    main()

