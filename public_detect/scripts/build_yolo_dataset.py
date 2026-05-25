#!/usr/bin/env python3
"""Convert a downloaded starter pack to Ultralytics YOLO format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from public_detect.elements import load_element_spec
from public_detect.score_api import parse_starter_assets
from public_detect.yolo_dataset import write_yolo_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--element-config", required=True, type=Path)
    parser.add_argument("--starter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    spec = load_element_spec(args.element_config)
    detail = json.loads((args.starter_dir / "raw" / "element_detail.json").read_text())
    assets = parse_starter_assets(detail)
    summary = write_yolo_dataset(
        assets=assets,
        spec=spec,
        image_dir=args.starter_dir / "images",
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2)[:4000])


if __name__ == "__main__":
    main()

