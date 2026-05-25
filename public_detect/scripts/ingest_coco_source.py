#!/usr/bin/env python3
"""Ingest a COCO-format outside source into a review-gated YOLO candidate set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from public_detect.ingest import ingest_coco_detection, load_coco_ingest_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=project_path)
    parser.add_argument("--coco-json", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=project_path)
    args = parser.parse_args()

    config = load_coco_ingest_config(args.config, PROJECT_ROOT)
    manifest = ingest_coco_detection(
        coco_json=args.coco_json,
        image_root=args.image_root,
        output_dir=args.output_dir,
        config=config,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
