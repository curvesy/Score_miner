#!/usr/bin/env python3
"""Download Score starterPack assets for a public Detect element."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from public_detect.elements import load_element_spec
from public_detect.score_api import (
    download_bytes,
    fetch_element_detail,
    fetch_json,
    parse_starter_assets,
    write_json,
)


def _suffix_from_url(url: str, default: str) -> str:
    suffix = Path(urlparse(url).path).suffix
    return suffix or default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--element-config", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("score_miner_project/public_detect/data/starter_packs"), type=Path)
    parser.add_argument("--lookback-days", default=4, type=int)
    args = parser.parse_args()

    spec = load_element_spec(args.element_config)
    detail = fetch_element_detail(spec.element_id, lookback_days=args.lookback_days)
    assets = parse_starter_assets(detail)

    out = args.output_root / spec.slug
    images = out / "images"
    annotations = out / "annotations"
    raw = out / "raw"
    for path in (images, annotations, raw):
        path.mkdir(parents=True, exist_ok=True)

    write_json(raw / "element_detail.json", detail)
    manifest_rows = []
    for asset in assets:
        if asset.objects:
            spec.assert_objects_match(asset.objects)

        image_suffix = _suffix_from_url(asset.image_url, ".png")
        image_path = images / f"{asset.asset_id}{image_suffix}"
        image_path.write_bytes(download_bytes(asset.image_url))

        embedded_path = annotations / f"{asset.asset_id}.embedded.json"
        write_json(embedded_path, [asdict(item) for item in asset.annotations])

        annotation_path = None
        if asset.annotation_url:
            annotation_path = annotations / f"{asset.asset_id}.source.json"
            try:
                write_json(annotation_path, fetch_json(asset.annotation_url))
            except Exception as exc:
                annotation_path.write_text(json.dumps({"error": str(exc), "url": asset.annotation_url}, indent=2) + "\n")

        manifest_rows.append(
            {
                "asset_id": asset.asset_id,
                "image": str(image_path),
                "embedded_annotation": str(embedded_path),
                "source_annotation": str(annotation_path) if annotation_path else None,
                "frame_index": asset.frame_index,
                "box_count": len(asset.annotations),
                "objects": list(asset.objects),
                "image_url": asset.image_url,
                "annotation_url": asset.annotation_url,
            }
        )

    write_json(
        out / "manifest.json",
        {
            "element_id": spec.element_id,
            "slug": spec.slug,
            "objects": list(spec.objects),
            "asset_count": len(manifest_rows),
            "assets": manifest_rows,
        },
    )
    print(f"downloaded {len(manifest_rows)} starter assets to {out}")


if __name__ == "__main__":
    main()

