#!/usr/bin/env python3
"""Download Score manako challenge frames and render untrusted prediction overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from public_detect.manako import MANAKO_INDEX_URL, download_manako_frames


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-url", default=MANAKO_INDEX_URL)
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "data" / "proof_frames" / "manako_challenges",
        type=project_path,
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = download_manako_frames(
        output_dir=args.output_dir,
        index_url=args.index_url,
        limit=args.limit,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "frames": manifest["frames"]}, indent=2))


if __name__ == "__main__":
    main()
