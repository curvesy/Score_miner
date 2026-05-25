#!/usr/bin/env python3
"""Extract review/labeling frames from a local video file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from public_detect.ingest import extract_video_frames


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=project_path)
    parser.add_argument("--fps", default=0.5, type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--prefix")
    args = parser.parse_args()

    manifest = extract_video_frames(
        video=args.video,
        output_dir=args.output_dir,
        fps=args.fps,
        max_frames=args.max_frames,
        prefix=args.prefix,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
