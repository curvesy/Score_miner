#!/usr/bin/env python3
"""Download Score manako challenge frames and render untrusted prediction overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from public_detect.manako import (
    MANAKO_INDEX_URL,
    download_manako_frames,
    fetch_manako_index,
    parse_manako_frames,
)


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
    parser.add_argument(
        "--element-filter",
        action="append",
        default=[],
        help="Only scan manako refs containing this element text. Repeat for multiple filters.",
    )
    parser.add_argument(
        "--max-refs",
        type=int,
        help="Maximum filtered refs to scan before stopping.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        help="Only follow response payloads from evaluations with at least this composite score.",
    )
    parser.add_argument("--debug-index", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.debug_index:
        data = fetch_manako_index(args.index_url)
        print(json.dumps(_summarize_index(data), indent=2, sort_keys=True))
        return
    manifest = download_manako_frames(
        output_dir=args.output_dir,
        index_url=args.index_url,
        limit=args.limit,
        element_filters=tuple(args.element_filter),
        max_refs=args.max_refs,
        min_score=args.min_score,
        verbose=not args.quiet,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "frames": manifest["frames"]}, indent=2))


def _summarize_index(data: object) -> dict[str, object]:
    if isinstance(data, dict):
        return {
            "type": "dict",
            "keys": list(data)[:40],
            "direct_frames": len(parse_manako_frames(data)),
            "sample": _safe_sample(data),
        }
    if isinstance(data, list):
        return {
            "type": "list",
            "length": len(data),
            "direct_frames": len(parse_manako_frames(data)),
            "element_counts": _element_counts(data),
            "sample": _safe_sample(data[:3]),
        }
    return {"type": type(data).__name__, "sample": str(data)[:1000]}


def _safe_sample(value: object) -> object:
    text = json.dumps(value, default=str)[:4000]
    return json.loads(text)


def _element_counts(items: list[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        if not isinstance(item, str):
            continue
        for part in item.split("/"):
            if part.startswith("manak0_Detect-") or part.startswith("manako_Detect-"):
                counts[part] = counts.get(part, 0) + 1
    return dict(sorted(counts.items(), key=lambda row: row[1], reverse=True)[:30])


if __name__ == "__main__":
    main()
