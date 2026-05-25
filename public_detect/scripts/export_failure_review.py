#!/usr/bin/env python3
"""Export visual failure review artifacts from Phase 3 diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from public_detect.review import export_failure_review


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=project_path)
    parser.add_argument("--diagnostics", required=True, type=project_path)
    parser.add_argument("--predictions", required=True, type=project_path)
    parser.add_argument("--summary", type=project_path)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--output-root",
        default=PROJECT_ROOT / "reports" / "failure_reviews",
        type=project_path,
    )
    parser.add_argument("--crop-margin", default=1.5, type=float)
    parser.add_argument("--display-confidence", default=0.25, type=float)
    args = parser.parse_args()

    summary = export_failure_review(
        data_yaml=args.data,
        diagnostics_json=args.diagnostics,
        predictions_json=args.predictions,
        summary_json=args.summary,
        output_dir=args.output_root / args.name,
        crop_margin=args.crop_margin,
        display_confidence=args.display_confidence,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
