#!/usr/bin/env python3
"""Check a deploy repo/folder against the public Detect size cap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from public_detect.export_utils import check_size_gate, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=project_path)
    parser.add_argument("--max-mb", type=float, default=30.0)
    parser.add_argument("--report")
    args = parser.parse_args()

    report = check_size_gate(args.path, max_mb=args.max_mb)
    if args.report:
        write_json(project_path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
