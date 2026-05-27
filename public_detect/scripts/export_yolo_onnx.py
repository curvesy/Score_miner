#!/usr/bin/env python3
"""Export a YOLO checkpoint to a minimal ONNX repo and run the 30MB size gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from public_detect.export_utils import check_size_gate, copy_export_to_repo, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=project_path)
    parser.add_argument("--output-dir", required=True, type=project_path)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--max-mb", type=float, default=30.0)
    args = parser.parse_args()

    model = YOLO(str(args.model))
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            half=args.half,
            dynamic=args.dynamic,
            simplify=True,
        )
    )
    target = copy_export_to_repo(exported_path, args.output_dir)
    report = check_size_gate(args.output_dir, max_mb=args.max_mb)
    report.update(
        {
            "source_model": str(args.model),
            "exported_model": str(exported_path),
            "repo_model": str(target),
            "imgsz": args.imgsz,
            "opset": args.opset,
            "half": args.half,
            "dynamic": args.dynamic,
        }
    )
    write_json(Path(args.output_dir) / "size_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
