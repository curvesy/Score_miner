#!/usr/bin/env python3
"""Build a minimal public Detect deploy repo containing miner.py and weights.onnx."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from public_detect.deploy_repo import build_deploy_repo
from public_detect.elements import load_element_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=project_path)
    parser.add_argument("--element-config", required=True, type=project_path)
    parser.add_argument("--output-dir", required=True, type=project_path)
    parser.add_argument("--input-size", type=int, default=768)
    parser.add_argument("--conf", default="0.25")
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--min-side", type=float, default=4.0)
    parser.add_argument("--min-box-area", type=float, default=16.0)
    parser.add_argument("--max-aspect-ratio", type=float, default=12.0)
    parser.add_argument("--max-mb", type=float, default=30.0)
    args = parser.parse_args()

    spec = load_element_spec(args.element_config)
    confs = parse_conf(args.conf, class_count=len(spec.objects))
    report = build_deploy_repo(
        weights=args.weights,
        output_dir=args.output_dir,
        class_names=spec.objects,
        input_size=args.input_size,
        conf_thresholds=confs,
        max_det=args.max_det,
        iou_thres=args.iou,
        min_side=args.min_side,
        min_box_area=args.min_box_area,
        max_aspect_ratio=args.max_aspect_ratio,
        max_mb=args.max_mb,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passes"]:
        raise SystemExit(2)


def parse_conf(value: str, class_count: int) -> list[float]:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) == 1:
        return parts * class_count
    if len(parts) != class_count:
        raise ValueError(f"expected 1 or {class_count} confidence values, got {len(parts)}")
    return parts


if __name__ == "__main__":
    main()
