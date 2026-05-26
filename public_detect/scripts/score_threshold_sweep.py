#!/usr/bin/env python3
"""Run a local Score-style threshold sweep for a trained YOLO checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from public_detect.score_eval import (
    Box,
    boxes_from_json,
    boxes_to_json,
    evaluate_score,
    filter_predictions,
    load_yolo_dataset,
    load_yolo_ground_truth,
    match_diagnostics,
    save_json,
)
from public_detect.sahi_inference import predict_sahi_ultralytics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_float_grid(value: str) -> list[float]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("threshold grid cannot be empty")
    return values


def run_predictions(
    model_path: Path,
    data_yaml: Path,
    imgsz: int,
    iou: float,
    max_det: int,
    base_conf: float,
) -> list[Box]:
    image_paths, _ = load_yolo_dataset(data_yaml)
    model = YOLO(str(model_path))
    boxes: list[Box] = []
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        conf=base_conf,
        iou=iou,
        max_det=max_det,
        stream=False,
        verbose=False,
    )
    for image_path, result in zip(image_paths, results, strict=True):
        if result.boxes is None:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy()
        for coords, score, cls_id in zip(xyxy, conf, cls, strict=True):
            boxes.append(
                Box(
                    image_id=image_path.stem,
                    cls=int(cls_id),
                    xyxy=tuple(float(value) for value in coords),
                    conf=float(score),
                )
            )
    return boxes


def run_sahi_predictions(
    model_path: Path,
    data_yaml: Path,
    base_conf: float,
    device: str | None,
    slice_height: int,
    slice_width: int,
    overlap: float,
    postprocess_iou: float,
) -> list[Box]:
    image_paths, _ = load_yolo_dataset(data_yaml)
    return predict_sahi_ultralytics(
        model_path=model_path,
        image_paths=image_paths,
        confidence=base_conf,
        device=device,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap=overlap,
        postprocess_iou=postprocess_iou,
    )


def threshold_rows(
    ground_truth: list[Box],
    predictions: list[Box],
    class_names: dict[int, str],
    thresholds: list[float],
    max_det_values: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for confidence, max_det in product(thresholds, max_det_values):
        filtered = filter_predictions(predictions, confidence=confidence, max_det=max_det)
        metrics = evaluate_score(ground_truth, filtered, class_names)
        row = {
            "mode": "global",
            "confidence": confidence,
            "max_det": max_det,
            "score": metrics.score,
            "map50": metrics.map50,
            "fp_score": metrics.fp_score,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "predictions": metrics.predictions,
            "gt": metrics.gt,
            "per_class_confidence": "",
        }
        rows.append(row)
    return rows


def per_class_rows(
    ground_truth: list[Box],
    predictions: list[Box],
    class_names: dict[int, str],
    base_confidence: float,
    class_thresholds: list[float],
    max_det: int,
    limit: int,
) -> list[dict[str, Any]]:
    classes = sorted(class_names)
    combos = product(class_thresholds, repeat=len(classes))
    rows = []
    for idx, combo in enumerate(combos):
        if idx >= limit:
            break
        per_class = dict(zip(classes, combo, strict=True))
        filtered = filter_predictions(
            predictions,
            confidence=base_confidence,
            per_class_confidence=per_class,
            max_det=max_det,
        )
        metrics = evaluate_score(ground_truth, filtered, class_names)
        rows.append(
            {
                "mode": "per_class",
                "confidence": base_confidence,
                "max_det": max_det,
                "score": metrics.score,
                "map50": metrics.map50,
                "fp_score": metrics.fp_score,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "tp": metrics.tp,
                "fp": metrics.fp,
                "fn": metrics.fn,
                "predictions": metrics.predictions,
                "gt": metrics.gt,
                "per_class_confidence": json.dumps(
                    {class_names[key]: value for key, value in per_class.items()},
                    sort_keys=True,
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=project_path)
    parser.add_argument("--data", required=True, type=project_path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "reports" / "score_sweeps", type=project_path)
    parser.add_argument("--imgsz", default=960, type=int)
    parser.add_argument("--iou", default=0.7, type=float)
    parser.add_argument("--base-conf", default=0.001, type=float)
    parser.add_argument("--max-det", default=300, type=int)
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70")
    parser.add_argument("--max-det-grid", default="20,50,100,300")
    parser.add_argument("--per-class", action="store_true")
    parser.add_argument("--per-class-thresholds", default="0.10,0.20,0.30,0.40,0.50")
    parser.add_argument("--per-class-limit", default=5000, type=int)
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--prediction-mode", choices=["single", "sahi"], default="single")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sahi-slice-height", default=640, type=int)
    parser.add_argument("--sahi-slice-width", default=640, type=int)
    parser.add_argument("--sahi-overlap", default=0.25, type=float)
    parser.add_argument("--sahi-postprocess-iou", default=0.5, type=float)
    args = parser.parse_args()

    output_dir = args.output_dir / args.name
    predictions_path = output_dir / "raw_predictions.json"
    ground_truth, class_names = load_yolo_ground_truth(args.data)

    if args.reuse_predictions and predictions_path.exists():
        predictions = boxes_from_json(predictions_path)
    elif args.prediction_mode == "sahi":
        predictions = run_sahi_predictions(
            model_path=args.model,
            data_yaml=args.data,
            base_conf=args.base_conf,
            device=args.device,
            slice_height=args.sahi_slice_height,
            slice_width=args.sahi_slice_width,
            overlap=args.sahi_overlap,
            postprocess_iou=args.sahi_postprocess_iou,
        )
        save_json(predictions_path, boxes_to_json(predictions))
    else:
        predictions = run_predictions(
            model_path=args.model,
            data_yaml=args.data,
            imgsz=args.imgsz,
            iou=args.iou,
            max_det=args.max_det,
            base_conf=args.base_conf,
        )
        save_json(predictions_path, boxes_to_json(predictions))

    thresholds = parse_float_grid(args.thresholds)
    max_det_values = [int(value) for value in parse_float_grid(args.max_det_grid)]
    rows = threshold_rows(
        ground_truth=ground_truth,
        predictions=predictions,
        class_names=class_names,
        thresholds=thresholds,
        max_det_values=max_det_values,
    )
    best_global = max(rows, key=lambda item: item["score"])

    all_rows = list(rows)
    if args.per_class:
        all_rows.extend(
            per_class_rows(
                ground_truth=ground_truth,
                predictions=predictions,
                class_names=class_names,
                base_confidence=float(best_global["confidence"]),
                class_thresholds=parse_float_grid(args.per_class_thresholds),
                max_det=int(best_global["max_det"]),
                limit=args.per_class_limit,
            )
        )

    best = max(all_rows, key=lambda item: item["score"])
    filtered = filter_predictions(
        predictions,
        confidence=float(best["confidence"]),
        per_class_confidence=_parse_per_class_confidence(best["per_class_confidence"], class_names),
        max_det=int(best["max_det"]),
    )
    metrics = evaluate_score(ground_truth, filtered, class_names)
    diagnostics = match_diagnostics(ground_truth, filtered, class_names)

    write_csv(output_dir / "sweep.csv", all_rows)
    save_json(output_dir / "diagnostics.json", diagnostics)
    save_json(
        output_dir / "summary.json",
        {
            "model": str(args.model),
            "data": str(args.data),
            "imgsz": args.imgsz,
            "iou": args.iou,
            "base_conf": args.base_conf,
            "prediction_mode": args.prediction_mode,
            "sahi": {
                "slice_height": args.sahi_slice_height,
                "slice_width": args.sahi_slice_width,
                "overlap": args.sahi_overlap,
                "postprocess_iou": args.sahi_postprocess_iou,
            } if args.prediction_mode == "sahi" else None,
            "raw_predictions": len(predictions),
            "best": best,
            "metrics": metrics.to_dict(),
            "diagnostics": {
                "false_positives": len(diagnostics["false_positives"]),
                "misses": len(diagnostics["misses"]),
            },
            "class_names": class_names,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "best": best, "metrics": metrics.to_dict()}, indent=2))


def _parse_per_class_confidence(value: str, class_names: dict[int, str]) -> dict[int, float] | None:
    if not value:
        return None
    by_name = json.loads(value)
    name_to_id = {name: idx for idx, name in class_names.items()}
    return {name_to_id[name]: float(threshold) for name, threshold in by_name.items()}


if __name__ == "__main__":
    main()
