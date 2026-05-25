"""Failure-review rendering for Phase 4 data planning."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from public_detect.score_eval import (
    Box,
    boxes_from_json,
    filter_predictions,
    load_yolo_dataset,
    load_yolo_ground_truth,
)


COLORS = {
    "ground_truth": "#00a676",
    "prediction": "#e84855",
    "miss": "#f2af29",
    "false_positive": "#8e44ad",
}


@dataclass(frozen=True)
class ReviewItem:
    kind: str
    image_id: str
    class_name: str
    cls: int
    xyxy: tuple[float, float, float, float]
    conf: float
    best_iou: float | None = None


def load_diagnostics(path: str | Path) -> list[ReviewItem]:
    data = json.loads(Path(path).read_text())
    items: list[ReviewItem] = []
    for kind, source_key in (("false_positive", "false_positives"), ("miss", "misses")):
        for row in data.get(source_key, []):
            items.append(
                ReviewItem(
                    kind=kind,
                    image_id=str(row["image_id"]),
                    class_name=str(row["class_name"]),
                    cls=int(row["cls"]),
                    xyxy=tuple(float(value) for value in row["xyxy"]),
                    conf=float(row.get("conf", 1.0)),
                    best_iou=(
                        float(row["best_iou"])
                        if row.get("best_iou") is not None
                        else None
                    ),
                )
            )
    return items


def export_failure_review(
    *,
    data_yaml: str | Path,
    diagnostics_json: str | Path,
    predictions_json: str | Path,
    summary_json: str | Path | None = None,
    output_dir: str | Path,
    crop_margin: float = 1.5,
    display_confidence: float = 0.25,
) -> dict[str, Any]:
    image_paths, class_names = load_yolo_dataset(data_yaml)
    gt_boxes, _ = load_yolo_ground_truth(data_yaml)
    predictions = boxes_from_json(predictions_json)
    predictions = _filter_display_predictions(
        predictions,
        class_names=class_names,
        summary_json=summary_json,
        fallback_confidence=display_confidence,
    )
    review_items = load_diagnostics(diagnostics_json)

    out = Path(output_dir)
    full_dir = out / "full"
    crop_dir = out / "crops"
    full_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    images_by_id = {path.stem: path for path in image_paths}
    gt_by_image = _group_boxes(gt_boxes)
    pred_by_image = _group_boxes(predictions)
    review_by_image = _group_review_items(review_items)

    summary_rows = []
    for image_id in sorted(images_by_id):
        image_path = images_by_id[image_id]
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for box in gt_by_image.get(image_id, []):
            _draw_box(
                draw,
                box.xyxy,
                COLORS["ground_truth"],
                f"GT {class_names.get(box.cls, box.cls)}",
                font,
                width=2,
            )
        for box in pred_by_image.get(image_id, []):
            _draw_box(
                draw,
                box.xyxy,
                COLORS["prediction"],
                f"P {class_names.get(box.cls, box.cls)} {box.conf:.2f}",
                font,
                width=2,
            )
        for item in review_by_image.get(image_id, []):
            label = _review_label(item)
            _draw_box(
                draw,
                item.xyxy,
                COLORS[item.kind],
                label,
                font,
                width=5,
            )
            crop_path = _save_crop(
                image_path=image_path,
                item=item,
                output_dir=crop_dir,
                margin=crop_margin,
            )
            summary_rows.append(
                {
                    "kind": item.kind,
                    "image_id": image_id,
                    "class_name": item.class_name,
                    "cls": item.cls,
                    "conf": item.conf,
                    "best_iou": "" if item.best_iou is None else item.best_iou,
                    "xyxy": json.dumps(list(item.xyxy)),
                    "crop": str(crop_path),
                    "full": str(full_dir / f"{image_id}.jpg"),
                }
            )

        image.save(full_dir / f"{image_id}.jpg", quality=95)

    _write_csv(out / "review_items.csv", summary_rows)
    summary = {
        "images": len(image_paths),
        "review_items": len(review_items),
        "display_predictions": len(predictions),
        "misses": sum(1 for item in review_items if item.kind == "miss"),
        "false_positives": sum(1 for item in review_items if item.kind == "false_positive"),
        "by_class": _by_class_summary(review_items),
        "output_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _filter_display_predictions(
    predictions: list[Box],
    *,
    class_names: dict[int, str],
    summary_json: str | Path | None,
    fallback_confidence: float,
) -> list[Box]:
    if summary_json is None:
        return filter_predictions(predictions, confidence=fallback_confidence)
    summary_path = Path(summary_json)
    if not summary_path.exists():
        return filter_predictions(predictions, confidence=fallback_confidence)
    data = json.loads(summary_path.read_text())
    best = data.get("best") or {}
    confidence = float(best.get("confidence", fallback_confidence))
    max_det = int(best["max_det"]) if best.get("max_det") not in (None, "") else None
    per_class = _per_class_thresholds(best.get("per_class_confidence"), class_names)
    return filter_predictions(
        predictions,
        confidence=confidence,
        per_class_confidence=per_class,
        max_det=max_det,
    )


def _per_class_thresholds(value: object, class_names: dict[int, str]) -> dict[int, float] | None:
    if not value:
        return None
    by_name = json.loads(str(value)) if isinstance(value, str) else value
    if not isinstance(by_name, dict):
        return None
    name_to_id = {name: idx for idx, name in class_names.items()}
    return {
        name_to_id[name]: float(threshold)
        for name, threshold in by_name.items()
        if name in name_to_id
    }


def crop_bounds(
    xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    margin: float = 1.5,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * margin
    pad_y = height * margin
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(image_width, int(x2 + pad_x)),
        min(image_height, int(y2 + pad_y)),
    )


def _draw_box(
    draw: ImageDraw.ImageDraw,
    xyxy: tuple[float, float, float, float],
    color: str,
    label: str,
    font: ImageFont.ImageFont,
    width: int,
) -> None:
    x1, y1, x2, y2 = xyxy
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    text_box = draw.textbbox((x1, max(0, y1 - 12)), label, font=font)
    draw.rectangle(text_box, fill=color)
    draw.text((text_box[0], text_box[1]), label, fill="white", font=font)


def _review_label(item: ReviewItem) -> str:
    if item.kind == "false_positive":
        iou = "" if item.best_iou is None else f" iou={item.best_iou:.2f}"
        return f"FP {item.class_name} {item.conf:.2f}{iou}"
    return f"MISS {item.class_name}"


def _save_crop(
    *,
    image_path: Path,
    item: ReviewItem,
    output_dir: Path,
    margin: float,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    bounds = crop_bounds(item.xyxy, image.width, image.height, margin=margin)
    crop = image.crop(bounds)
    target = output_dir / item.kind / item.class_name.replace(" ", "_")
    target.mkdir(parents=True, exist_ok=True)
    output = target / f"{item.image_id}_{len(list(target.glob(item.image_id + '_*.jpg'))):03d}.jpg"
    draw = ImageDraw.Draw(crop)
    font = ImageFont.load_default()
    shifted = (
        item.xyxy[0] - bounds[0],
        item.xyxy[1] - bounds[1],
        item.xyxy[2] - bounds[0],
        item.xyxy[3] - bounds[1],
    )
    _draw_box(draw, shifted, COLORS[item.kind], _review_label(item), font, width=4)
    crop.save(output, quality=95)
    return output


def _group_boxes(boxes: list[Box]) -> dict[str, list[Box]]:
    grouped: dict[str, list[Box]] = {}
    for box in boxes:
        grouped.setdefault(box.image_id, []).append(box)
    return grouped


def _group_review_items(items: list[ReviewItem]) -> dict[str, list[ReviewItem]]:
    grouped: dict[str, list[ReviewItem]] = {}
    for item in items:
        grouped.setdefault(item.image_id, []).append(item)
    return grouped


def _by_class_summary(items: list[ReviewItem]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for item in items:
        row = summary.setdefault(item.class_name, {"miss": 0, "false_positive": 0})
        row[item.kind] += 1
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["kind", "image_id", "class_name", "cls", "conf", "best_iou", "xyxy", "crop", "full"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
