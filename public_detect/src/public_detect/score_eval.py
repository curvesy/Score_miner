"""Score-style evaluation helpers for SN44 public Detect.

This is not the private validator. It is a local instrument for comparing our
own checkpoints and thresholds against the public starter/proof data we have.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Box:
    image_id: str
    cls: int
    xyxy: tuple[float, float, float, float]
    conf: float = 1.0


@dataclass(frozen=True)
class ClassMetrics:
    cls: int
    name: str
    gt: int
    predictions: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    ap50: float
    fp_score: float
    score: float


@dataclass(frozen=True)
class ScoreMetrics:
    score: float
    map50: float
    fp_score: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    gt: int
    predictions: int
    classes: tuple[ClassMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["classes"] = [asdict(item) for item in self.classes]
        return data


def load_yolo_dataset(data_yaml: str | Path) -> tuple[list[Path], dict[int, str]]:
    """Return validation image paths and class names from a YOLO data.yaml."""
    path = Path(data_yaml)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"invalid YOLO data yaml: {path}")

    root = _resolve_dataset_root(path, data.get("path", path.parent))

    val = data.get("val") or data.get("train")
    image_paths = _resolve_image_paths(root, val)

    names = data.get("names")
    if isinstance(names, dict):
        class_names = {int(key): str(value) for key, value in names.items()}
    elif isinstance(names, list):
        class_names = {idx: str(value) for idx, value in enumerate(names)}
    else:
        raise ValueError(f"YOLO names must be list or dict in {path}")
    return image_paths, class_names


def load_yolo_ground_truth(data_yaml: str | Path) -> tuple[list[Box], dict[int, str]]:
    image_paths, class_names = load_yolo_dataset(data_yaml)
    boxes: list[Box] = []
    for image_path in image_paths:
        label_path = _label_path_for_image(image_path)
        if not label_path.exists():
            continue
        width, height = _image_size(image_path)
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, bw, bh = (float(value) for value in parts[1:5])
            boxes.append(
                Box(
                    image_id=image_path.stem,
                    cls=cls,
                    xyxy=_yolo_to_xyxy(xc, yc, bw, bh, width=width, height=height),
                )
            )
    return boxes, class_names


def filter_predictions(
    predictions: list[Box],
    confidence: float,
    per_class_confidence: dict[int, float] | None = None,
    max_det: int | None = None,
) -> list[Box]:
    grouped: dict[str, list[Box]] = {}
    for pred in predictions:
        threshold = confidence
        if per_class_confidence and pred.cls in per_class_confidence:
            threshold = per_class_confidence[pred.cls]
        if pred.conf >= threshold:
            grouped.setdefault(pred.image_id, []).append(pred)

    filtered: list[Box] = []
    for image_id in sorted(grouped):
        items = sorted(grouped[image_id], key=lambda item: item.conf, reverse=True)
        if max_det is not None:
            items = items[:max_det]
        filtered.extend(items)
    return filtered


def evaluate_score(
    ground_truth: list[Box],
    predictions: list[Box],
    class_names: dict[int, str],
    iou_threshold: float = 0.5,
) -> ScoreMetrics:
    class_metrics = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "predictions": 0}
    image_count = len({box.image_id for box in ground_truth})
    for cls in sorted(class_names):
        gt_cls = [box for box in ground_truth if box.cls == cls]
        pred_cls = [box for box in predictions if box.cls == cls]
        ap50 = average_precision_50(gt_cls, pred_cls, iou_threshold=iou_threshold)
        tp, fp, fn = match_counts(gt_cls, pred_cls, iou_threshold=iou_threshold)
        gt_count = len(gt_cls)
        pred_count = len(pred_cls)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        fp_score = false_positive_score(fp=fp, image_count=image_count)
        score = 0.6 * ap50 + 0.4 * fp_score
        class_metrics.append(
            ClassMetrics(
                cls=cls,
                name=class_names[cls],
                gt=gt_count,
                predictions=pred_count,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                ap50=ap50,
                fp_score=fp_score,
                score=score,
            )
        )
        for key, value in (
            ("tp", tp),
            ("fp", fp),
            ("fn", fn),
            ("gt", gt_count),
            ("predictions", pred_count),
        ):
            totals[key] += value

    map50 = _mean([item.ap50 for item in class_metrics])
    fp_score = false_positive_score(fp=totals["fp"], image_count=image_count)
    score = 0.6 * map50 + 0.4 * fp_score
    return ScoreMetrics(
        score=score,
        map50=map50,
        fp_score=fp_score,
        precision=_safe_div(totals["tp"], totals["tp"] + totals["fp"]),
        recall=_safe_div(totals["tp"], totals["tp"] + totals["fn"]),
        tp=totals["tp"],
        fp=totals["fp"],
        fn=totals["fn"],
        gt=totals["gt"],
        predictions=totals["predictions"],
        classes=tuple(class_metrics),
    )


def false_positive_score(fp: int, image_count: int) -> float:
    """TurboVision public Detect false-positive pillar.

    Mirrors scorevision/vlm_pipeline/non_vlm_scoring/objects.py:
    ffpi = global_fp / len(per_image)
    false_positive = max(0, 1 - ffpi / 10)
    """
    ffpi = _safe_div(float(fp), float(image_count))
    return max(0.0, 1.0 - (ffpi / 10.0))


def average_precision_50(
    ground_truth: list[Box],
    predictions: list[Box],
    iou_threshold: float = 0.5,
) -> float:
    if not ground_truth:
        return 0.0
    predictions = sorted(predictions, key=lambda item: item.conf, reverse=True)
    matched: set[int] = set()
    tp_curve = []
    fp_curve = []
    for pred in predictions:
        gt_index, best_iou = _best_unmatched_gt(pred, ground_truth, matched)
        if gt_index is not None and best_iou >= iou_threshold:
            matched.add(gt_index)
            tp_curve.append(1.0)
            fp_curve.append(0.0)
        else:
            tp_curve.append(0.0)
            fp_curve.append(1.0)
    if not tp_curve:
        return 0.0

    cum_tp = []
    cum_fp = []
    tp_total = 0.0
    fp_total = 0.0
    for tp, fp in zip(tp_curve, fp_curve, strict=True):
        tp_total += tp
        fp_total += fp
        cum_tp.append(tp_total)
        cum_fp.append(fp_total)

    recalls = [value / len(ground_truth) for value in cum_tp]
    precisions = [
        _safe_div(tp, tp + fp)
        for tp, fp in zip(cum_tp, cum_fp, strict=True)
    ]
    return interpolated_ap(recalls, precisions)


def interpolated_ap(recalls: list[float], precisions: list[float]) -> float:
    """COCO/VOC-style all-point interpolated AP for one IoU threshold."""
    if not recalls:
        return 0.0
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    ap = 0.0
    for idx in range(1, len(mrec)):
        if mrec[idx] != mrec[idx - 1]:
            ap += (mrec[idx] - mrec[idx - 1]) * mpre[idx]
    return ap


def match_counts(
    ground_truth: list[Box],
    predictions: list[Box],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    predictions = sorted(predictions, key=lambda item: item.conf, reverse=True)
    matched: set[int] = set()
    tp = 0
    fp = 0
    for pred in predictions:
        gt_index, best_iou = _best_unmatched_gt(pred, ground_truth, matched)
        if gt_index is not None and best_iou >= iou_threshold:
            matched.add(gt_index)
            tp += 1
        else:
            fp += 1
    fn = len(ground_truth) - tp
    return tp, fp, fn


def match_diagnostics(
    ground_truth: list[Box],
    predictions: list[Box],
    class_names: dict[int, str],
    iou_threshold: float = 0.5,
) -> dict[str, list[dict[str, Any]]]:
    """Return false positives and missed ground-truth boxes after greedy matching."""
    diagnostics = {"false_positives": [], "misses": []}
    for cls in sorted(class_names):
        gt_cls = [box for box in ground_truth if box.cls == cls]
        pred_cls = sorted(
            [box for box in predictions if box.cls == cls],
            key=lambda item: item.conf,
            reverse=True,
        )
        matched: set[int] = set()
        for pred in pred_cls:
            gt_index, best_iou = _best_unmatched_gt(pred, gt_cls, matched)
            if gt_index is not None and best_iou >= iou_threshold:
                matched.add(gt_index)
            else:
                row = asdict(pred)
                row["class_name"] = class_names[cls]
                row["best_iou"] = best_iou
                diagnostics["false_positives"].append(row)
        for idx, gt in enumerate(gt_cls):
            if idx not in matched:
                row = asdict(gt)
                row["class_name"] = class_names[cls]
                diagnostics["misses"].append(row)
    return diagnostics


def save_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def boxes_to_json(boxes: list[Box]) -> list[dict[str, Any]]:
    return [asdict(box) for box in boxes]


def boxes_from_json(path: str | Path) -> list[Box]:
    raw = json.loads(Path(path).read_text())
    return [
        Box(
            image_id=str(item["image_id"]),
            cls=int(item["cls"]),
            xyxy=tuple(float(value) for value in item["xyxy"]),
            conf=float(item.get("conf", 1.0)),
        )
        for item in raw
    ]


def _resolve_image_paths(root: Path, value: Any) -> list[Path]:
    if isinstance(value, list):
        paths = []
        for item in value:
            paths.extend(_resolve_image_paths(root, item))
        return sorted(paths)
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    if path.is_dir():
        return sorted(
            item for item in path.iterdir()
            if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
    if path.is_file() and path.suffix.lower() == ".txt":
        return [
            Path(line.strip())
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"could not resolve YOLO image path: {path}")


def _resolve_dataset_root(data_yaml: Path, value: Any) -> Path:
    root = Path(str(value)).expanduser()
    candidates = []
    if root.is_absolute():
        candidates.append(root)
    else:
        candidates.append((data_yaml.parent / root).resolve())
        candidates.append((Path.cwd() / root).resolve())
        # Older generated data.yaml files were written with the monorepo-relative
        # public_detect prefix. If the file has been copied into public_detect,
        # the closest correct root is usually the data.yaml parent.
        if root.parts[-2:] == data_yaml.parent.parts[-2:]:
            candidates.append(data_yaml.parent.resolve())
        if str(root).startswith("score_miner_project/public_detect/"):
            stripped = Path(str(root).removeprefix("score_miner_project/public_detect/"))
            candidates.append((Path.cwd() / stripped).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def _yolo_to_xyxy(
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    box_w = bw * width
    box_h = bh * height
    center_x = xc * width
    center_y = yc * height
    return (
        center_x - box_w / 2.0,
        center_y - box_h / 2.0,
        center_x + box_w / 2.0,
        center_y + box_h / 2.0,
    )


def _best_unmatched_gt(
    pred: Box,
    ground_truth: list[Box],
    matched: set[int],
) -> tuple[int | None, float]:
    best_index = None
    best_iou = 0.0
    for idx, gt in enumerate(ground_truth):
        if idx in matched or gt.image_id != pred.image_id:
            continue
        score = iou_xyxy(pred.xyxy, gt.xyxy)
        if score > best_iou:
            best_index = idx
            best_iou = score
    return best_index, best_iou


def iou_xyxy(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    inter_x1 = max(lx1, rx1)
    inter_y1 = max(ly1, ry1)
    inter_x2 = min(lx2, rx2)
    inter_y2 = min(ly2, ry2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - inter_area
    return _safe_div(inter_area, union)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
