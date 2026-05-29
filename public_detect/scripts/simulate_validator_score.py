#!/usr/bin/env python3
"""Estimate TurboVision object-detection pillars on a YOLO val set.

Lifts the pillar metric algorithms from
turbovision/scorevision/vlm_pipeline/non_vlm_scoring/objects.py and applies them
to a YOLO val set + your trained model.

Pillars:
  map50          - per-class AP at IoU 0.5 (strict)
  precision      - global TP / (TP+FP) at IoU 0.5
  recall         - global TP / (TP+FN) at IoU 0.5
  false_positive - max(0, 1 - ffpi/10), ffpi = global_fp / num_frames
  count          - Hungarian F1 at IoU 0.3 (lenient, label-agnostic)
  iou            - AUC-F1 across IoU {0.3, 0.5} (lenient, label-agnostic)

For Detect-beverage-detect, the public manifest uses:
  score = 0.6 * map50 + 0.4 * false_positive

The count/iou outputs below are diagnostics only for Beverage. They are useful
for understanding placement/count behavior, but they are not part of the current
Beverage manifest score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from public_detect.score_eval import (
    Box,
    evaluate_score,
    load_yolo_dataset,
    load_yolo_ground_truth,
)


def iou_box(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def hungarian_f1(p_boxes, h_boxes, iou_thresh):
    """Label-agnostic Hungarian F1 at one IoU threshold."""
    if not p_boxes and not h_boxes:
        return 1.0
    if not p_boxes or not h_boxes:
        return 0.0
    N, M = len(p_boxes), len(h_boxes)
    cost = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            cost[i, j] = -iou_box(p_boxes[i], h_boxes[j])
    rows, cols = linear_sum_assignment(cost)
    matched_h, matched_g = set(), set()
    tp = 0
    for r, c in zip(rows, cols):
        if -cost[r, c] >= iou_thresh:
            tp += 1
            matched_h.add(c)
            matched_g.add(r)
    fp = M - len(matched_h)
    fn = N - len(matched_g)
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom > 0 else 1.0


def auc_f1(p_boxes, h_boxes, thresholds):
    vals = [hungarian_f1(p_boxes, h_boxes, t) for t in thresholds]
    return float(sum(vals) / len(vals)) if vals else 0.0


def parse_weights(value: str) -> dict[str, float]:
    weights = json.loads(value)
    if not isinstance(weights, dict):
        raise argparse.ArgumentTypeError("--weights must be a JSON object")
    parsed = {str(key): float(score) for key, score in weights.items()}
    total = sum(parsed.values())
    if abs(total - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(f"--weights must sum to 1.0, got {total}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou-nms", type=float, default=0.5)
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default='{"map50": 0.6, "false_positive": 0.4}',
        help=(
            "JSON pillar weights. Default matches Detect-beverage-detect public manifest: "
            "'{\"map50\": 0.6, \"false_positive\": 0.4}'."
        ),
    )
    args = parser.parse_args()

    image_paths, class_names = load_yolo_dataset(args.data)
    gt_boxes, _ = load_yolo_ground_truth(args.data)

    yolo = YOLO(args.model)
    results = yolo.predict(
        source=[str(p) for p in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou_nms,
        max_det=args.max_det,
        verbose=False,
    )
    pred_boxes: list[Box] = []
    for img_path, r in zip(image_paths, results, strict=True):
        if r.boxes is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        for coords, score, c in zip(xyxy, conf, cls, strict=True):
            pred_boxes.append(Box(
                image_id=img_path.stem,
                cls=int(c),
                xyxy=tuple(float(v) for v in coords),
                conf=float(score),
            ))

    metrics = evaluate_score(gt_boxes, pred_boxes, class_names)

    print("=" * 60)
    print(f"images: {len(image_paths)}  gt_boxes: {metrics.gt}  pred_boxes: {metrics.predictions}")
    print("=" * 60)
    print("\n[strict pillars] (IoU 0.5, class-strict)")
    print(f"  map50          : {metrics.map50:.4f}")
    print(f"  precision      : {metrics.precision:.4f}")
    print(f"  recall         : {metrics.recall:.4f}")
    print(f"  false_positive : {metrics.fp_score:.4f}")
    print("  per-class:")
    for c in metrics.classes:
        print(f"    {c.name:8s}: ap50={c.ap50:.3f} prec={c.precision:.3f} "
              f"rec={c.recall:.3f}  tp={c.tp} fp={c.fp} fn={c.fn}")

    gt_by_img: dict[str, list[Box]] = {}
    pred_by_img: dict[str, list[Box]] = {}
    for b in gt_boxes:
        gt_by_img.setdefault(b.image_id, []).append(b)
    for b in pred_boxes:
        pred_by_img.setdefault(b.image_id, []).append(b)

    count_scores = []
    iou_scores = []
    for img in image_paths:
        gts = gt_by_img.get(img.stem, [])
        preds = pred_by_img.get(img.stem, [])
        g_xyxy = [b.xyxy for b in gts]
        p_xyxy = [b.xyxy for b in preds]
        count_scores.append(hungarian_f1(g_xyxy, p_xyxy, 0.3))
        iou_scores.append(auc_f1(g_xyxy, p_xyxy, [0.3, 0.5]))

    count_pillar = sum(count_scores) / len(count_scores) if count_scores else 0.0
    iou_pillar = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    print("\n[lenient pillars] (Hungarian, label-agnostic)")
    print(f"  count (F1@IoU 0.3)         : {count_pillar:.4f}")
    print(f"  iou   (AUC-F1@IoU 0.3,0.5) : {iou_pillar:.4f}")

    pillars = {
        "map50": metrics.map50,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "false_positive": metrics.fp_score,
        "count": count_pillar,
        "iou": iou_pillar,
    }
    unknown = sorted(set(args.weights) - set(pillars))
    if unknown:
        raise SystemExit(f"unknown pillar(s) in --weights: {unknown}")
    manifest_score = sum(pillars[key] * weight for key, weight in args.weights.items())
    print("\n[manifest score estimate]")
    print(f"  weights        : {json.dumps(args.weights, sort_keys=True)}")
    print(f"  score          : {manifest_score:.4f}")
    print("  note           : default weights match current Detect-beverage-detect manifest")

    print("\n[reference]")
    print("  live top-miner score : 0.7438  (console snapshot; check current console)")
    print("  live baseline_theta  : 0.1955")
    print("  target_score         : 0.9400")


if __name__ == "__main__":
    main()
