#!/usr/bin/env python3
"""Score a model against a YOLO dataset using the WINNER'S inference recipe.

Replicates the post-processing pipeline used by navierstocks/drink (live 0.748)
and SuperBitDev/bev1, both confirmed to use the same recipe:

  - per-class confidence floor (cup, bottle, can)
  - per-class "rescue bonus": if a class has 0 boxes passing its threshold in
    a frame, admit its top-1 candidate when score >= (threshold - bonus)
  - flip TTA (horizontal flip; predictions fused via per-class NMS + cluster-max)
  - per-class hard NMS, then cross-class dedup ordered by (score - threshold)
  - sane-box filter (min side, min area, max aspect ratio, < 95% image)

Run example:
  PYTHONPATH=src uv run python scripts/score_winner_style.py \\
      --model runs/beverage/yolo11n_phase4_external_v1_local/weights/best.pt \\
      --data data/yolo_candidates/beverage_winner_proxy_v1/data.yaml \\
      --imgsz 1280
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from public_detect.score_eval import (
    Box,
    evaluate_score,
    load_yolo_dataset,
    load_yolo_ground_truth,
)


# Winner profile defaults (navierstocks/drink @ 0.748)
DEFAULT_CONF = (0.60, 0.45, 0.50)      # cup, bottle, can
DEFAULT_BONUS = (0.00, 0.00, 0.20)
DEFAULT_IOU_NMS = 0.40
DEFAULT_CROSS_IOU = 0.70
DEFAULT_MIN_SIDE = 8.0
DEFAULT_MIN_AREA = 100.0
DEFAULT_MAX_AR = 10.0
DEFAULT_MAX_DET = 300


def conf_filter(scores: np.ndarray, cls_ids: np.ndarray,
                conf_thr: np.ndarray, bonus: np.ndarray) -> np.ndarray:
    """Per-class threshold with top-1 rescue when a class has zero passers."""
    if len(scores) == 0:
        return np.zeros(0, dtype=bool)
    keep = scores >= conf_thr[cls_ids]
    for c in np.unique(cls_ids):
        b = float(bonus[c])
        if b <= 0.0:
            continue
        cm = cls_ids == c
        if keep[cm].any():
            continue
        idx = np.where(cm)[0]
        top = int(idx[int(np.argmax(scores[idx]))])
        if scores[top] >= conf_thr[c] - b:
            keep[top] = True
    return keep


def hard_nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    n = len(boxes)
    if n == 0:
        return np.array([], dtype=np.intp)
    order = np.argsort(-scores)
    keep: list[int] = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        a_i = max(0.0, (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]))
        a_r = (np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) *
               np.maximum(0.0, boxes[rest, 3] - boxes[rest, 1]))
        iou = inter / (a_i + a_r - inter + 1e-7)
        order = rest[iou <= iou_thresh]
    return np.array(keep, dtype=np.intp)


def per_class_hard_nms(boxes, scores, cls_ids, iou_thresh):
    if len(boxes) == 0:
        return np.array([], dtype=np.intp)
    all_keep: list[int] = []
    for c in np.unique(cls_ids):
        mask = cls_ids == c
        indices = np.where(mask)[0]
        keep = hard_nms(boxes[mask], scores[mask], iou_thresh)
        all_keep.extend(indices[keep].tolist())
    all_keep.sort()
    return np.array(all_keep, dtype=np.intp)


def cross_class_dedup(boxes, scores, cls_ids, iou_thresh, conf_thr):
    n = len(boxes)
    if n <= 1:
        return boxes, scores, cls_ids
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    cls_ids = np.asarray(cls_ids, dtype=np.int32)
    areas = (np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) *
             np.maximum(0.0, boxes[:, 3] - boxes[:, 1]))
    margins = scores - conf_thr[cls_ids]
    order = np.lexsort((-areas, -margins))
    suppressed = np.zeros(n, dtype=bool)
    keep: list[int] = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        bi = boxes[i]
        xx1 = np.maximum(bi[0], boxes[:, 0])
        yy1 = np.maximum(bi[1], boxes[:, 1])
        xx2 = np.minimum(bi[2], boxes[:, 2])
        yy2 = np.minimum(bi[3], boxes[:, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        a_i = max(1e-7, float((bi[2] - bi[0]) * (bi[3] - bi[1])))
        iou = inter / (a_i + areas - inter + 1e-7)
        dup = iou > iou_thresh
        dup[i] = False
        suppressed |= dup
    keep_idx = np.array(keep, dtype=np.intp)
    return boxes[keep_idx], scores[keep_idx], cls_ids[keep_idx]


def filter_sane_boxes(boxes, scores, cls_ids, orig_size,
                       min_side, min_area, max_ar):
    if len(boxes) == 0:
        return boxes, scores, cls_ids
    orig_w, orig_h = orig_size
    image_area = float(orig_w * orig_h)
    bw = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    bh = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    area = bw * bh
    ar = np.where(
        (bw > 0) & (bh > 0),
        np.maximum(bw / np.maximum(bh, 1e-6), bh / np.maximum(bw, 1e-6)),
        np.inf,
    )
    keep = (
        (bw >= min_side) & (bh >= min_side) &
        (area >= min_area) &
        (area <= 0.95 * image_area) &
        (ar <= max_ar)
    )
    return boxes[keep], scores[keep], cls_ids[keep]


def max_score_per_cluster(post_boxes, post_cls, full_boxes, full_scores,
                          full_cls, iou_thresh):
    n = len(post_boxes)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    full_areas = (np.maximum(0.0, full_boxes[:, 2] - full_boxes[:, 0]) *
                  np.maximum(0.0, full_boxes[:, 3] - full_boxes[:, 1]))
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        bi = post_boxes[i]
        xx1 = np.maximum(bi[0], full_boxes[:, 0])
        yy1 = np.maximum(bi[1], full_boxes[:, 1])
        xx2 = np.minimum(bi[2], full_boxes[:, 2])
        yy2 = np.minimum(bi[3], full_boxes[:, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        a_i = max(0.0, float((bi[2] - bi[0]) * (bi[3] - bi[1])))
        iou = inter / (a_i + full_areas - inter + 1e-7)
        cluster = (iou >= iou_thresh) & (full_cls == post_cls[i])
        out[i] = float(np.max(full_scores[cluster])) if np.any(cluster) else 0.0
    return out


def winner_predict_one(yolo, image_path: Path, imgsz: int, conf_thr, bonus,
                       iou_nms, cross_iou, min_side, min_area, max_ar,
                       max_det, use_tta: bool):
    """Run yolo at conf=0.001, then apply winner's full pipeline. Returns
    list of (xyxy, score, cls_id) tuples in original-image coordinates."""
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    orig_h, orig_w = img.shape[:2]

    def candidates(image):
        r = yolo.predict(source=image, imgsz=imgsz, conf=0.001,
                         iou=0.7, max_det=1000, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return (np.zeros((0, 4), np.float32),
                    np.zeros(0, np.float32), np.zeros(0, np.int32))
        return (r.boxes.xyxy.cpu().numpy().astype(np.float32),
                r.boxes.conf.cpu().numpy().astype(np.float32),
                r.boxes.cls.cpu().numpy().astype(np.int32))

    b_o, s_o, c_o = candidates(img)

    if use_tta:
        flipped = cv2.flip(img, 1)
        b_f, s_f, c_f = candidates(flipped)
        if len(b_f) > 0:
            # remap flipped x-coordinates to original frame
            x1 = orig_w - b_f[:, 2]
            x2 = orig_w - b_f[:, 0]
            b_f = np.stack([x1, b_f[:, 1], x2, b_f[:, 3]], axis=1).astype(np.float32)
        boxes = np.concatenate([b_o, b_f], axis=0) if len(b_f) else b_o
        scores = np.concatenate([s_o, s_f], axis=0) if len(s_f) else s_o
        cls_ids = np.concatenate([c_o, c_f], axis=0) if len(c_f) else c_o
    else:
        boxes, scores, cls_ids = b_o, s_o, c_o

    if len(boxes) == 0:
        return []

    keep = conf_filter(scores, cls_ids, conf_thr, bonus)
    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
    if len(boxes) == 0:
        return []

    boxes, scores, cls_ids = filter_sane_boxes(
        boxes, scores, cls_ids, (orig_w, orig_h), min_side, min_area, max_ar
    )
    if len(boxes) == 0:
        return []

    full_b, full_s, full_c = boxes.copy(), scores.copy(), cls_ids.copy()

    if len(boxes) > 1:
        keep = per_class_hard_nms(boxes, scores, cls_ids, iou_nms)
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]

    if len(boxes) > max_det:
        top = np.argsort(-scores)[:max_det]
        boxes, scores, cls_ids = boxes[top], scores[top], cls_ids[top]

    if use_tta and len(boxes) > 0:
        boosted = max_score_per_cluster(boxes, cls_ids, full_b, full_s, full_c, iou_nms)
        scores = boosted

    if len(boxes) > 1:
        boxes, scores, cls_ids = cross_class_dedup(
            boxes, scores, cls_ids, cross_iou, conf_thr
        )

    return [(tuple(b.tolist()), float(s), int(c))
            for b, s, c in zip(boxes, scores, cls_ids)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--profile", choices=["drink", "bev1"], default="drink",
                   help="winner profile: drink=navierstocks/drink (0.748)")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--cup-conf", type=float, default=None)
    p.add_argument("--bottle-conf", type=float, default=None)
    p.add_argument("--can-conf", type=float, default=None)
    p.add_argument("--cup-bonus", type=float, default=None)
    p.add_argument("--bottle-bonus", type=float, default=None)
    p.add_argument("--can-bonus", type=float, default=None)
    args = p.parse_args()

    if args.profile == "bev1":
        conf_default = (0.68, 0.42, 0.48)
        bonus_default = (0.05, 0.10, 0.15)
        iou_nms = 0.5
        cross_iou = 0.8
        min_side = 12.0
    else:  # drink
        conf_default = DEFAULT_CONF
        bonus_default = DEFAULT_BONUS
        iou_nms = DEFAULT_IOU_NMS
        cross_iou = DEFAULT_CROSS_IOU
        min_side = DEFAULT_MIN_SIDE

    conf_thr = np.array([
        args.cup_conf    if args.cup_conf    is not None else conf_default[0],
        args.bottle_conf if args.bottle_conf is not None else conf_default[1],
        args.can_conf    if args.can_conf    is not None else conf_default[2],
    ], dtype=np.float32)
    bonus = np.array([
        args.cup_bonus    if args.cup_bonus    is not None else bonus_default[0],
        args.bottle_bonus if args.bottle_bonus is not None else bonus_default[1],
        args.can_bonus    if args.can_bonus    is not None else bonus_default[2],
    ], dtype=np.float32)

    use_tta = not args.no_tta

    print(f"[cfg] profile={args.profile} imgsz={args.imgsz} tta={use_tta}")
    print(f"[cfg] conf  cup={conf_thr[0]:.3f} bottle={conf_thr[1]:.3f} can={conf_thr[2]:.3f}")
    print(f"[cfg] bonus cup={bonus[0]:.3f} bottle={bonus[1]:.3f} can={bonus[2]:.3f}")
    print(f"[cfg] iou_nms={iou_nms} cross_iou={cross_iou} min_side={min_side}")

    image_paths, class_names = load_yolo_dataset(args.data)
    gt_boxes, _ = load_yolo_ground_truth(args.data)
    print(f"[data] images={len(image_paths)} gt={len(gt_boxes)} classes={class_names}")

    yolo = YOLO(str(args.model))

    pred_boxes: list[Box] = []
    for img_path in image_paths:
        dets = winner_predict_one(
            yolo, img_path, args.imgsz, conf_thr, bonus,
            iou_nms, cross_iou, min_side, DEFAULT_MIN_AREA,
            DEFAULT_MAX_AR, DEFAULT_MAX_DET, use_tta,
        )
        for xyxy, score, cls in dets:
            pred_boxes.append(Box(
                image_id=img_path.stem,
                cls=int(cls),
                xyxy=xyxy,
                conf=float(score),
            ))

    m = evaluate_score(gt_boxes, pred_boxes, class_names)
    manifest = 0.6 * m.map50 + 0.4 * m.fp_score
    print()
    print("=" * 64)
    print(f"  WINNER-STYLE INFERENCE  --  model={args.model.name}")
    print("=" * 64)
    print(f"  gt={m.gt}  preds={m.predictions}")
    print(f"  map50          : {m.map50:.4f}")
    print(f"  precision      : {m.precision:.4f}")
    print(f"  recall         : {m.recall:.4f}")
    print(f"  false_positive : {m.fp_score:.4f}")
    print()
    print(f"  manifest score : {manifest:.4f}   (0.6*map50 + 0.4*fp_score)")
    print()
    print(f"  per-class:")
    for c in m.classes:
        print(f"    {c.name:8s}: ap50={c.ap50:.3f} prec={c.precision:.3f} "
              f"rec={c.recall:.3f}  tp={c.tp} fp={c.fp} fn={c.fn}")


if __name__ == "__main__":
    main()
