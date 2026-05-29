"""Build minimal public Detect deploy repos."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from public_detect.export_utils import check_size_gate, write_json


MINER_TEMPLATE = r'''from pathlib import Path
import math

import cv2
import numpy as np
import onnxruntime as ort
from numpy import ndarray
from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    conf: float


class TVFrameResult(BaseModel):
    frame_id: int
    boxes: list[BoundingBox]
    keypoints: list[tuple[int, int]]


class Miner:
    """Winner-style ONNX miner.

    Pipeline (per frame):
      preprocess (letterbox + normalize)
      -> ONNX inference (optionally on hflip too -> TTA)
      -> per-class conf filter WITH rescue bonus (top-1 admit when class is empty)
      -> sane-box filter (min side, min area, max aspect ratio, < 95% image)
      -> per-class hard NMS
      -> cluster-max same-class score boost (TTA mode)
      -> cross-class dedup ordered by (score - per-class threshold) margin
    """

    class_names = __CLASS_NAMES__
    input_size = __INPUT_SIZE__
    iou_thres = __IOU_THRES__
    cross_iou_thres = __CROSS_IOU_THRES__
    min_side = __MIN_SIDE__
    min_box_area = __MIN_BOX_AREA__
    max_aspect_ratio = __MAX_ASPECT_RATIO__
    max_det = __MAX_DET__
    use_tta = __USE_TTA__
    conf_thres = np.array(__CONF_THRESHOLDS__, dtype=np.float32)
    rescue_bonus = np.array(__RESCUE_BONUS__, dtype=np.float32)

    def __init__(self, path_hf_repo: Path) -> None:
        model_path = path_hf_repo / "weights.onnx"
        try:
            ort.preload_dlls()
        except Exception:
            pass
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        providers = [provider for provider in providers if provider in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        shape = self.session.get_inputs()[0].shape
        self.input_height = self._safe_dim(shape[2], self.input_size)
        self.input_width = self._safe_dim(shape[3], self.input_size)

    def __repr__(self) -> str:
        return f"ONNXRuntime(providers={self.session.get_providers()}, input={self.input_width}x{self.input_height}, tta={self.use_tta})"

    @staticmethod
    def _safe_dim(value, default: int) -> int:
        return int(value) if isinstance(value, int) and value > 0 else default

    def _letterbox(self, image: ndarray) -> tuple[ndarray, float, tuple[float, float], tuple[int, int]]:
        h, w = image.shape[:2]
        ratio = min(self.input_width / w, self.input_height / h)
        resized_w = int(round(w * ratio))
        resized_h = int(round(h * ratio))
        if (resized_w, resized_h) != (w, h):
            interp = cv2.INTER_CUBIC if ratio > 1.0 else cv2.INTER_LINEAR
            image = cv2.resize(image, (resized_w, resized_h), interpolation=interp)
        dw = (self.input_width - resized_w) / 2.0
        dh = (self.input_height - resized_h) / 2.0
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, ratio, (dw, dh), (w, h)

    def _preprocess(self, image: ndarray) -> tuple[np.ndarray, float, tuple[float, float], tuple[int, int]]:
        image, ratio, pad, original_size = self._letterbox(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = image.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(tensor, dtype=np.float32), ratio, pad, original_size

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        out = np.empty_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return out

    @staticmethod
    def _clip_boxes(boxes: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        w, h = size
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        return boxes

    @staticmethod
    def _hard_nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
        if len(boxes) == 0:
            return np.empty(0, dtype=np.intp)
        order = np.argsort(-scores)
        keep = []
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        while len(order):
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
            iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
            order = rest[iou <= iou_thresh]
        return np.asarray(keep, dtype=np.intp)

    def _per_class_hard_nms(self, boxes, scores, cls_ids, iou_thresh):
        if len(boxes) == 0:
            return np.empty(0, dtype=np.intp)
        all_keep = []
        for c in np.unique(cls_ids):
            mask = cls_ids == c
            indices = np.where(mask)[0]
            keep = self._hard_nms(boxes[mask], scores[mask], iou_thresh)
            all_keep.extend(indices[keep].tolist())
        all_keep.sort()
        return np.asarray(all_keep, dtype=np.intp)

    def _conf_filter_mask(self, scores: np.ndarray, cls_ids: np.ndarray) -> np.ndarray:
        """Per-class conf gate with per-class top-1 rescue when a class is empty."""
        if len(scores) == 0:
            return np.zeros(0, dtype=bool)
        keep = scores >= self.conf_thres[cls_ids]
        for c in np.unique(cls_ids):
            bonus = float(self.rescue_bonus[c])
            if bonus <= 0.0:
                continue
            cm = cls_ids == c
            if keep[cm].any():
                continue
            idx = np.where(cm)[0]
            top = int(idx[int(np.argmax(scores[idx]))])
            if scores[top] >= self.conf_thres[c] - bonus:
                keep[top] = True
        return keep

    def _cross_class_dedup(self, boxes, scores, cls_ids, iou_thresh):
        n = len(boxes)
        if n <= 1:
            return boxes, scores, cls_ids
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        margins = scores - self.conf_thres[cls_ids]
        order = np.lexsort((-areas, -margins))
        suppressed = np.zeros(n, dtype=bool)
        keep = []
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
        keep_idx = np.asarray(keep, dtype=np.intp)
        return boxes[keep_idx], scores[keep_idx], cls_ids[keep_idx]

    def _filter_boxes(self, boxes, scores, cls_ids, image_size):
        if len(boxes) == 0:
            return boxes, scores, cls_ids
        bw = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        bh = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        area = bw * bh
        aspect = np.where((bw > 0) & (bh > 0), np.maximum(bw / np.maximum(bh, 1e-6), bh / np.maximum(bw, 1e-6)), np.inf)
        image_area = float(image_size[0] * image_size[1])
        keep = (
            (bw >= self.min_side)
            & (bh >= self.min_side)
            & (area >= self.min_box_area)
            & (area <= 0.95 * image_area)
            & (aspect <= self.max_aspect_ratio)
        )
        return boxes[keep], scores[keep], cls_ids[keep]

    @staticmethod
    def _max_score_per_cluster(post_boxes, post_cls, full_boxes, full_scores, full_cls, iou_thresh):
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

    def _decode_one_view(self, output, ratio, pad, image_size):
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        if output.ndim != 2:
            raise ValueError(f"Unexpected ONNX output shape: {output.shape}")
        if output.shape[1] >= 6 and output.shape[1] == 6:
            # already final detections: [x1,y1,x2,y2,score,cls]
            boxes = output[:, :4].astype(np.float32)
            scores = output[:, 4].astype(np.float32)
            cls_ids = output[:, 5].astype(np.int32)
            boxes_xyxy_already = True
        else:
            if output.shape[0] == 4 + len(self.class_names):
                output = output.T
            if output.shape[1] < 5:
                raise ValueError(f"Unexpected ONNX output shape: {output.shape}")
            boxes_xywh = output[:, :4].astype(np.float32)
            class_scores = output[:, 4 : 4 + len(self.class_names)].astype(np.float32)
            cls_ids = np.argmax(class_scores, axis=1).astype(np.int32)
            scores = class_scores[np.arange(len(class_scores)), cls_ids]
            boxes = self._xywh_to_xyxy(boxes_xywh)
            boxes_xyxy_already = False

        keep = self._conf_filter_mask(scores, cls_ids)
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        if len(scores) == 0:
            return boxes, scores, cls_ids

        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes /= ratio
        boxes = self._clip_boxes(boxes, image_size)
        boxes, scores, cls_ids = self._filter_boxes(boxes, scores, cls_ids, image_size)
        return boxes, scores, cls_ids

    def _infer_view(self, image):
        tensor, ratio, pad, image_size = self._preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return self._decode_one_view(outputs[0], ratio, pad, image_size), image_size

    def _predict_one(self, image: ndarray) -> list[BoundingBox]:
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        (boxes_o, scores_o, cls_o), image_size = self._infer_view(image)
        if self.use_tta:
            flipped = cv2.flip(image, 1)
            (boxes_f, scores_f, cls_f), _ = self._infer_view(flipped)
            if len(boxes_f) > 0:
                w = image_size[0]
                x1 = w - boxes_f[:, 2]
                x2 = w - boxes_f[:, 0]
                boxes_f = np.stack([x1, boxes_f[:, 1], x2, boxes_f[:, 3]], axis=1).astype(np.float32)
            boxes = np.concatenate([boxes_o, boxes_f], axis=0) if len(boxes_f) else boxes_o
            scores = np.concatenate([scores_o, scores_f], axis=0) if len(scores_f) else scores_o
            cls_ids = np.concatenate([cls_o, cls_f], axis=0) if len(cls_f) else cls_o
        else:
            boxes, scores, cls_ids = boxes_o, scores_o, cls_o

        if len(boxes) == 0:
            return []

        full_boxes, full_scores, full_cls = boxes.copy(), scores.copy(), cls_ids.copy()

        keep = self._per_class_hard_nms(boxes, scores, cls_ids, self.iou_thres)
        if len(keep) == 0:
            return []
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]

        if len(boxes) > self.max_det:
            top = np.argsort(-scores)[: self.max_det]
            boxes, scores, cls_ids = boxes[top], scores[top], cls_ids[top]

        if self.use_tta and len(boxes) > 0:
            scores = self._max_score_per_cluster(boxes, cls_ids, full_boxes, full_scores, full_cls, self.iou_thres)

        if len(boxes) > 1:
            boxes, scores, cls_ids = self._cross_class_dedup(boxes, scores, cls_ids, self.cross_iou_thres)

        out = []
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = box.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(
                BoundingBox(
                    x1=int(math.floor(x1)),
                    y1=int(math.floor(y1)),
                    x2=int(math.ceil(x2)),
                    y2=int(math.ceil(y2)),
                    cls_id=int(cls_id),
                    conf=float(score),
                )
            )
        return out

    def predict_batch(self, batch_images: list[ndarray], offset: int, n_keypoints: int) -> list[TVFrameResult]:
        results = []
        keypoints = [(0, 0) for _ in range(max(0, int(n_keypoints)))]
        for index, image in enumerate(batch_images):
            try:
                boxes = self._predict_one(image)
            except Exception as exc:
                print(f"inference failed frame={offset + index}: {type(exc).__name__}: {exc}")
                boxes = []
            results.append(TVFrameResult(frame_id=offset + index, boxes=boxes, keypoints=keypoints))
        return results
'''

CHUTE_CONFIG_TEMPLATE = """Image:
  from_base: parachutes/python:3.12
  run_command:
    - pip install --upgrade pip
    - pip install 'numpy>=2.0' 'opencv-python-headless>=4.10' 'pydantic>=2.12' 'onnxruntime-gpu>=1.20'
    - python -c "import cv2, numpy, onnxruntime, pydantic; print('public-detect deps ok')"

NodeSelector:
  gpu_count: 1
  min_vram_gb_per_gpu: 16
  exclude:
    - a100
    - h100
    - h200
    - b200
    - h20
    - mi300x
    - "5090"

Chute:
  shutdown_after_seconds: 300
  concurrency: 4
  max_instances: 1
  scaling_threshold: 0.75
"""


def build_deploy_repo(
    *,
    weights: str | Path,
    output_dir: str | Path,
    class_names: list[str],
    input_size: int,
    conf_thresholds: list[float],
    rescue_bonus: list[float] | None = None,
    use_tta: bool = True,
    max_det: int = 300,
    iou_thres: float = 0.4,
    cross_iou_thres: float = 0.7,
    min_side: float = 8.0,
    min_box_area: float = 100.0,
    max_aspect_ratio: float = 10.0,
    max_mb: float = 30.0,
) -> dict[str, Any]:
    if rescue_bonus is None:
        rescue_bonus = [0.0] * len(class_names)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target_weights = output / "weights.onnx"
    if Path(weights).resolve() != target_weights.resolve():
        shutil.copy2(weights, target_weights)
    miner = render_miner(
        class_names=class_names,
        input_size=input_size,
        conf_thresholds=conf_thresholds,
        rescue_bonus=rescue_bonus,
        use_tta=use_tta,
        max_det=max_det,
        iou_thres=iou_thres,
        cross_iou_thres=cross_iou_thres,
        min_side=min_side,
        min_box_area=min_box_area,
        max_aspect_ratio=max_aspect_ratio,
    )
    (output / "miner.py").write_text(miner)
    (output / "chute_config.yml").write_text(CHUTE_CONFIG_TEMPLATE)
    config = {
        "class_names": class_names,
        "input_size": input_size,
        "conf_thresholds": conf_thresholds,
        "rescue_bonus": rescue_bonus,
        "use_tta": use_tta,
        "max_det": max_det,
        "iou_thres": iou_thres,
        "cross_iou_thres": cross_iou_thres,
        "min_side": min_side,
        "min_box_area": min_box_area,
        "max_aspect_ratio": max_aspect_ratio,
    }
    write_json(output / "miner_config.json", config)
    report = check_size_gate(output, max_mb=max_mb)
    write_json(output / "size_report.json", report)
    return report


def render_miner(
    *,
    class_names: list[str],
    input_size: int,
    conf_thresholds: list[float],
    rescue_bonus: list[float],
    use_tta: bool,
    max_det: int,
    iou_thres: float,
    cross_iou_thres: float,
    min_side: float,
    min_box_area: float,
    max_aspect_ratio: float,
) -> str:
    if len(class_names) != len(conf_thresholds):
        raise ValueError("class_names and conf_thresholds must have the same length")
    if len(class_names) != len(rescue_bonus):
        raise ValueError("class_names and rescue_bonus must have the same length")
    replacements = {
        "__CLASS_NAMES__": repr(class_names),
        "__INPUT_SIZE__": str(int(input_size)),
        "__CONF_THRESHOLDS__": repr([float(item) for item in conf_thresholds]),
        "__RESCUE_BONUS__": repr([float(item) for item in rescue_bonus]),
        "__USE_TTA__": "True" if use_tta else "False",
        "__MAX_DET__": str(int(max_det)),
        "__IOU_THRES__": repr(float(iou_thres)),
        "__CROSS_IOU_THRES__": repr(float(cross_iou_thres)),
        "__MIN_SIDE__": repr(float(min_side)),
        "__MIN_BOX_AREA__": repr(float(min_box_area)),
        "__MAX_ASPECT_RATIO__": repr(float(max_aspect_ratio)),
    }
    text = MINER_TEMPLATE
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
