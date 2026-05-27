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
    class_names = __CLASS_NAMES__
    input_size = __INPUT_SIZE__
    iou_thres = __IOU_THRES__
    cross_iou_thres = __CROSS_IOU_THRES__
    min_side = __MIN_SIDE__
    min_box_area = __MIN_BOX_AREA__
    max_aspect_ratio = __MAX_ASPECT_RATIO__
    max_det = __MAX_DET__
    conf_thres = np.array(__CONF_THRESHOLDS__, dtype=np.float32)

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
        return f"ONNXRuntime(providers={self.session.get_providers()}, input={self.input_width}x{self.input_height})"

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
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
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

    def _filter_boxes(self, boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _postprocess(self, output: np.ndarray, ratio: float, pad: tuple[float, float], image_size: tuple[int, int]) -> list[BoundingBox]:
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        if output.ndim != 2:
            raise ValueError(f"Unexpected ONNX output shape: {output.shape}")
        if output.shape[0] == 4 + len(self.class_names):
            output = output.T
        if output.shape[1] < 5:
            raise ValueError(f"Unexpected ONNX output shape: {output.shape}")
        boxes_xywh = output[:, :4].astype(np.float32)
        class_scores = output[:, 4 : 4 + len(self.class_names)].astype(np.float32)
        cls_ids = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = class_scores[np.arange(len(class_scores)), cls_ids]
        keep = scores >= self.conf_thres[cls_ids]
        boxes_xywh, scores, cls_ids = boxes_xywh[keep], scores[keep], cls_ids[keep]
        if len(scores) == 0:
            return []
        boxes = self._xywh_to_xyxy(boxes_xywh)
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes /= ratio
        boxes = self._clip_boxes(boxes, image_size)
        boxes, scores, cls_ids = self._filter_boxes(boxes, scores, cls_ids, image_size)
        kept = []
        for cls_id in np.unique(cls_ids):
            idx = np.where(cls_ids == cls_id)[0]
            cls_keep = self._nms(boxes[idx], scores[idx], self.iou_thres)
            kept.extend(idx[cls_keep].tolist())
        if not kept:
            return []
        kept = np.asarray(kept, dtype=np.intp)
        if len(kept) > self.max_det:
            kept = kept[np.argsort(-scores[kept])[: self.max_det]]
        out = []
        for box, score, cls_id in zip(boxes[kept], scores[kept], cls_ids[kept]):
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

    def _predict_one(self, image: ndarray) -> list[BoundingBox]:
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        tensor, ratio, pad, image_size = self._preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return self._postprocess(outputs[0], ratio, pad, image_size)

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


def build_deploy_repo(
    *,
    weights: str | Path,
    output_dir: str | Path,
    class_names: list[str],
    input_size: int,
    conf_thresholds: list[float],
    max_det: int = 20,
    iou_thres: float = 0.4,
    cross_iou_thres: float = 0.7,
    min_side: float = 4.0,
    min_box_area: float = 16.0,
    max_aspect_ratio: float = 12.0,
    max_mb: float = 30.0,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, output / "weights.onnx")
    miner = render_miner(
        class_names=class_names,
        input_size=input_size,
        conf_thresholds=conf_thresholds,
        max_det=max_det,
        iou_thres=iou_thres,
        cross_iou_thres=cross_iou_thres,
        min_side=min_side,
        min_box_area=min_box_area,
        max_aspect_ratio=max_aspect_ratio,
    )
    (output / "miner.py").write_text(miner)
    config = {
        "class_names": class_names,
        "input_size": input_size,
        "conf_thresholds": conf_thresholds,
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
    max_det: int,
    iou_thres: float,
    cross_iou_thres: float,
    min_side: float,
    min_box_area: float,
    max_aspect_ratio: float,
) -> str:
    if len(class_names) != len(conf_thresholds):
        raise ValueError("class_names and conf_thresholds must have the same length")
    replacements = {
        "__CLASS_NAMES__": repr(class_names),
        "__INPUT_SIZE__": str(int(input_size)),
        "__CONF_THRESHOLDS__": repr([float(item) for item in conf_thresholds]),
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
