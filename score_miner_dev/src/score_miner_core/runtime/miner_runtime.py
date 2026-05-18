from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
import supervision as sv

from score_miner_core.detector.base import DetectorBase
from score_miner_core.runtime.memory_budget import MemoryBudget


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    conf: float
    team_id: int | str | None = None
    cluster_id: int | str | None = None


class TVFrameResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: int
    boxes: list[BoundingBox]
    keypoints: list[tuple[int, int]]


class MinerRuntime:
    """Safe runtime shell for plugging detector candidates behind one interface."""

    def __init__(self, path_hf_repo: str | Path, detector: DetectorBase | None = None) -> None:
        self.path_hf_repo = Path(path_hf_repo)
        self.memory_budget = MemoryBudget()
        self.detector = detector
        self.load_status: dict[str, Any] = {
            "path_hf_repo": str(self.path_hf_repo),
            "memory": self.memory_budget.status(),
            "detector": type(detector).__name__ if detector else None,
        }

    def __repr__(self) -> str:
        return f"MinerRuntime(status={self.load_status})"

    def predict_batch(
        self,
        batch_images: list[np.ndarray],
        offset: int,
        n_keypoints: int,
    ) -> list[TVFrameResult]:
        self.memory_budget.assert_within_limit()
        detections_by_frame = self._predict_boxes(batch_images)
        empty_keypoints = [(0, 0) for _ in range(n_keypoints)]
        return [
            TVFrameResult(
                frame_id=offset + frame_idx,
                boxes=_detections_to_boxes(detections_by_frame[frame_idx]),
                keypoints=empty_keypoints,
            )
            for frame_idx, _image in enumerate(batch_images)
        ]

    def _predict_boxes(self, batch_images: list[np.ndarray]) -> list[sv.Detections]:
        if self.detector is None:
            return [sv.Detections.empty() for _ in batch_images]
        predictions = self.detector.predict(batch_images)
        if len(predictions) != len(batch_images):
            raise ValueError(
                f"Detector returned {len(predictions)} predictions for {len(batch_images)} images."
            )
        return predictions


def _detections_to_boxes(detections: sv.Detections) -> list[BoundingBox]:
    boxes: list[BoundingBox] = []
    class_ids = detections.class_id
    confidences = detections.confidence
    for idx, xyxy in enumerate(detections.xyxy):
        if class_ids is None:
            continue
        confidence = 1.0 if confidences is None else float(confidences[idx])
        x1, y1, x2, y2 = xyxy.tolist()
        boxes.append(
            BoundingBox(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                cls_id=int(class_ids[idx]),
                conf=confidence,
            )
        )
    return boxes
