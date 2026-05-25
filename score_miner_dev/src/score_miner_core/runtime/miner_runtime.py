from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import supervision as sv

from score_miner_core.detector.base import DetectorBase
from score_miner_core.runtime.memory_budget import MemoryBudget
from score_miner_core.runtime.postprocess import PostprocessConfig, filter_boxes_by_config
from score_miner_core.runtime.role_cleanup import RoleCleanupConfig, cleanup_roles_by_color
from score_miner_core.runtime.team_color import TeamColorConfig, TeamColorMemory, assign_team_ids_by_color
from score_miner_core.runtime.tracking import DetectionTracker, TrackingConfig

VALID_INPUT_COLOR_SPACES = {"bgr", "rgb"}


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
    track_id: int | None = Field(default=None, exclude=True)


class TVFrameResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: int
    boxes: list[BoundingBox]
    keypoints: list[tuple[int, int]]


class MinerRuntime:
    """Safe runtime shell for plugging detector candidates behind one interface."""

    def __init__(
        self,
        path_hf_repo: str | Path,
        detector: DetectorBase | None = None,
        postprocess_config: PostprocessConfig | None = None,
        team_color_config: TeamColorConfig | None = None,
        tracking_config: TrackingConfig | None = None,
        role_cleanup_config: RoleCleanupConfig | None = None,
        input_color_space: str = "bgr",
    ) -> None:
        normalized_color_space = input_color_space.strip().lower()
        if normalized_color_space not in VALID_INPUT_COLOR_SPACES:
            raise ValueError(
                f"input_color_space must be one of {sorted(VALID_INPUT_COLOR_SPACES)}, "
                f"got {input_color_space!r}."
            )
        self.path_hf_repo = Path(path_hf_repo)
        self.memory_budget = MemoryBudget()
        self.detector = detector
        self.input_color_space = normalized_color_space
        self.postprocess_config = postprocess_config or PostprocessConfig()
        self.team_color_config = team_color_config or TeamColorConfig()
        self.team_color_memory = TeamColorMemory(self.team_color_config)
        self.tracking_config = tracking_config or TrackingConfig()
        self.tracker = DetectionTracker(self.tracking_config)
        self.role_cleanup_config = role_cleanup_config or RoleCleanupConfig()
        self.load_status: dict[str, Any] = {
            "path_hf_repo": str(self.path_hf_repo),
            "memory": self.memory_budget.status(),
            "detector": type(detector).__name__ if detector else None,
            "postprocess": self.postprocess_config.model_dump(mode="json"),
            "team_color": self.team_color_config.model_dump(mode="json"),
            "tracking": self.tracking_config.model_dump(mode="json"),
            "role_cleanup": self.role_cleanup_config.model_dump(mode="json"),
            "input_color_space": self.input_color_space,
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
        rgb_batch = self._to_rgb_batch(batch_images)
        detections_by_frame = self._predict_boxes(rgb_batch)
        empty_keypoints = [(0, 0) for _ in range(n_keypoints)]
        results: list[TVFrameResult] = []
        for frame_idx, image_rgb in enumerate(rgb_batch):
            boxes = filter_boxes_by_config(
                _detections_to_boxes(detections_by_frame[frame_idx]),
                self.postprocess_config,
            )
            boxes = assign_team_ids_by_color(image_rgb, boxes, self.team_color_config)
            boxes = self.team_color_memory.stabilize(boxes)
            boxes = cleanup_roles_by_color(
                image_rgb,
                boxes,
                role_config=self.role_cleanup_config,
                team_config=self.team_color_config,
            )
            results.append(
                TVFrameResult(
                    frame_id=offset + frame_idx,
                    boxes=boxes,
                    keypoints=empty_keypoints,
                )
            )
        return results

    def _to_rgb_batch(self, batch_images: list[np.ndarray]) -> list[np.ndarray]:
        if self.input_color_space == "rgb":
            return batch_images
        return [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in batch_images]

    def _predict_boxes(self, batch_images: list[np.ndarray]) -> list[sv.Detections]:
        if self.detector is None:
            return [sv.Detections.empty() for _ in batch_images]
        predictions = self.detector.predict(batch_images)
        if len(predictions) != len(batch_images):
            raise ValueError(
                f"Detector returned {len(predictions)} predictions for {len(batch_images)} images."
            )
        return self.tracker.update(predictions)


def _detections_to_boxes(detections: sv.Detections) -> list[BoundingBox]:
    boxes: list[BoundingBox] = []
    class_ids = detections.class_id
    confidences = detections.confidence
    tracker_ids = detections.tracker_id
    for idx, xyxy in enumerate(detections.xyxy):
        if class_ids is None:
            continue
        confidence = 1.0 if confidences is None else float(confidences[idx])
        track_id = None
        if tracker_ids is not None and int(tracker_ids[idx]) >= 0:
            track_id = int(tracker_ids[idx])
        x1, y1, x2, y2 = xyxy.tolist()
        boxes.append(
            BoundingBox(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                cls_id=int(class_ids[idx]),
                conf=confidence,
                track_id=track_id,
            )
        )
    return boxes
