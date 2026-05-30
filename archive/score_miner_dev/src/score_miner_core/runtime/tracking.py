from __future__ import annotations

from os import getenv

from pydantic import BaseModel, ConfigDict, Field
import supervision as sv
import numpy as np


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    track_activation_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    lost_track_buffer: int = Field(default=30, ge=1)
    minimum_matching_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    tracker_assignment_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    frame_rate: float = Field(default=25.0, gt=0.0)
    minimum_consecutive_frames: int = Field(default=1, ge=1)

    @classmethod
    def from_env(cls) -> "TrackingConfig":
        return cls(
            enabled=_env_bool("SCORE_MINER_TRACKING_ENABLED", default=True),
            track_activation_threshold=float(getenv("SCORE_MINER_TRACK_ACTIVATION_THRESHOLD", "0.25")),
            lost_track_buffer=int(getenv("SCORE_MINER_LOST_TRACK_BUFFER", "30")),
            minimum_matching_threshold=float(getenv("SCORE_MINER_MINIMUM_MATCHING_THRESHOLD", "0.8")),
            tracker_assignment_iou_threshold=float(getenv("SCORE_MINER_TRACK_ASSIGNMENT_IOU", "0.5")),
            frame_rate=float(getenv("SCORE_MINER_TRACK_FRAME_RATE", "25.0")),
            minimum_consecutive_frames=int(getenv("SCORE_MINER_MINIMUM_CONSECUTIVE_FRAMES", "1")),
        )


class DetectionTracker:
    """Library-backed tracking stage.

    ByteTrack IDs are kept internal for team smoothing. TurboVision currently
    parses `cluster_id` as team color, so tracker IDs must not be emitted there.
    """

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self._tracker = None
        if config.enabled:
            self._tracker = sv.ByteTrack(
                track_activation_threshold=config.track_activation_threshold,
                lost_track_buffer=config.lost_track_buffer,
                minimum_matching_threshold=config.minimum_matching_threshold,
                frame_rate=config.frame_rate,
                minimum_consecutive_frames=config.minimum_consecutive_frames,
            )

    def update(self, detections_by_frame: list[sv.Detections]) -> list[sv.Detections]:
        if self._tracker is None:
            return detections_by_frame
        tracked_by_frame: list[sv.Detections] = []
        for detections in detections_by_frame:
            tracked = self._tracker.update_with_detections(detections)
            tracked_by_frame.append(_copy_tracker_ids_to_original(detections, tracked, self.config))
        return tracked_by_frame


def _copy_tracker_ids_to_original(
    original: sv.Detections,
    tracked: sv.Detections,
    config: TrackingConfig,
) -> sv.Detections:
    if len(original) == 0:
        return original

    tracker_ids = np.full((len(original),), -1, dtype=int)
    if len(tracked) == 0 or tracked.tracker_id is None:
        original.tracker_id = None
        return original

    used_tracked: set[int] = set()
    for original_idx, original_xyxy in enumerate(original.xyxy):
        best_idx = -1
        best_iou = 0.0
        for tracked_idx, tracked_xyxy in enumerate(tracked.xyxy):
            if tracked_idx in used_tracked:
                continue
            if not _same_class(original, tracked, original_idx, tracked_idx):
                continue
            iou = _iou(original_xyxy, tracked_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = tracked_idx
        if best_idx >= 0 and best_iou >= config.tracker_assignment_iou_threshold:
            tracker_ids[original_idx] = int(tracked.tracker_id[best_idx])
            used_tracked.add(best_idx)

    original.tracker_id = tracker_ids
    return original


def _same_class(
    original: sv.Detections,
    tracked: sv.Detections,
    original_idx: int,
    tracked_idx: int,
) -> bool:
    if original.class_id is None or tracked.class_id is None:
        return True
    return int(original.class_id[original_idx]) == int(tracked.class_id[tracked_idx])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _env_bool(name: str, *, default: bool) -> bool:
    raw = getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
