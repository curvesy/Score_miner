import numpy as np
import supervision as sv

from score_miner_core.runtime.tracking import DetectionTracker, TrackingConfig


def test_tracking_disabled_returns_original_detections() -> None:
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 10, 10]], dtype=float),
        confidence=np.array([0.9], dtype=float),
        class_id=np.array([0], dtype=int),
    )
    tracker = DetectionTracker(TrackingConfig(enabled=False))

    assert tracker.update([detections])[0] is detections


def test_tracking_assigns_tracker_ids() -> None:
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 10, 10]], dtype=float),
        confidence=np.array([0.9], dtype=float),
        class_id=np.array([0], dtype=int),
    )
    tracker = DetectionTracker(TrackingConfig(enabled=True, minimum_consecutive_frames=1))

    tracked = tracker.update([detections])[0]

    assert tracked.tracker_id is not None
    assert len(tracked.tracker_id) == 1


def test_tracking_preserves_untracked_detections() -> None:
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 10, 10], [100, 100, 110, 110]], dtype=float),
        confidence=np.array([0.9, 0.1], dtype=float),
        class_id=np.array([0, 0], dtype=int),
    )
    tracker = DetectionTracker(TrackingConfig(enabled=True, minimum_consecutive_frames=1))

    tracked = tracker.update([detections])[0]

    assert len(tracked) == 2
    assert tracked.tracker_id is not None
    assert len(tracked.tracker_id) == 2
