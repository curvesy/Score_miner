import numpy as np
import supervision as sv

from score_miner_core.detector.base import DetectorBase
from score_miner_core.runtime.miner_runtime import MinerRuntime


def test_runtime_returns_one_frame_per_input() -> None:
    runtime = MinerRuntime(".")
    frames = [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)]

    results = runtime.predict_batch(frames, offset=10, n_keypoints=4)

    assert [frame.frame_id for frame in results] == [10, 11]
    assert results[0].boxes == []
    assert results[0].keypoints == [(0, 0), (0, 0), (0, 0), (0, 0)]


def test_runtime_converts_supervision_detections() -> None:
    runtime = MinerRuntime(".", detector=_StaticDetector())

    result = runtime.predict_batch([np.zeros((8, 8, 3), dtype=np.uint8)], offset=3, n_keypoints=2)[0]

    assert result.frame_id == 3
    assert len(result.boxes) == 1
    assert result.boxes[0].x1 == 1
    assert result.boxes[0].cls_id == 2
    assert result.boxes[0].conf == 0.75


class _StaticDetector(DetectorBase):
    def predict(self, batch_images: list[np.ndarray]) -> list[sv.Detections]:
        return [
            sv.Detections(
                xyxy=np.array([[1, 2, 3, 4]], dtype=float),
                confidence=np.array([0.75], dtype=float),
                class_id=np.array([2], dtype=int),
            )
            for _ in batch_images
        ]
