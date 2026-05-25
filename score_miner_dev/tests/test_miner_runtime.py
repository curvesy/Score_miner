import numpy as np
import supervision as sv

from score_miner_core.detector.base import DetectorBase
from score_miner_core.runtime.miner_runtime import MinerRuntime
from score_miner_core.runtime.postprocess import PostprocessConfig
from score_miner_core.runtime.team_color import TeamColorConfig


def test_runtime_returns_one_frame_per_input() -> None:
    runtime = MinerRuntime(".")
    frames = [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)]

    results = runtime.predict_batch(frames, offset=10, n_keypoints=4)

    assert [frame.frame_id for frame in results] == [10, 11]
    assert results[0].boxes == []
    assert results[0].keypoints == [(0, 0), (0, 0), (0, 0), (0, 0)]


def test_runtime_converts_supervision_detections() -> None:
    runtime = MinerRuntime(
        ".",
        detector=_StaticDetector(),
        postprocess_config=PostprocessConfig(confidence_threshold=0.0),
        team_color_config=TeamColorConfig(enabled=False),
    )

    result = runtime.predict_batch([np.zeros((8, 8, 3), dtype=np.uint8)], offset=3, n_keypoints=2)[0]

    assert result.frame_id == 3
    assert len(result.boxes) == 1
    assert result.boxes[0].x1 == 1
    assert result.boxes[0].cls_id == 2
    assert result.boxes[0].conf == 0.75


def test_runtime_postprocess_filters_threshold_and_top_k() -> None:
    runtime = MinerRuntime(
        ".",
        detector=_MultiBoxDetector(),
        postprocess_config=PostprocessConfig(
            confidence_threshold=0.5,
            max_boxes_per_frame=2,
            min_box_area=1,
        ),
        team_color_config=TeamColorConfig(enabled=False),
    )

    result = runtime.predict_batch([np.zeros((8, 8, 3), dtype=np.uint8)], offset=0, n_keypoints=0)[0]

    assert [box.conf for box in result.boxes] == [0.9, 0.8]
    assert [box.x1 for box in result.boxes] == [10, 20]


def test_runtime_converts_bgr_input_to_rgb_before_detector() -> None:
    detector = _CaptureFirstPixelDetector()
    runtime = MinerRuntime(
        ".",
        detector=detector,
        postprocess_config=PostprocessConfig(confidence_threshold=0.0),
        team_color_config=TeamColorConfig(enabled=False),
    )
    bgr_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr_frame[0, 0] = np.array([10, 20, 200], dtype=np.uint8)

    runtime.predict_batch([bgr_frame], offset=0, n_keypoints=0)

    assert detector.first_pixel == (200, 20, 10)


def test_runtime_keeps_rgb_input_when_configured() -> None:
    detector = _CaptureFirstPixelDetector()
    runtime = MinerRuntime(
        ".",
        detector=detector,
        postprocess_config=PostprocessConfig(confidence_threshold=0.0),
        team_color_config=TeamColorConfig(enabled=False),
        input_color_space="rgb",
    )
    rgb_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb_frame[0, 0] = np.array([200, 20, 10], dtype=np.uint8)

    runtime.predict_batch([rgb_frame], offset=0, n_keypoints=0)

    assert detector.first_pixel == (200, 20, 10)


def test_runtime_rejects_unknown_input_color_space() -> None:
    try:
        MinerRuntime(".", input_color_space="yuv")
    except ValueError as exc:
        assert "input_color_space" in str(exc)
    else:
        raise AssertionError("expected invalid color space to fail")


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


class _CaptureFirstPixelDetector(DetectorBase):
    def __init__(self) -> None:
        self.first_pixel: tuple[int, int, int] | None = None

    def predict(self, batch_images: list[np.ndarray]) -> list[sv.Detections]:
        pixel = batch_images[0][0, 0]
        self.first_pixel = tuple(int(channel) for channel in pixel.tolist())
        return [sv.Detections.empty() for _ in batch_images]


class _MultiBoxDetector(DetectorBase):
    def predict(self, batch_images: list[np.ndarray]) -> list[sv.Detections]:
        return [
            sv.Detections(
                xyxy=np.array(
                    [
                        [0, 0, 10, 10],
                        [10, 10, 20, 20],
                        [20, 20, 30, 30],
                        [30, 30, 30, 30],
                    ],
                    dtype=float,
                ),
                confidence=np.array([0.4, 0.9, 0.8, 0.99], dtype=float),
                class_id=np.array([0, 0, 0, 0], dtype=int),
            )
            for _ in batch_images
        ]
