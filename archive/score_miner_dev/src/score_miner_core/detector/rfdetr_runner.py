from __future__ import annotations

from typing import Any

import numpy as np
import supervision as sv

from score_miner_core.detector.base import DetectorBase
from score_miner_core.detector.class_id_mapper import DetectionClassIdMapper


class RFDETRRunner(DetectorBase):
    """RF-DETR adapter using the official `rfdetr` package."""

    name = "rfdetr"

    def __init__(
        self,
        model_size: str = "medium",
        threshold: float = 0.35,
        model: Any | None = None,
        class_id_mapper: DetectionClassIdMapper | None = None,
        optimize_for_inference: bool = True,
    ) -> None:
        self.threshold = threshold
        self.model = model or _create_rfdetr_model(model_size)
        if optimize_for_inference:
            _optimize_for_inference(self.model)
        self.class_id_mapper = class_id_mapper

    def predict(self, batch_images: list[np.ndarray]) -> list[sv.Detections]:
        # RF-DETR's public API is image-oriented; keep batching at our router
        # level until the library exposes a stable batched API.
        detections = [
            _coerce_to_detections(self.model.predict(image, threshold=self.threshold))
            for image in batch_images
        ]
        if self.class_id_mapper is None:
            return detections
        return [self.class_id_mapper.apply(frame_detections) for frame_detections in detections]


def _create_rfdetr_model(model_size: str):
    try:
        import rfdetr
    except ImportError as exc:
        raise ImportError(
            "RF-DETR runner requires the optional `rfdetr` package. "
            "Install `score-miner-core[rfdetr]` or add `rfdetr` to chute_config.yml."
        ) from exc

    size = model_size.strip().lower().replace("-", "_")
    class_by_size = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "m": "RFDETRMedium",
        "large": "RFDETRLarge",
        "l": "RFDETRLarge",
    }
    class_name = class_by_size.get(size, model_size)
    try:
        model_cls = getattr(rfdetr, class_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported RF-DETR model size/class: {model_size}") from exc
    return model_cls()


def _optimize_for_inference(model: Any) -> None:
    optimize = getattr(model, "optimize_for_inference", None)
    if callable(optimize):
        optimize()


def _coerce_to_detections(result: Any) -> sv.Detections:
    if isinstance(result, sv.Detections):
        return result
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], sv.Detections):
        return result[0]
    try:
        return sv.Detections.from_inference(result)
    except Exception as exc:
        raise TypeError(f"Unsupported RF-DETR prediction type: {type(result).__name__}") from exc
