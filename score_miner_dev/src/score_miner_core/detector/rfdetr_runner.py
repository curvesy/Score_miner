from __future__ import annotations

from typing import Any

import numpy as np
import supervision as sv

from score_miner_core.detector.base import DetectorBase


class RFDETRRunner(DetectorBase):
    """RF-DETR adapter using the official `rfdetr` package."""

    name = "rfdetr"

    def __init__(
        self,
        model_size: str = "medium",
        threshold: float = 0.35,
        model: Any | None = None,
    ) -> None:
        self.threshold = threshold
        self.model = model or _create_rfdetr_model(model_size)

    def predict(self, batch_images: list[np.ndarray]) -> list[sv.Detections]:
        # RF-DETR's public API is image-oriented; keep batching at our router
        # level until the library exposes a stable batched API.
        return [_coerce_to_detections(self.model.predict(image, threshold=self.threshold)) for image in batch_images]


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


def _coerce_to_detections(result: Any) -> sv.Detections:
    if isinstance(result, sv.Detections):
        return result
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], sv.Detections):
        return result[0]
    try:
        return sv.Detections.from_inference(result)
    except Exception as exc:
        raise TypeError(f"Unsupported RF-DETR prediction type: {type(result).__name__}") from exc
