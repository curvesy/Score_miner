from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import supervision as sv


class DetectorBase(ABC):
    """Common interface for RF-DETR, DEIMv2, D-FINE, and fallback detectors.

    The internal detection representation is `supervision.Detections`, which is
    already supported by RF-DETR and has converters for Transformers,
    Ultralytics, Roboflow Inference, Detectron2, MMDetection, and more.
    """

    name: str = "detector"

    @abstractmethod
    def predict(self, batch_images: list[np.ndarray]) -> list[sv.Detections]:
        """Return one `sv.Detections` object per input image."""
