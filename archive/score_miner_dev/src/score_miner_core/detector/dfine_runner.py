from __future__ import annotations

import numpy as np
import supervision as sv

from score_miner_core.detector.base import DetectorBase


class DFineRunner(DetectorBase):
    """D-FINE adapter placeholder."""

    name = "dfine"

    def predict(self, batch_images: list[np.ndarray]) -> list[sv.Detections]:
        raise NotImplementedError(
            "D-FINE adapter should wrap the official repo/exported ONNX/TensorRT output "
            "and return `sv.Detections`."
        )
