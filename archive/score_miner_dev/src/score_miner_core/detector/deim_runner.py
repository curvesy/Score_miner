from __future__ import annotations

import numpy as np
import supervision as sv

from score_miner_core.detector.base import DetectorBase


class DEIMRunner(DetectorBase):
    """DEIMv2 adapter placeholder."""

    name = "deimv2"

    def predict(self, batch_images: list[np.ndarray]) -> list[sv.Detections]:
        raise NotImplementedError(
            "DEIMv2 does not expose a stable pip-level inference API here yet. "
            "Wire the official repo/Hugging Face checkpoint behind `sv.Detections`."
        )
