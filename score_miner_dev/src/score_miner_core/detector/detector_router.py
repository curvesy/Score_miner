from __future__ import annotations

from score_miner_core.detector.base import DetectorBase
from score_miner_core.detector.deim_runner import DEIMRunner
from score_miner_core.detector.dfine_runner import DFineRunner
from score_miner_core.detector.rfdetr_runner import RFDETRRunner


def create_detector(name: str) -> DetectorBase:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"rfdetr", "rf_detr", "rfdetr_l", "rfdetr_m"}:
        return RFDETRRunner()
    if normalized in {"deim", "deimv2", "deimv2_l"}:
        return DEIMRunner()
    if normalized in {"dfine", "d_fine", "dfine_l"}:
        return DFineRunner()
    raise ValueError(f"Unknown detector candidate: {name}")
