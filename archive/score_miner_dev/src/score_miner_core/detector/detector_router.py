from __future__ import annotations

from score_miner_core.detector.base import DetectorBase
from score_miner_core.detector.class_id_mapper import DetectionClassIdMapper
from score_miner_core.detector.deim_runner import DEIMRunner
from score_miner_core.detector.dfine_runner import DFineRunner
from score_miner_core.detector.rfdetr_runner import RFDETRRunner


def create_detector(
    name: str,
    *,
    threshold: float = 0.35,
    player_cls_id: int = 0,
    ball_cls_id: int | None = None,
    optimize_for_inference: bool = True,
) -> DetectorBase | None:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"none", "empty", "stub"}:
        return None
    if normalized in {"rfdetr", "rf_detr", "rfdetr_l", "rfdetr_m"}:
        size = "medium" if normalized in {"rfdetr", "rf_detr", "rfdetr_m"} else "large"
        return RFDETRRunner(
            model_size=size,
            threshold=threshold,
            optimize_for_inference=optimize_for_inference,
            class_id_mapper=DetectionClassIdMapper.coco_to_turbovision(
                player_cls_id=player_cls_id,
                ball_cls_id=ball_cls_id,
            ),
        )
    if normalized in {"deim", "deimv2", "deimv2_l"}:
        return DEIMRunner()
    if normalized in {"dfine", "d_fine", "dfine_l"}:
        return DFineRunner()
    raise ValueError(f"Unknown detector candidate: {name}")
