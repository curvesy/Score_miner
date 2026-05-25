from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from score_miner_core.validator_sim.replay_loader import load_json_object


def load_pseudo_ground_truth(
    path: Path,
    *,
    turbovision_path: Path,
) -> list[object]:
    _ensure_turbovision_importable(turbovision_path)

    from scorevision.vlm_pipeline.domain_specific_schemas.football import Action
    from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
    from scorevision.vlm_pipeline.utils.response_models import BoundingBox, FrameAnnotation

    payload = load_json_object(path)
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("PGT JSON must contain annotations: [...]")

    by_frame: dict[int, list[BoundingBox]] = {}
    for idx, item in enumerate(annotations):
        if not isinstance(item, dict):
            raise ValueError(f"annotations[{idx}] must be an object")
        frame_id = item.get("frame_id", item.get("frame_idx", item.get("frame_number")))
        if isinstance(frame_id, str) and frame_id.isdigit():
            frame_id = int(frame_id)
        if not isinstance(frame_id, int):
            raise ValueError(f"annotations[{idx}] missing integer frame_id")

        raw_bbox = item.get("bbox", item.get("bbox_2d"))
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            raise ValueError(f"annotations[{idx}] missing bbox/bbox_2d length 4")
        x1, y1, x2, y2 = [int(v) for v in raw_bbox]
        label = str(item.get("label", item.get("class", "player")))
        score = item.get("score")
        by_frame.setdefault(frame_id, []).append(
            BoundingBox(
                bbox_2d=(x1, y1, x2, y2),
                label=label,
                score=float(score) if score is not None else None,
                cluster_id=None,
            )
        )

    video_name = str(payload.get("video_name", path.stem))
    spatial_stub = np.zeros((1, 1, 3), dtype=np.uint8)
    temporal_stub = np.zeros((1, 1, 3), dtype=np.uint8)
    return [
        PseudoGroundTruth(
            video_name=video_name,
            frame_number=frame_number,
            spatial_image=spatial_stub,
            temporal_image=temporal_stub,
            annotation=FrameAnnotation(
                bboxes=boxes,
                category=Action.NONE,
                confidence=100,
                reason="validator_sim ground truth",
            ),
        )
        for frame_number, boxes in sorted(by_frame.items())
    ]


def _ensure_turbovision_importable(turbovision_path: Path) -> None:
    resolved = turbovision_path.resolve()
    if not (resolved / "scorevision").is_dir():
        raise FileNotFoundError(f"TurboVision path does not contain scorevision/: {resolved}")
    path_str = str(resolved)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

