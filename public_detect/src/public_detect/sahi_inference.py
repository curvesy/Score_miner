"""SAHI-backed tiled inference for Score-style threshold sweeps."""

from __future__ import annotations

from pathlib import Path

from public_detect.score_eval import Box


def predict_sahi_ultralytics(
    *,
    model_path: str | Path,
    image_paths: list[Path],
    confidence: float,
    device: str | None,
    slice_height: int,
    slice_width: int,
    overlap: float,
    postprocess_iou: float,
) -> list[Box]:
    """Run SAHI sliced prediction with an Ultralytics checkpoint.

    SAHI is an optional dependency. Install with:
    `uv sync --group inference`
    """
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError as exc:
        raise RuntimeError(
            "SAHI is not installed. Run `uv sync --group inference` before using "
            "`--prediction-mode sahi`."
        ) from exc

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=confidence,
        device=device,
    )
    boxes: list[Box] = []
    for image_path in image_paths:
        result = get_sliced_prediction(
            str(image_path),
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_type="NMS",
            postprocess_match_threshold=postprocess_iou,
            verbose=0,
        )
        for prediction in result.object_prediction_list:
            bbox = prediction.bbox
            score = prediction.score
            category = prediction.category
            boxes.append(
                Box(
                    image_id=image_path.stem,
                    cls=int(category.id),
                    xyxy=(
                        float(bbox.minx),
                        float(bbox.miny),
                        float(bbox.maxx),
                        float(bbox.maxy),
                    ),
                    conf=float(score.value),
                )
            )
    return boxes
