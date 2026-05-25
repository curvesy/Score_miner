from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from score_miner_core.benchmark.schema_check import SchemaCheckResult, validate_frame_results


class ChutePredictionBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int = Field(ge=0)
    conf: float = Field(ge=0.0, le=1.0)
    team_id: int | str | None = None
    cluster_id: int | str | None = None


class ChutePredictionFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: int = Field(ge=0)
    boxes: list[ChutePredictionBox]
    keypoints: list[tuple[int, int]]

    @field_validator("boxes")
    @classmethod
    def boxes_must_have_valid_geometry(
        cls, boxes: list[ChutePredictionBox]
    ) -> list[ChutePredictionBox]:
        for idx, box in enumerate(boxes):
            if box.x2 < box.x1 or box.y2 < box.y1:
                raise ValueError(
                    f"box[{idx}] invalid xyxy ({box.x1}, {box.y1}, {box.x2}, {box.y2})"
                )
        return boxes


class ChutePredictions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frames: list[ChutePredictionFrame]


class ChutePredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    predictions: ChutePredictions | None = None
    error: str | None = None


class NumericStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None


class PredictionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    error: str | None
    schema_check: SchemaCheckResult
    frames_returned: int
    frame_id_min: int | None
    frame_id_max: int | None
    frame_ids_contiguous: bool
    duplicate_frame_ids: list[int]
    missing_frame_ids: list[int]
    empty_frames: int
    nonempty_frames: int
    boxes_total: int
    boxes_per_frame: NumericStats
    confidence: NumericStats
    class_counts: dict[str, int]
    team_id_counts: dict[str, int]
    cluster_id_counts: dict[str, int]
    keypoints_per_frame: NumericStats
    valid_keypoints_total: int
    valid_keypoints_per_frame: NumericStats


def summarize_chute_response(
    raw_response: dict[str, Any],
    *,
    n_keypoints: int = 32,
    expected_frame_count: int | None = None,
) -> PredictionSummary:
    try:
        response = ChutePredictResponse.model_validate(raw_response)
        validation_errors: list[str] = []
    except ValidationError as exc:
        response = ChutePredictResponse(success=False, predictions=None, error=str(exc))
        validation_errors = [str(exc)]

    frames = response.predictions.frames if response.predictions else []
    expected = expected_frame_count if expected_frame_count is not None else len(frames)
    schema_check = validate_frame_results(
        [frame.model_dump(mode="json") for frame in frames],
        expected_frame_count=expected,
        n_keypoints=n_keypoints,
    )
    if validation_errors:
        schema_check = SchemaCheckResult(
            valid=False,
            frame_count=schema_check.frame_count,
            expected_frame_count=schema_check.expected_frame_count,
            errors=[*validation_errors, *schema_check.errors],
        )

    frame_ids = [frame.frame_id for frame in frames]
    frame_id_counts = Counter(frame_ids)
    duplicate_frame_ids = sorted(
        frame_id for frame_id, count in frame_id_counts.items() if count > 1
    )
    frame_id_min = min(frame_ids) if frame_ids else None
    frame_id_max = max(frame_ids) if frame_ids else None
    expected_ids = set(range(frame_id_min, frame_id_max + 1)) if frame_ids else set()
    missing_frame_ids = sorted(expected_ids.difference(frame_ids))
    frame_ids_contiguous = bool(frame_ids) and not duplicate_frame_ids and not missing_frame_ids

    box_counts = [len(frame.boxes) for frame in frames]
    confidences = [box.conf for frame in frames for box in frame.boxes]
    class_counts = Counter(str(box.cls_id) for frame in frames for box in frame.boxes)
    team_id_counts = Counter(_nullable_key(box.team_id) for frame in frames for box in frame.boxes)
    cluster_id_counts = Counter(
        _nullable_key(box.cluster_id) for frame in frames for box in frame.boxes
    )
    keypoint_counts = [len(frame.keypoints) for frame in frames]
    valid_keypoint_counts = [
        sum(1 for x, y in frame.keypoints if int(x) != 0 or int(y) != 0) for frame in frames
    ]

    return PredictionSummary(
        success=response.success,
        error=response.error,
        schema_check=schema_check,
        frames_returned=len(frames),
        frame_id_min=frame_id_min,
        frame_id_max=frame_id_max,
        frame_ids_contiguous=frame_ids_contiguous,
        duplicate_frame_ids=duplicate_frame_ids,
        missing_frame_ids=missing_frame_ids,
        empty_frames=sum(1 for count in box_counts if count == 0),
        nonempty_frames=sum(1 for count in box_counts if count > 0),
        boxes_total=sum(box_counts),
        boxes_per_frame=_numeric_stats([float(count) for count in box_counts]),
        confidence=_numeric_stats(confidences),
        class_counts=dict(sorted(class_counts.items())),
        team_id_counts=dict(sorted(team_id_counts.items())),
        cluster_id_counts=dict(sorted(cluster_id_counts.items())),
        keypoints_per_frame=_numeric_stats([float(count) for count in keypoint_counts]),
        valid_keypoints_total=sum(valid_keypoint_counts),
        valid_keypoints_per_frame=_numeric_stats(
            [float(count) for count in valid_keypoint_counts]
        ),
    )


def _numeric_stats(values: list[float]) -> NumericStats:
    if not values:
        return NumericStats(count=0)
    sorted_values = sorted(values)
    return NumericStats(
        count=len(sorted_values),
        min=round(sorted_values[0], 6),
        max=round(sorted_values[-1], 6),
        mean=round(sum(sorted_values) / len(sorted_values), 6),
        p50=round(_percentile(sorted_values, 50), 6),
        p95=round(_percentile(sorted_values, 95), 6),
        p99=round(_percentile(sorted_values, 99), 6),
    )


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _nullable_key(value: int | str | None) -> str:
    return "null" if value is None else str(value)

