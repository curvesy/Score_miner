from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class BenchmarkBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int = Field(ge=0)
    conf: float = Field(ge=0.0, le=1.0)
    team_id: int | str | None = None
    cluster_id: int | str | None = None

    @field_validator("x2")
    @classmethod
    def x2_must_not_be_less_than_zero(cls, value: int) -> int:
        if value < 0:
            raise ValueError("x2 must be non-negative")
        return value

    @field_validator("y2")
    @classmethod
    def y2_must_not_be_less_than_zero(cls, value: int) -> int:
        if value < 0:
            raise ValueError("y2 must be non-negative")
        return value


class BenchmarkFrameResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: int = Field(ge=0)
    boxes: list[BenchmarkBox]
    keypoints: list[tuple[int, int]]


class SchemaCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    frame_count: int
    expected_frame_count: int
    errors: list[str]


def validate_frame_results(
    raw_results: list[Any],
    *,
    expected_frame_count: int,
    n_keypoints: int,
) -> SchemaCheckResult:
    errors: list[str] = []
    validated_frames: list[BenchmarkFrameResult] = []

    for idx, raw in enumerate(raw_results):
        try:
            payload = raw.model_dump() if hasattr(raw, "model_dump") else raw
            frame = BenchmarkFrameResult.model_validate(payload)
            if len(frame.keypoints) != n_keypoints:
                errors.append(
                    f"frame[{idx}] frame_id={frame.frame_id}: expected {n_keypoints} "
                    f"keypoints, got {len(frame.keypoints)}"
                )
            for box_idx, box in enumerate(frame.boxes):
                if box.x2 < box.x1 or box.y2 < box.y1:
                    errors.append(
                        f"frame[{idx}] box[{box_idx}]: invalid xyxy "
                        f"({box.x1}, {box.y1}, {box.x2}, {box.y2})"
                    )
            validated_frames.append(frame)
        except ValidationError as exc:
            errors.append(f"frame[{idx}]: {exc}")

    if len(validated_frames) != expected_frame_count:
        errors.append(
            f"expected {expected_frame_count} frames, got {len(validated_frames)} valid frames"
        )

    return SchemaCheckResult(
        valid=not errors,
        frame_count=len(validated_frames),
        expected_frame_count=expected_frame_count,
        errors=errors,
    )
