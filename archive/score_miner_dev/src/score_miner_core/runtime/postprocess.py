from __future__ import annotations

from os import getenv
from typing import Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class PostprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_boxes_per_frame: int | None = Field(default=18, ge=1)
    min_box_area: float = Field(default=0.0, ge=0.0)

    @classmethod
    def from_env(cls) -> "PostprocessConfig":
        max_boxes_raw = getenv("SCORE_MINER_MAX_BOXES_PER_FRAME", "18").strip()
        max_boxes = None if max_boxes_raw.lower() in {"", "none", "0"} else int(max_boxes_raw)
        return cls(
            confidence_threshold=float(getenv("SCORE_MINER_THRESHOLD", "0.75")),
            max_boxes_per_frame=max_boxes,
            min_box_area=float(getenv("SCORE_MINER_MIN_BOX_AREA", "0")),
        )


class PostprocessBoxLike(Protocol):
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float


BoxT = TypeVar("BoxT", bound=PostprocessBoxLike)


def filter_boxes_by_config(
    boxes: list[BoxT],
    config: PostprocessConfig,
) -> list[BoxT]:
    filtered = [
        box
        for box in boxes
        if box.conf >= config.confidence_threshold and _box_area(box) >= config.min_box_area
    ]
    filtered.sort(key=lambda box: box.conf, reverse=True)
    if config.max_boxes_per_frame is not None:
        filtered = filtered[: config.max_boxes_per_frame]
    return filtered


def _box_area(box: PostprocessBoxLike) -> float:
    return float(max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1))
