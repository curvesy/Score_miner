from __future__ import annotations

from os import getenv
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from score_miner_core.runtime.team_color import TeamColorConfig, extract_torso_lab_feature


class RoleCleanupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    player_cls_id: int = 0
    referee_cls_id: int | None = None
    referee_min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    referee_min_team_distance: float = Field(default=35.0, ge=0.0)
    referee_margin: float = Field(default=8.0, ge=0.0)
    referee_max_per_frame: int = Field(default=2, ge=0)

    @classmethod
    def from_env(cls) -> "RoleCleanupConfig":
        return cls(
            enabled=_env_bool("SCORE_MINER_ROLE_CLEANUP_ENABLED", default=True),
            player_cls_id=int(getenv("SCORE_MINER_PLAYER_CLS_ID", "0")),
            referee_cls_id=_env_optional_int("SCORE_MINER_REFEREE_CLS_ID"),
            referee_min_confidence=float(getenv("SCORE_MINER_REFEREE_MIN_CONFIDENCE", "0.85")),
            referee_min_team_distance=float(getenv("SCORE_MINER_REFEREE_MIN_TEAM_DISTANCE", "35")),
            referee_margin=float(getenv("SCORE_MINER_REFEREE_MARGIN", "8")),
            referee_max_per_frame=int(getenv("SCORE_MINER_REFEREE_MAX_PER_FRAME", "2")),
        )


class RoleBoxLike(Protocol):
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    conf: float
    team_id: int | str | None


def cleanup_roles_by_color(
    image_rgb: np.ndarray,
    boxes: list[RoleBoxLike],
    *,
    role_config: RoleCleanupConfig,
    team_config: TeamColorConfig,
) -> list[RoleBoxLike]:
    if (
        not role_config.enabled
        or role_config.referee_cls_id is None
        or role_config.referee_max_per_frame == 0
    ):
        return boxes

    features_by_idx = _player_torso_features(image_rgb, boxes, role_config, team_config)
    if len(features_by_idx) < team_config.min_players_per_frame:
        return boxes

    team_centroids = _team_centroids(features_by_idx, boxes)
    if len(team_centroids) < 2:
        return boxes

    candidates: list[tuple[float, int]] = []
    for idx, feature in features_by_idx.items():
        box = boxes[idx]
        if box.conf < role_config.referee_min_confidence:
            continue
        distances = sorted(float(np.linalg.norm(feature - centroid)) for centroid in team_centroids)
        nearest = distances[0]
        second = distances[1] if len(distances) > 1 else nearest
        if nearest >= role_config.referee_min_team_distance and second - nearest <= role_config.referee_margin:
            candidates.append((nearest, idx))

    candidates.sort(reverse=True)
    for _score, idx in candidates[: role_config.referee_max_per_frame]:
        boxes[idx].cls_id = role_config.referee_cls_id
        boxes[idx].team_id = None
    return boxes


def _player_torso_features(
    image_rgb: np.ndarray,
    boxes: list[RoleBoxLike],
    role_config: RoleCleanupConfig,
    team_config: TeamColorConfig,
) -> dict[int, np.ndarray]:
    features: dict[int, np.ndarray] = {}
    for idx, box in enumerate(boxes):
        if box.cls_id != role_config.player_cls_id:
            continue
        feature = extract_torso_lab_feature(image_rgb, box, team_config)
        if feature is not None:
            features[idx] = feature
    return features


def _team_centroids(
    features_by_idx: dict[int, np.ndarray],
    boxes: list[RoleBoxLike],
) -> list[np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for idx, feature in features_by_idx.items():
        team_id = boxes[idx].team_id
        if team_id is None:
            continue
        grouped.setdefault(str(team_id), []).append(feature)
    centroids: list[np.ndarray] = []
    for team_features in grouped.values():
        if team_features:
            centroids.append(np.median(np.asarray(team_features, dtype=np.float32), axis=0))
    return centroids


def _env_optional_int(name: str) -> int | None:
    raw = getenv(name)
    if raw is None or raw.strip().lower() in {"", "none", "null", "-1"}:
        return None
    return int(raw)


def _env_bool(name: str, *, default: bool) -> bool:
    raw = getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
