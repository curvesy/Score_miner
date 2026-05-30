from __future__ import annotations

from os import getenv
from typing import Protocol

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class TeamColorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    player_cls_id: int = 0
    min_players_per_frame: int = Field(default=4, ge=2)
    min_crop_pixels: int = Field(default=24, ge=1)
    torso_top_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    torso_bottom_ratio: float = Field(default=0.65, ge=0.0, le=1.0)
    torso_center_width_ratio: float = Field(default=0.70, gt=0.0, le=1.0)
    exclude_grass: bool = True
    grass_hue_min: int = Field(default=35, ge=0, le=179)
    grass_hue_max: int = Field(default=95, ge=0, le=179)
    grass_saturation_min: int = Field(default=45, ge=0, le=255)
    kmeans_attempts: int = Field(default=3, ge=1)
    kmeans_max_iter: int = Field(default=20, ge=1)
    random_seed: int = 2026
    track_memory_enabled: bool = True
    track_memory_min_votes: int = Field(default=2, ge=1)
    track_memory_max_votes: int = Field(default=20, ge=1)

    @classmethod
    def from_env(cls) -> "TeamColorConfig":
        return cls(
            enabled=_env_bool("SCORE_MINER_TEAM_COLOR_ENABLED", default=True),
            player_cls_id=int(getenv("SCORE_MINER_PLAYER_CLS_ID", "0")),
            min_players_per_frame=int(getenv("SCORE_MINER_TEAM_MIN_PLAYERS", "4")),
            min_crop_pixels=int(getenv("SCORE_MINER_TEAM_MIN_CROP_PIXELS", "24")),
            torso_top_ratio=float(getenv("SCORE_MINER_TEAM_TORSO_TOP_RATIO", "0.15")),
            torso_bottom_ratio=float(getenv("SCORE_MINER_TEAM_TORSO_BOTTOM_RATIO", "0.65")),
            torso_center_width_ratio=float(getenv("SCORE_MINER_TEAM_TORSO_CENTER_WIDTH_RATIO", "0.70")),
            exclude_grass=_env_bool("SCORE_MINER_TEAM_EXCLUDE_GRASS", default=True),
            grass_hue_min=int(getenv("SCORE_MINER_TEAM_GRASS_HUE_MIN", "35")),
            grass_hue_max=int(getenv("SCORE_MINER_TEAM_GRASS_HUE_MAX", "95")),
            grass_saturation_min=int(getenv("SCORE_MINER_TEAM_GRASS_SATURATION_MIN", "45")),
            kmeans_attempts=int(getenv("SCORE_MINER_TEAM_KMEANS_ATTEMPTS", "3")),
            kmeans_max_iter=int(getenv("SCORE_MINER_TEAM_KMEANS_MAX_ITER", "20")),
            random_seed=int(getenv("SCORE_MINER_TEAM_RANDOM_SEED", "2026")),
            track_memory_enabled=_env_bool("SCORE_MINER_TEAM_TRACK_MEMORY_ENABLED", default=True),
            track_memory_min_votes=int(getenv("SCORE_MINER_TEAM_TRACK_MEMORY_MIN_VOTES", "2")),
            track_memory_max_votes=int(getenv("SCORE_MINER_TEAM_TRACK_MEMORY_MAX_VOTES", "20")),
        )


class TeamBoxLike(Protocol):
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    team_id: int | str | None
    track_id: int | None


def assign_team_ids_by_color(
    image_rgb: np.ndarray,
    boxes: list[TeamBoxLike],
    config: TeamColorConfig,
) -> list[TeamBoxLike]:
    if not config.enabled:
        return boxes

    player_indices: list[int] = []
    features: list[np.ndarray] = []
    for idx, box in enumerate(boxes):
        if box.cls_id != config.player_cls_id:
            continue
        feature = extract_torso_lab_feature(image_rgb, box, config)
        if feature is None:
            continue
        player_indices.append(idx)
        features.append(feature)

    if len(features) < config.min_players_per_frame:
        return boxes

    labels = _cluster_two_teams(np.asarray(features, dtype=np.float32), config)
    for player_idx, label in zip(player_indices, labels):
        boxes[player_idx].team_id = int(label) + 1
    return boxes


def extract_torso_lab_feature(
    image_rgb: np.ndarray,
    box: TeamBoxLike,
    config: TeamColorConfig,
) -> np.ndarray | None:
    height, width = image_rgb.shape[:2]
    x1 = int(np.clip(box.x1, 0, width - 1))
    x2 = int(np.clip(box.x2, 0, width))
    y1 = int(np.clip(box.y1, 0, height - 1))
    y2 = int(np.clip(box.y2, 0, height))
    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1
    center_x = (x1 + x2) / 2.0
    half_w = box_w * config.torso_center_width_ratio / 2.0
    crop_x1 = int(np.clip(center_x - half_w, 0, width - 1))
    crop_x2 = int(np.clip(center_x + half_w, 0, width))
    crop_y1 = int(np.clip(y1 + box_h * config.torso_top_ratio, 0, height - 1))
    crop_y2 = int(np.clip(y1 + box_h * config.torso_bottom_ratio, 0, height))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    crop = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    pixels = crop.reshape(-1, 3)
    if config.exclude_grass:
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).reshape(-1, 3)
        hue = hsv[:, 0]
        saturation = hsv[:, 1]
        green = (
            (hue >= config.grass_hue_min)
            & (hue <= config.grass_hue_max)
            & (saturation >= config.grass_saturation_min)
        )
        pixels = pixels[~green]

    if len(pixels) < config.min_crop_pixels:
        return None

    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return np.median(lab.astype(np.float32), axis=0)


def _cluster_two_teams(features: np.ndarray, config: TeamColorConfig) -> np.ndarray:
    cv2.setRNGSeed(config.random_seed)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        config.kmeans_max_iter,
        1.0,
    )
    _compactness, labels, centers = cv2.kmeans(
        features,
        2,
        None,
        criteria,
        config.kmeans_attempts,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(-1)

    # Deterministic TEAM1/TEAM2 naming. The scorer checks both orientations, but
    # stable naming still reduces frame-to-frame palette noise.
    order = np.lexsort((centers[:, 0], centers[:, 2], centers[:, 1]))
    remap = {int(old_label): new_label for new_label, old_label in enumerate(order)}
    return np.asarray([remap[int(label)] for label in labels], dtype=int)


class TeamColorMemory:
    def __init__(self, config: TeamColorConfig) -> None:
        self.config = config
        self._votes: dict[int, list[int]] = {}

    def stabilize(self, boxes: list[TeamBoxLike]) -> list[TeamBoxLike]:
        """Carry stable team assignments through weak frames.

        Team color can be missing when there are too few usable player crops, and
        tracker IDs can be missing when ByteTrack cannot attach a detection. Use
        track vote memory first; for track-less boxes, use the closest same-frame
        assigned player as a conservative fallback.
        """
        if not self.config.enabled or not self.config.track_memory_enabled:
            return boxes

        recent_assignments: list[tuple[float, float, int]] = []
        for box in boxes:
            if box.cls_id != self.config.player_cls_id:
                continue
            if box.team_id in (1, "1", 2, "2"):
                team = 1 if box.team_id in (1, "1") else 2
                recent_assignments.append((*_box_center(box), team))

        for box in boxes:
            if box.cls_id != self.config.player_cls_id:
                continue
            if box.track_id is None:
                if box.team_id not in (1, "1", 2, "2") and recent_assignments:
                    cx, cy = _box_center(box)
                    nearest = min(
                        recent_assignments,
                        key=lambda assignment: (assignment[0] - cx) ** 2 + (assignment[1] - cy) ** 2,
                    )
                    box.team_id = nearest[2]
                continue

            if box.team_id in (1, "1"):
                self._add_vote(box.track_id, team_index=0)
            elif box.team_id in (2, "2"):
                self._add_vote(box.track_id, team_index=1)

            memory_team = self._memory_team(box.track_id)
            if memory_team is not None:
                box.team_id = memory_team
        return boxes

    def _add_vote(self, track_id: int, *, team_index: int) -> None:
        votes = self._votes.setdefault(int(track_id), [0, 0])
        votes[team_index] += 1
        total = votes[0] + votes[1]
        if total > self.config.track_memory_max_votes:
            votes[0] = max(0, votes[0] - 1)
            votes[1] = max(0, votes[1] - 1)

    def _memory_team(self, track_id: int) -> int | None:
        votes = self._votes.get(int(track_id))
        if votes is None or max(votes) < self.config.track_memory_min_votes:
            return None
        return 1 if votes[0] >= votes[1] else 2


def _box_center(box: TeamBoxLike) -> tuple[float, float]:
    return (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0


def _env_bool(name: str, *, default: bool) -> bool:
    raw = getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
