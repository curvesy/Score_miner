"""ScoreVision Console V2 helpers for public Detect starter packs."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONSOLE_BASE = "https://console.scorevision.io/api/v2"
USER_AGENT = "Mozilla/5.0 score-public-detect-phase0/0.1"


@dataclass(frozen=True)
class StarterAnnotation:
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class StarterAsset:
    asset_id: str
    image_url: str
    annotation_url: str | None
    frame_index: int | None
    objects: tuple[str, ...]
    annotations: tuple[StarterAnnotation, ...]


def fetch_json(url: str, retries: int = 8, timeout: int = 30) -> Any:
    last_error: str | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            last_error = f"HTTP {exc.code}: {body}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(30, attempt * 4))
    raise RuntimeError(f"failed to fetch JSON {url}: {last_error}")


def download_bytes(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    last_error: str | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(15, attempt * 2))
    raise RuntimeError(f"failed to download {url}: {last_error}")


def element_detail_url(element_id: str, lookback_days: int = 4) -> str:
    encoded = urllib.parse.quote(element_id, safe="")
    return f"{CONSOLE_BASE}/elements/{encoded}?lookback_days={lookback_days}"


def fetch_element_detail(element_id: str, lookback_days: int = 4) -> dict[str, Any]:
    data = fetch_json(element_detail_url(element_id, lookback_days=lookback_days))
    if not isinstance(data, dict):
        raise ValueError(f"element detail for {element_id} is not an object")
    return data


def parse_starter_assets(detail: dict[str, Any]) -> list[StarterAsset]:
    starter_pack = detail.get("starterPack")
    if not isinstance(starter_pack, dict):
        raise ValueError("element detail has no starterPack object")
    raw_assets = starter_pack.get("starterAssets")
    if not isinstance(raw_assets, list):
        raise ValueError("starterPack has no starterAssets list")

    assets: list[StarterAsset] = []
    for raw in raw_assets:
        if not isinstance(raw, dict):
            continue
        annotations = []
        for item in raw.get("annotations") or []:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            annotations.append(
                StarterAnnotation(
                    class_name=str(item.get("className")),
                    bbox_xyxy=tuple(float(v) for v in bbox),
                )
            )
        objects = raw.get("objects") or []
        assets.append(
            StarterAsset(
                asset_id=str(raw.get("assetId")),
                image_url=str(raw.get("imageUrl")),
                annotation_url=raw.get("annotationUrl"),
                frame_index=raw.get("frameIndex"),
                objects=tuple(str(item) for item in objects),
                annotations=tuple(annotations),
            )
        )
    return assets


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
