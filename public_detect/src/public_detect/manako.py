"""Download and review Score manako challenge frames.

The JSON predictions from the public manako endpoint are treated as untrusted
pseudo-labels. They are useful for finding Score-distribution frames and for
visual review, but they should not be promoted to ground truth without review.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from public_detect.score_api import USER_AGENT, download_bytes, fetch_json


MANAKO_INDEX_URL = "https://turbo.scoredata.me/manako/index.json"
PALETTE = ["#00A676", "#F2AF29", "#E84855", "#3D5A80", "#8E44AD", "#00A8E8"]


@dataclass(frozen=True)
class ManakoFrame:
    challenge_id: str
    frame_id: int
    url: str
    image_name: str
    boxes: tuple[dict[str, Any], ...]


def fetch_manako_index(url: str = MANAKO_INDEX_URL) -> Any:
    return fetch_json(url)


def parse_manako_frames(index_data: Any) -> list[ManakoFrame]:
    """Extract frames and attached predictions from common manako JSON shapes."""
    entries = _candidate_entries(index_data)
    frames: list[ManakoFrame] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        challenge_id = _challenge_id(entry, idx)
        predicted_by_frame = _prediction_boxes_by_frame(entry)
        for raw_frame in entry.get("frames") or []:
            if not isinstance(raw_frame, dict) or not raw_frame.get("url"):
                continue
            frame_id = int(raw_frame.get("frame_id") or raw_frame.get("frameId") or 0)
            url = str(raw_frame["url"])
            frames.append(
                ManakoFrame(
                    challenge_id=challenge_id,
                    frame_id=frame_id,
                    url=url,
                    image_name=_image_name(challenge_id, frame_id, url),
                    boxes=tuple(predicted_by_frame.get(frame_id, ())),
                )
            )
    return _dedupe_frames(frames)


def download_manako_frames(
    *,
    output_dir: str | Path,
    index_url: str = MANAKO_INDEX_URL,
    limit: int | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    images_dir = output / "images"
    overlays_dir = output / "overlays"
    images_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    index_data = fetch_manako_index(index_url)
    frames = parse_manako_frames(index_data)
    if limit is not None:
        frames = frames[:limit]

    rows = []
    for frame in frames:
        image_path = images_dir / frame.image_name
        if not image_path.exists():
            image_path.write_bytes(download_bytes(frame.url))
        overlay_path = overlays_dir / f"{image_path.stem}.jpg"
        render_prediction_overlay(
            image_path=image_path,
            output_path=overlay_path,
            boxes=frame.boxes,
        )
        rows.append(
            {
                **asdict(frame),
                "image": str(image_path),
                "overlay": str(overlay_path),
                "review_status": "needs_manual_review",
                "label_warning": "prediction boxes are untrusted pseudo-labels, not ground truth",
            }
        )

    manifest = {
        "index_url": index_url,
        "frames": len(rows),
        "rows": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def render_prediction_overlay(
    *,
    image_path: str | Path,
    output_path: str | Path,
    boxes: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box in boxes:
        cls_id = int(box.get("cls_id", box.get("cls", 0)))
        conf = float(box.get("conf", 0.0))
        color = PALETTE[cls_id % len(PALETTE)]
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"pred {cls_id} {conf:.2f}"
        text_bbox = draw.textbbox((x1, max(0, y1 - 12)), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((text_bbox[0], text_bbox[1]), label, fill="white", font=font)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, quality=95)


def _candidate_entries(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        if isinstance(data.get("frames"), list):
            return [data]
        for key in ("items", "challenges", "results", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        entries = [item for item in data.values() if isinstance(item, dict) and "frames" in item]
        if entries:
            return entries
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _challenge_id(entry: dict[str, Any], idx: int) -> str:
    for key in ("challenge_id", "challengeId", "id", "uid"):
        if entry.get(key):
            return str(entry[key])
    for frame in entry.get("frames") or []:
        if isinstance(frame, dict) and frame.get("url"):
            parts = str(frame["url"]).split("/")
            if "challenge-objects" in parts:
                pos = parts.index("challenge-objects")
                if pos + 1 < len(parts):
                    return parts[pos + 1]
    return f"challenge_{idx:06d}"


def _prediction_boxes_by_frame(entry: dict[str, Any]) -> dict[int, tuple[dict[str, Any], ...]]:
    predictions = entry.get("predictions") or {}
    if not isinstance(predictions, dict):
        return {}
    by_frame: dict[int, tuple[dict[str, Any], ...]] = {}
    for raw_frame in predictions.get("frames") or []:
        if not isinstance(raw_frame, dict):
            continue
        frame_id = int(raw_frame.get("frame_id") or raw_frame.get("frameId") or 0)
        boxes = tuple(box for box in raw_frame.get("boxes") or [] if _is_box(box))
        by_frame[frame_id] = boxes
    return by_frame


def _is_box(value: object) -> bool:
    return isinstance(value, dict) and all(key in value for key in ("x1", "y1", "x2", "y2"))


def _image_name(challenge_id: str, frame_id: int, url: str) -> str:
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".png"
    safe_challenge = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in challenge_id)
    return f"{safe_challenge}_{frame_id:06d}{suffix}"


def _dedupe_frames(frames: list[ManakoFrame]) -> list[ManakoFrame]:
    seen = set()
    deduped = []
    for frame in frames:
        key = (frame.challenge_id, frame.frame_id, frame.url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(frame)
    return deduped
