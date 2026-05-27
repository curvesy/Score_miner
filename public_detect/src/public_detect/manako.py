"""Download and review Score manako challenge frames.

The JSON predictions from the public manako endpoint are treated as untrusted
pseudo-labels. They are useful for finding Score-distribution frames and for
visual review, but they should not be promoted to ground truth without review.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
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
    element_filters: tuple[str, ...] = (),
    max_refs: int | None = None,
    min_score: float | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    images_dir = output / "images"
    overlays_dir = output / "overlays"
    images_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[manako] fetching index: {index_url}", flush=True)
    index_data = fetch_manako_index(index_url)
    frames, skipped_refs = load_manako_frames_from_index(
        index_data=index_data,
        index_url=index_url,
        limit=limit,
        element_filters=element_filters,
        max_refs=max_refs,
        min_score=min_score,
        verbose=verbose,
        return_skipped=True,
    )
    if limit is not None:
        frames = frames[:limit]
    if verbose:
        print(f"[manako] usable frames: {len(frames)} skipped refs: {len(skipped_refs)}", flush=True)

    rows = []
    for idx, frame in enumerate(frames, start=1):
        image_path = images_dir / frame.image_name
        if not image_path.exists():
            if verbose:
                print(f"[manako] downloading {idx}/{len(frames)} {frame.url}", flush=True)
            image_path.write_bytes(download_bytes(frame.url))
        elif verbose:
            print(f"[manako] exists {idx}/{len(frames)} {image_path.name}", flush=True)
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
        "element_filters": element_filters,
        "max_refs": max_refs,
        "min_score": min_score,
        "frames": len(rows),
        "skipped_refs": skipped_refs,
        "rows": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_manako_frames_from_index(
    *,
    index_data: Any,
    index_url: str = MANAKO_INDEX_URL,
    limit: int | None = None,
    element_filters: tuple[str, ...] = (),
    max_refs: int | None = None,
    min_score: float | None = None,
    verbose: bool = False,
    return_skipped: bool = False,
) -> list[ManakoFrame] | tuple[list[ManakoFrame], list[dict[str, str]]]:
    """Parse frames directly or follow challenge-file references from an index."""
    direct_frames = parse_manako_frames(index_data)
    if direct_frames:
        if verbose:
            print(f"[manako] index contains direct frames: {len(direct_frames)}", flush=True)
        if return_skipped:
            return direct_frames, []
        return direct_frames

    frames: list[ManakoFrame] = []
    skipped_refs: list[dict[str, str]] = []
    seen_refs: set[str] = set()
    refs = _collect_challenge_refs(index_data, base_url=index_url)
    if element_filters:
        refs = [ref for ref in refs if _matches_element_filters(ref, element_filters)]
    if max_refs is not None:
        refs = refs[:max_refs]
    if verbose:
        print(f"[manako] index refs found: {len(refs)}", flush=True)
    for idx, ref in enumerate(refs, start=1):
        if ref in seen_refs:
            continue
        seen_refs.add(ref)
        try:
            if verbose:
                print(f"[manako] fetching ref {idx}/{len(refs)} {ref}", flush=True)
            payload = fetch_manako_payload(ref)
        except Exception as exc:
            skipped_refs.append({"url": ref, "error": f"{type(exc).__name__}: {exc}"})
            if verbose:
                print(f"[manako] skipped ref {ref}: {type(exc).__name__}: {exc}", flush=True)
            continue
        parsed = parse_manako_frames(payload)
        if not parsed:
            for response_ref in _collect_response_refs(payload, base_url=ref, min_score=min_score):
                try:
                    if verbose:
                        print(f"[manako] fetching response {response_ref}", flush=True)
                    response_payload = fetch_manako_payload(response_ref)
                except Exception as exc:
                    skipped_refs.append({"url": response_ref, "error": f"{type(exc).__name__}: {exc}"})
                    if verbose:
                        print(
                            f"[manako] skipped response {response_ref}: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                    continue
                parsed.extend(parse_manako_frames(response_payload))
        if verbose:
            print(f"[manako] parsed frames from ref: {len(parsed)}", flush=True)
        frames.extend(parsed)
        frames = _dedupe_frames(frames)
        if limit is not None and len(frames) >= limit:
            frames = frames[:limit]
            if return_skipped:
                return frames, skipped_refs
            return frames
    if return_skipped:
        return frames, skipped_refs
    return frames


def fetch_manako_payload(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as response:
        raw = response.read().decode("utf-8", "replace")
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(raw)
    return json.loads(raw)


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
        nested = []
        for value in data.values():
            nested.extend(_candidate_entries(value))
        if nested:
            return nested
    if isinstance(data, list):
        direct = [item for item in data if isinstance(item, dict) and "frames" in item]
        if direct:
            return direct
        nested = []
        for item in data:
            nested.extend(_candidate_entries(item))
        if nested:
            return nested
    return []


def _collect_challenge_refs(data: Any, base_url: str) -> list[str]:
    refs: list[str] = []
    parsed_base = urllib.parse.urlparse(base_url)
    origin = f"{parsed_base.scheme}://{parsed_base.netloc}/"

    def visit(value: Any) -> None:
        if isinstance(value, str):
            ref = value.strip()
            if _looks_like_challenge_ref(ref):
                refs.append(_resolve_manako_ref(ref, base_url=base_url, origin=origin))
            return
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if isinstance(value, dict):
            for key in ("url", "href", "path", "key", "file", "filename"):
                if key in value:
                    visit(value[key])
            for item in value.values():
                visit(item)

    visit(data)
    return list(reversed(refs))


def _collect_response_refs(data: Any, base_url: str, min_score: float | None = None) -> list[str]:
    refs: list[str] = []
    parsed_base = urllib.parse.urlparse(base_url)
    origin = f"{parsed_base.scheme}://{parsed_base.netloc}/"

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            telemetry = value.get("telemetry")
            if isinstance(telemetry, dict):
                run = telemetry.get("run")
                if (
                    isinstance(run, dict)
                    and isinstance(run.get("responses_key"), str)
                    and _response_is_usable(container=value, run=run, min_score=min_score)
                ):
                    refs.append(_resolve_manako_ref(run["responses_key"], base_url=base_url, origin=origin))
            run = value.get("run")
            if isinstance(run, dict) and isinstance(run.get("responses_key"), str):
                if _response_is_usable(container=value, run=run, min_score=min_score):
                    refs.append(_resolve_manako_ref(run["responses_key"], base_url=base_url, origin=origin))
            if (
                isinstance(value.get("responses_key"), str)
                and _response_is_usable(container=value, run=value, min_score=min_score)
            ):
                refs.append(_resolve_manako_ref(value["responses_key"], base_url=base_url, origin=origin))
            for item in value.values():
                visit(item)
            return
        if isinstance(value, list):
            for item in value:
                visit(item)

    visit(data)
    return list(dict.fromkeys(refs))


def _response_is_usable(
    *,
    container: dict[str, Any],
    run: dict[str, Any],
    min_score: float | None,
) -> bool:
    if run.get("success") is False:
        return False
    if min_score is None:
        return True
    score = _score_from_payload(container)
    return score is not None and score >= min_score


def _score_from_payload(value: dict[str, Any]) -> float | None:
    candidates = [
        value.get("composite_score"),
        (value.get("metrics") or {}).get("composite_score") if isinstance(value.get("metrics"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, int | float):
            return float(candidate)
    return None


def _looks_like_challenge_ref(value: str) -> bool:
    path = urllib.parse.urlparse(value).path.lower()
    return path.endswith((".json", ".yaml", ".yml")) and not path.endswith("/index.json")


def _matches_element_filters(ref: str, element_filters: tuple[str, ...]) -> bool:
    normalized_ref = ref.lower().replace("_", "-").replace("%2f", "/")
    return any(
        item.lower().replace("_", "-").replace("%2f", "/") in normalized_ref
        for item in element_filters
    )


def _resolve_manako_ref(ref: str, *, base_url: str, origin: str) -> str:
    if urllib.parse.urlparse(ref).scheme:
        return ref
    normalized = ref.lstrip("/")
    if normalized.startswith("manako/"):
        return urllib.parse.urljoin(origin, normalized)
    return urllib.parse.urljoin(base_url, ref)


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
        key = (frame.challenge_id, frame.frame_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(frame)
    return deduped
