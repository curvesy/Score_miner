"""Phase 4 data ingestion utilities."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from public_detect.elements import ElementSpec, load_element_spec
from public_detect.yolo_dataset import xyxy_to_yolo


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class CocoIngestConfig:
    source_id: str
    source_type: str
    element_config: Path
    class_map: dict[str, tuple[str, ...]]
    hard_negative_categories: tuple[str, ...]
    review_status: str
    license_note: str | None
    max_images: int | None
    include_images_with_mapped_labels: bool
    include_hard_negative_only_images: bool


def load_coco_ingest_config(path: str | Path, project_root: str | Path) -> CocoIngestConfig:
    config_path = Path(path)
    project = Path(project_root)
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"invalid ingest config: {config_path}")
    filters = data.get("filters") or {}
    class_map = {
        str(target): tuple(_normalize_label(item) for item in values)
        for target, values in (data.get("class_map") or {}).items()
    }
    return CocoIngestConfig(
        source_id=str(data["source_id"]),
        source_type=str(data["source_type"]),
        element_config=_project_path(project, data["element_config"]),
        class_map=class_map,
        hard_negative_categories=tuple(
            _normalize_label(item) for item in data.get("hard_negative_categories") or []
        ),
        review_status=str(data.get("review_status") or "needs_review"),
        license_note=data.get("license_note"),
        max_images=filters.get("max_images"),
        include_images_with_mapped_labels=bool(filters.get("include_images_with_mapped_labels", True)),
        include_hard_negative_only_images=bool(filters.get("include_hard_negative_only_images", False)),
    )


def ingest_coco_detection(
    *,
    coco_json: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    config: CocoIngestConfig,
) -> dict[str, Any]:
    spec = load_element_spec(config.element_config)
    data = json.loads(Path(coco_json).read_text())
    categories = {
        int(item["id"]): _normalize_label(item["name"])
        for item in data.get("categories", [])
    }
    target_by_source = _target_by_source_label(config.class_map, spec)
    hard_negative_categories = set(config.hard_negative_categories)

    images = {int(item["id"]): item for item in data.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    hard_negative_hits: dict[int, list[str]] = {}
    for ann in data.get("annotations", []):
        image_id = int(ann["image_id"])
        category_name = categories.get(int(ann["category_id"]), "")
        if category_name in target_by_source:
            annotations_by_image.setdefault(image_id, []).append(ann)
        elif category_name in hard_negative_categories:
            hard_negative_hits.setdefault(image_id, []).append(category_name)

    out = Path(output_dir)
    images_out = out / "images" / "train"
    labels_out = out / "labels" / "train"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_id, image_row in sorted(images.items()):
        mapped = annotations_by_image.get(image_id, [])
        hard_negative_labels = sorted(set(hard_negative_hits.get(image_id, [])))
        include = bool(mapped and config.include_images_with_mapped_labels)
        include = include or bool(
            hard_negative_labels and config.include_hard_negative_only_images
        )
        if not include:
            continue
        if config.max_images is not None and len(rows) >= int(config.max_images):
            break

        source_image = _resolve_coco_image_path(Path(image_root), str(image_row["file_name"]))
        suffix = source_image.suffix.lower() if source_image.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
        stem = f"{config.source_id}_{image_id}"
        target_image = images_out / f"{stem}{suffix}"
        target_label = labels_out / f"{stem}.txt"
        shutil.copy2(source_image, target_image)

        width = int(image_row.get("width") or 0)
        height = int(image_row.get("height") or 0)
        if width <= 0 or height <= 0:
            from PIL import Image

            with Image.open(target_image) as image:
                width, height = image.size

        label_lines = []
        mapped_labels = []
        for ann in mapped:
            category_name = categories[int(ann["category_id"])]
            target_name = target_by_source[category_name]
            cls_id = spec.class_id(target_name)
            x, y, w, h = (float(value) for value in ann["bbox"])
            xc, yc, bw, bh = xyxy_to_yolo((x, y, x + w, y + h), width=width, height=height)
            label_lines.append(f"{cls_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
            mapped_labels.append(target_name)
        target_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
        rows.append(
            {
                "source_id": config.source_id,
                "source_type": config.source_type,
                "source_image": str(source_image),
                "image": str(target_image),
                "label": str(target_label),
                "mapped_labels": sorted(set(mapped_labels)),
                "hard_negative_labels": hard_negative_labels,
                "review_status": config.review_status,
                "license_note": config.license_note,
            }
        )

    _write_data_yaml(out, spec)
    manifest = {
        "source_id": config.source_id,
        "source_type": config.source_type,
        "element_id": spec.element_id,
        "images": len(rows),
        "boxes": sum(len(Path(row["label"]).read_text().splitlines()) for row in rows),
        "rows": rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def extract_video_frames(
    *,
    video: str | Path,
    output_dir: str | Path,
    fps: float,
    max_frames: int | None = None,
    prefix: str | None = None,
) -> dict[str, Any]:
    video_path = Path(video)
    if not video_path.exists():
        raise FileNotFoundError(f"video does not exist: {video_path}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    name = prefix or video_path.stem
    pattern = output / f"{name}_%06d.jpg"
    vf = f"fps={fps}"
    if max_frames is not None:
        vf = f"{vf},select='lte(n\\,{int(max_frames) - 1})'"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-q:v",
        "2",
        str(pattern),
    ]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")
    frames = sorted(output.glob(f"{name}_*.jpg"))
    manifest = {
        "source_video": str(video_path),
        "output_dir": str(output),
        "fps": fps,
        "max_frames": max_frames,
        "frames": [str(path) for path in frames],
        "frame_count": len(frames),
        "review_status": "needs_teacher_or_manual_labels",
    }
    (output / f"{name}_frames_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _write_data_yaml(output_dir: Path, spec: ElementSpec) -> None:
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(spec.objects))
    (output_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output_dir}",
                "train: images/train",
                "val: images/train",
                f"nc: {len(spec.objects)}",
                "names:",
                names_block,
                "",
            ]
        )
    )


def _target_by_source_label(
    class_map: dict[str, tuple[str, ...]],
    spec: ElementSpec,
) -> dict[str, str]:
    target_by_source = {}
    for target_name, source_labels in class_map.items():
        spec.class_id(target_name)
        for label in source_labels:
            target_by_source[_normalize_label(label)] = target_name
    return target_by_source


def _resolve_coco_image_path(image_root: Path, file_name: str) -> Path:
    path = Path(file_name)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    candidates.append(image_root / path)
    candidates.append(image_root / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find COCO image {file_name} under {image_root}")


def _normalize_label(value: object) -> str:
    return str(value).strip().lower().replace("_", " ")


def _project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path
