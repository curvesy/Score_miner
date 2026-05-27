"""YOLO-format dataset conversion for Score public Detect starter packs."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from public_detect.elements import ElementSpec
from public_detect.score_api import StarterAsset


def clamp_xyxy(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid/clamped bbox {bbox} for image {width}x{height}")
    return x1, y1, x2, y2


def xyxy_to_yolo(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = clamp_xyxy(bbox, width=width, height=height)
    box_w = x2 - x1
    box_h = y2 - y1
    x_center = x1 + box_w / 2.0
    y_center = y1 + box_h / 2.0
    return (
        x_center / width,
        y_center / height,
        box_w / width,
        box_h / height,
    )


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def yolo_label_lines(asset: StarterAsset, spec: ElementSpec, image_path: str | Path) -> list[str]:
    width, height = image_size(image_path)
    if asset.objects:
        spec.assert_objects_match(asset.objects)
    lines = []
    for annotation in asset.annotations:
        cls_id = spec.class_id(annotation.class_name)
        x_center, y_center, box_w, box_h = xyxy_to_yolo(
            annotation.bbox_xyxy,
            width=width,
            height=height,
        )
        lines.append(
            f"{cls_id} {x_center:.8f} {y_center:.8f} {box_w:.8f} {box_h:.8f}"
        )
    return lines


def write_yolo_dataset(
    assets: list[StarterAsset],
    spec: ElementSpec,
    image_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    source_images = Path(image_dir)
    target = Path(output_dir)
    images_out = target / "images" / "train"
    labels_out = target / "labels" / "train"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for asset in assets:
        image_path = _find_image(source_images, asset.asset_id)
        suffix = image_path.suffix or ".png"
        image_out = images_out / f"{asset.asset_id}{suffix}"
        label_out = labels_out / f"{asset.asset_id}.txt"
        shutil.copy2(image_path, image_out)
        lines = yolo_label_lines(asset, spec, image_path)
        label_out.write_text("\n".join(lines) + ("\n" if lines else ""))
        rows.append(
            {
                "asset": asdict(asset),
                "image": str(image_out),
                "label": str(label_out),
                "box_count": len(lines),
            }
        )

    data_yaml = target / "data.yaml"
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(spec.objects))
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {target.resolve()}",
                "train: images/train",
                "val: images/train",
                f"nc: {len(spec.objects)}",
                "names:",
                names_block,
                "",
            ]
        )
    )
    return {"element_id": spec.element_id, "images": len(rows), "rows": rows}


def _find_image(image_dir: Path, asset_id: str) -> Path:
    matches = sorted(image_dir.glob(f"{asset_id}.*"))
    if not matches:
        raise FileNotFoundError(f"no image found for asset {asset_id} in {image_dir}")
    return matches[0]
