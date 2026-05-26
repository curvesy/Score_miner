#!/usr/bin/env python3
"""Render YOLO label overlays for dataset review."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#00a676", "#e84855", "#3d5a80", "#f2af29", "#8e44ad", "#00a8e8"]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_dataset(data_yaml: Path) -> tuple[list[Path], dict[int, str], Path]:
    data = yaml.safe_load(data_yaml.read_text())
    root = resolve_dataset_root(data_yaml, data.get("path", data_yaml.parent))
    image_dir = root / str(data.get("val") or data.get("train"))
    images = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    names = data["names"]
    if isinstance(names, dict):
        class_names = {int(k): str(v) for k, v in names.items()}
    else:
        class_names = {idx: str(v) for idx, v in enumerate(names)}
    return images, class_names, root


def resolve_dataset_root(data_yaml: Path, value: object) -> Path:
    root = Path(str(value))
    candidates = []
    if root.is_absolute():
        candidates.append(root)
    else:
        candidates.append((data_yaml.parent / root).resolve())
        candidates.append((PROJECT_ROOT / root).resolve())
        if str(root).startswith("score_miner_project/public_detect/"):
            stripped = Path(str(root).removeprefix("score_miner_project/public_detect/"))
            candidates.append((PROJECT_ROOT / stripped).resolve())
        if root.parts[-2:] == data_yaml.parent.parts[-2:]:
            candidates.append(data_yaml.parent.resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def render_dataset(
    *,
    data_yaml: Path,
    output_dir: Path,
    limit: int,
    seed: int,
) -> dict[str, int]:
    images, class_names, _root = load_dataset(data_yaml)
    random.Random(seed).shuffle(images)
    selected = images[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    empty = 0
    for image_path in selected:
        label_path = label_for_image(image_path)
        if not label_path.exists():
            empty += 1
            continue
        lines = [line for line in label_path.read_text().splitlines() if line.strip()]
        if not lines:
            empty += 1
        image = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for line in lines:
            cls, x1, y1, x2, y2 = parse_yolo_line(line, image.width, image.height)
            color = COLORS[cls % len(COLORS)]
            label = f"{cls}:{class_names.get(cls, cls)}"
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            text_box = draw.textbbox((x1, max(0, y1 - 12)), label, font=font)
            draw.rectangle(text_box, fill=color)
            draw.text((text_box[0], text_box[1]), label, fill="white", font=font)
        image.save(output_dir / f"{image_path.stem}.jpg", quality=95)
        rendered += 1
    return {"selected": len(selected), "rendered": rendered, "empty_labels": empty}


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def parse_yolo_line(line: str, width: int, height: int) -> tuple[int, float, float, float, float]:
    parts = line.split()
    cls = int(float(parts[0]))
    xc, yc, bw, bh = (float(value) for value in parts[1:5])
    box_w = bw * width
    box_h = bh * height
    center_x = xc * width
    center_y = yc * height
    return (
        cls,
        center_x - box_w / 2,
        center_y - box_h / 2,
        center_x + box_w / 2,
        center_y + box_h / 2,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=project_path)
    parser.add_argument("--output-dir", required=True, type=project_path)
    parser.add_argument("--limit", default=80, type=int)
    parser.add_argument("--seed", default=44, type=int)
    args = parser.parse_args()
    print(render_dataset(data_yaml=args.data, output_dir=args.output_dir, limit=args.limit, seed=args.seed))


if __name__ == "__main__":
    main()
