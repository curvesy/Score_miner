"""Visual sanity rendering for starter labels and predictions."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from public_detect.elements import ElementSpec
from public_detect.score_api import StarterAsset


PALETTE = [
    "#00A676",
    "#F2AF29",
    "#E84855",
    "#3D5A80",
    "#8E44AD",
    "#00A8E8",
]


def render_asset(
    asset: StarterAsset,
    spec: ElementSpec,
    image_path: str | Path,
    output_path: str | Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for annotation in asset.annotations:
        cls_id = spec.class_id(annotation.class_name)
        color = PALETTE[cls_id % len(PALETTE)]
        x1, y1, x2, y2 = annotation.bbox_xyxy
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{cls_id}:{annotation.class_name}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1), label, fill="white", font=font)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)

