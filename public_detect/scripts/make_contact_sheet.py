#!/usr/bin/env python3
"""Create contact sheets for quick source-data review."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def make_sheet(
    image_paths: list[Path],
    output_path: Path,
    thumb_width: int,
    columns: int,
) -> None:
    font = ImageFont.load_default()
    thumbs = []
    for path in image_paths:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image.convert("RGB"))
            ratio = thumb_width / image.width
            thumb_height = max(1, int(image.height * ratio))
            thumb = image.resize((thumb_width, thumb_height))
        label_height = 20
        tile = Image.new("RGB", (thumb_width, thumb_height + label_height), "white")
        tile.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((3, thumb_height + 3), path.name[:42], fill="black", font=font)
        thumbs.append(tile)

    if not thumbs:
        raise ValueError("no images to render")
    tile_height = max(item.height for item in thumbs)
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * tile_height), "white")
    for idx, tile in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        sheet.paste(tile, (col * thumb_width, row * tile_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", default=120, type=int)
    parser.add_argument("--thumb-width", default=240, type=int)
    parser.add_argument("--columns", default=5, type=int)
    args = parser.parse_args()

    paths = sorted(
        item for item in args.input_dir.iterdir()
        if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )[: args.limit]
    make_sheet(paths, args.output, thumb_width=args.thumb_width, columns=args.columns)
    print(args.output)


if __name__ == "__main__":
    main()
