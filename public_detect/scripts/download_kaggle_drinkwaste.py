#!/usr/bin/env python3
"""Download arkadiyhacks/drinking-waste-classification and remap to local classes.

Real on-disk layout (after unzip):
  Images_of_Waste/
    YOLO_imgs/                      <- labeled, flat, filename-prefix per class
      AluCan001.jpg + AluCan001.txt
      Glass042.jpg  + Glass042.txt
      HDPEM5.jpg    + HDPEM5.txt
      PET730.jpg    + PET730.txt
    rawimgs/<AluCan|Glass|HDPEM|PET>/   <- unlabeled raw duplicates (we skip)

Filename-prefix mapping:
  AluCan -> 2 (can)
  Glass  -> 1 (bottle)
  HDPEM  -> 1 (bottle)
  PET    -> 1 (bottle)

Prereq:
  pip install --user kaggle    (or: uv pip install kaggle)
  ~/.kaggle/kaggle.json with API key, chmod 600

Example:
  PYTHONPATH=src uv run python scripts/download_kaggle_drinkwaste.py \
      --output-dir data/yolo_candidates/beverage_drinkwaste_v1
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SLUG = "arkadiyhacks/drinking-waste-classification"

# Prefix matched against the filename stem (case-insensitive). Order matters:
# longer prefixes first so HDPEM is checked before HDPE etc.
PREFIX_TO_LOCAL: list[tuple[str, int]] = [
    ("alucan", 2),   # can
    ("hdpem", 1),    # bottle (HDPE milk bottle)
    ("glass", 1),    # bottle (glass)
    ("pet",   1),    # bottle (plastic)
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def classify_filename(stem: str) -> int | None:
    key = stem.lower()
    for prefix, cls in PREFIX_TO_LOCAL:
        if key.startswith(prefix):
            return cls
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/external/drinking_waste"),
        help="where to unzip the Kaggle archive",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="reuse existing download-dir contents instead of fetching from Kaggle",
    )
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    if not args.skip_download:
        if shutil.which("kaggle") is None:
            sys.exit(
                "kaggle CLI not found. Install with:\n"
                "  uv pip install kaggle\n"
                "Then put your API key at ~/.kaggle/kaggle.json (chmod 600)."
            )
        args.download_dir.mkdir(parents=True, exist_ok=True)
        print(f"[drinkwaste] downloading {SLUG} -> {args.download_dir}")
        cmd = ["kaggle", "datasets", "download", "-d", SLUG, "-p", str(args.download_dir), "--unzip"]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            sys.exit(f"kaggle CLI failed (exit {result.returncode})")
    else:
        print(f"[drinkwaste] skip-download, reusing {args.download_dir}")

    out_images = args.output_dir / "images" / "train"
    out_labels = args.output_dir / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # The labeled images live in Images_of_Waste/YOLO_imgs/ as a flat dir.
    # rawimgs/<class>/ are unlabeled duplicates we ignore.
    yolo_dirs = list(args.download_dir.rglob("YOLO_imgs"))
    if not yolo_dirs:
        sys.exit(f"could not find a YOLO_imgs folder under {args.download_dir}")
    print(f"[drinkwaste] using {len(yolo_dirs)} labeled dir(s): {[str(p) for p in yolo_dirs]}")

    class_counts = {0: 0, 1: 0, 2: 0}
    kept = 0
    skipped_no_label = 0
    skipped_no_prefix_match = 0

    for yolo_dir in yolo_dirs:
        for img_path in sorted(yolo_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            local_cls = classify_filename(img_path.stem)
            if local_cls is None:
                skipped_no_prefix_match += 1
                continue

            label_path = img_path.with_suffix(".txt")
            if not label_path.exists():
                skipped_no_label += 1
                continue

            rows = []
            for line in label_path.read_text().splitlines():
                bits = line.strip().split()
                if len(bits) != 5:
                    continue
                rows.append(f"{local_cls} {' '.join(bits[1:])}")
                class_counts[local_cls] += 1
            if not rows:
                continue

            # commas in original names ("PET1,107.jpg") break shell tooling later
            safe_stem = img_path.stem.replace(",", "_")
            stem = f"drinkwaste_{kept:06d}_{safe_stem}"
            shutil.copy2(img_path, out_images / f"{stem}{img_path.suffix.lower()}")
            (out_labels / f"{stem}.txt").write_text("\n".join(rows) + "\n")
            kept += 1
            if args.max_images is not None and kept >= args.max_images:
                break
        if args.max_images is not None and kept >= args.max_images:
            break

    print(f"[drinkwaste] kept {kept} image+label pairs")
    print(f"[drinkwaste] class counts (local ids): {class_counts}")
    print(f"[drinkwaste] skipped (no filename prefix match): {skipped_no_prefix_match}")
    print(f"[drinkwaste] skipped (no label .txt): {skipped_no_label}")
    print(f"[drinkwaste] output: {args.output_dir}")


if __name__ == "__main__":
    main()
