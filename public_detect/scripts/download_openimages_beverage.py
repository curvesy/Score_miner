#!/usr/bin/env python3
"""Download Open Images V7 cup/bottle/can subset via FiftyOne.

OIV7 boxable class names (exact, case-sensitive) we use:
  Mug         -> local cls 0 (cup)
  Coffee cup  -> local cls 0 (cup)
  Bottle      -> local cls 1 (bottle)
  Tin can     -> local cls 2 (can)

(OIV7 has no class called "Cup" — that error message means class is invalid.)

FiftyOne requires a local MongoDB. On Ubuntu install the bundled binary:
  UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \\
      uv pip install fiftyone-db-ubuntu2204

Writes YOLO format under <output-dir>:
  images/train/*.jpg
  labels/train/*.txt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


OIV7_TO_LOCAL = {
    "Mug": 0,
    "Coffee cup": 0,
    "Bottle": 1,
    "Tin can": 2,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
        help="OIV7 split. 'validation' is fastest (~41k base images).",
    )
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/fiftyone_oiv7_export"),
        help="staging dir for FiftyOne raw YOLO export",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="do not delete the staging YOLO export after remap",
    )
    args = parser.parse_args()

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        sys.exit(
            "fiftyone is not installed. Run:\n"
            "  UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \\\n"
            "    uv pip install fiftyone"
        )

    classes = list(OIV7_TO_LOCAL.keys())
    print(f"[oiv7] split={args.split} classes={classes} max={args.max_samples}")

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=args.split,
        label_types=["detections"],
        classes=classes,
        max_samples=args.max_samples,
    )
    print(f"[oiv7] loaded {len(dataset)} samples")

    if args.workdir.exists():
        shutil.rmtree(args.workdir)
    args.workdir.mkdir(parents=True, exist_ok=True)

    dataset.export(
        export_dir=str(args.workdir),
        dataset_type=fo.types.YOLOv5Dataset,
        classes=classes,
        label_field="ground_truth",
        split=args.split,
    )
    print(f"[oiv7] raw YOLO export -> {args.workdir}")

    src_images = args.workdir / "images" / args.split
    src_labels = args.workdir / "labels" / args.split
    if not src_labels.exists():
        sys.exit(f"FiftyOne did not produce labels at {src_labels}")

    out_images = args.output_dir / "images" / "train"
    out_labels = args.output_dir / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    oiv7_id_to_local = {idx: OIV7_TO_LOCAL[name] for idx, name in enumerate(classes)}
    print(f"[oiv7] id map (export -> local): {oiv7_id_to_local}")

    class_counts = {0: 0, 1: 0, 2: 0}
    kept_images = 0
    for label_path in sorted(src_labels.iterdir()):
        if label_path.suffix.lower() != ".txt":
            continue
        rows = []
        for line in label_path.read_text().splitlines():
            bits = line.strip().split()
            if len(bits) != 5:
                continue
            try:
                src_cls = int(bits[0])
            except ValueError:
                continue
            if src_cls not in oiv7_id_to_local:
                continue
            local_cls = oiv7_id_to_local[src_cls]
            rows.append(f"{local_cls} {' '.join(bits[1:])}")
            class_counts[local_cls] += 1

        if not rows:
            continue

        matches = sorted(src_images.glob(f"{label_path.stem}.*"))
        if not matches:
            continue
        src_img = matches[0]
        new_img = out_images / src_img.name
        new_lbl = out_labels / f"{label_path.stem}.txt"
        if not new_img.exists():
            shutil.copy2(src_img, new_img)
        new_lbl.write_text("\n".join(rows) + "\n")
        kept_images += 1

    print(f"[oiv7] kept {kept_images} image+label pairs")
    print(f"[oiv7] class counts (local ids): {class_counts}")
    print(f"[oiv7] output: {args.output_dir}")

    if not args.keep_workdir:
        shutil.rmtree(args.workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
