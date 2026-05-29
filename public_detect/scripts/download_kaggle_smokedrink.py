#!/usr/bin/env python3
"""Download prajjwalkumarpanzade/smoking-and-drinking-dataset-for-yolo and remap.

Dataset has two classes (canonical order):
  smoking
  drinking

The 'drinking' boxes are "person + drink in hand" scenes — the Score-like
distribution we want. We drop all 'smoking' boxes and treat 'drinking' as
class 0 (cup). If you'd rather merge it as bottle (1), pass --drinking-as bottle.

The dataset is shipped with a data.yaml indicating class order. We read it
when present; otherwise we default to drinking_idx=1 (the more common layout
in mirrors of this dataset) which matches "names: ['smoking','drinking']".

Prereq:
  uv pip install kaggle  +  ~/.kaggle/kaggle.json (chmod 600)

Example:
  PYTHONPATH=src uv run python scripts/download_kaggle_smokedrink.py \
      --output-dir data/yolo_candidates/beverage_smokedrink_v1 \
      --drinking-as cup
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SLUG = "prajjwalkumarpanzade/smoking-and-drinking-dataset-for-yolo"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

CLASS_NAME_TO_LOCAL = {"cup": 0, "bottle": 1, "can": 2}


def parse_data_yaml(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    try:
        import yaml
    except ImportError:
        return None
    try:
        data = yaml.safe_load(path.read_text())
    except Exception:
        return None
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, list):
        return [str(n).lower().strip() for n in names]
    if isinstance(names, dict):
        return [str(names[k]).lower().strip() for k in sorted(names)]
    return None


def find_drinking_index(download_dir: Path) -> int:
    for candidate in download_dir.rglob("data.yaml"):
        names = parse_data_yaml(candidate)
        if names and "drinking" in names:
            idx = names.index("drinking")
            print(f"[smokedrink] data.yaml -> drinking class id = {idx} ({candidate})")
            return idx
    print("[smokedrink] no data.yaml found; defaulting drinking_idx=1 (smoking,drinking)")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/external/smoking_drinking"),
    )
    parser.add_argument(
        "--drinking-as",
        choices=list(CLASS_NAME_TO_LOCAL),
        default="cup",
        help="which local class to assign drinking boxes to",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    if not args.skip_download:
        if shutil.which("kaggle") is None:
            sys.exit("kaggle CLI not found. uv pip install kaggle and place ~/.kaggle/kaggle.json.")
        args.download_dir.mkdir(parents=True, exist_ok=True)
        print(f"[smokedrink] downloading {SLUG} -> {args.download_dir}")
        cmd = ["kaggle", "datasets", "download", "-d", SLUG, "-p", str(args.download_dir), "--unzip"]
        if subprocess.run(cmd).returncode != 0:
            sys.exit("kaggle CLI failed")
    else:
        print(f"[smokedrink] skip-download, reusing {args.download_dir}")

    target_local = CLASS_NAME_TO_LOCAL[args.drinking_as]
    drinking_idx = find_drinking_index(args.download_dir)

    out_images = args.output_dir / "images" / "train"
    out_labels = args.output_dir / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_no_drinking = 0
    skipped_no_pair = 0
    for img_path in sorted(args.download_dir.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        # standard YOLO layout: same stem .txt next to image, OR ../labels/<stem>.txt
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            alt = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if alt.exists():
                label_path = alt
            else:
                # try sibling 'labels' folder at any ancestor
                for ancestor in img_path.parents:
                    cand = ancestor / "labels" / f"{img_path.stem}.txt"
                    if cand.exists():
                        label_path = cand
                        break
        if not label_path.exists():
            skipped_no_pair += 1
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
            if src_cls != drinking_idx:
                continue
            rows.append(f"{target_local} {' '.join(bits[1:])}")

        if not rows:
            skipped_no_drinking += 1
            continue

        stem = f"smokedrink_{kept:06d}_{img_path.stem}"
        shutil.copy2(img_path, out_images / f"{stem}{img_path.suffix.lower()}")
        (out_labels / f"{stem}.txt").write_text("\n".join(rows) + "\n")
        kept += 1
        if args.max_images is not None and kept >= args.max_images:
            break

    print(f"[smokedrink] kept {kept} image+label pairs (drinking -> {args.drinking_as}={target_local})")
    print(f"[smokedrink] skipped (no drinking box): {skipped_no_drinking}")
    print(f"[smokedrink] skipped (no label pair): {skipped_no_pair}")
    print(f"[smokedrink] output: {args.output_dir}")


if __name__ == "__main__":
    main()
