#!/usr/bin/env python3
"""Merge multiple YOLO-format source dirs into one Phase 4 dataset.

Each source is a YOLO dir with images/train and labels/train.
Sources are tagged so the manifest records origin per image.
Validation set is reserved for trusted starter images (anchor).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from public_detect.elements import load_element_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def collect_pairs(source_dir: Path, split: str = "train") -> list[tuple[Path, Path]]:
    images_dir = source_dir / "images" / split
    labels_dir = source_dir / "labels" / split
    if not images_dir.exists():
        return []
    pairs = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = labels_dir / f"{image.stem}.txt"
        if label.exists():
            pairs.append((image, label))
    return pairs


def validate_label_file(label_path: Path, class_count: int) -> dict[int, int]:
    counts: dict[int, int] = {idx: 0 for idx in range(class_count)}
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        bits = line.strip().split()
        if not bits:
            continue
        if len(bits) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected YOLO row with 5 columns")
        cls = int(float(bits[0]))
        if cls < 0 or cls >= class_count:
            raise ValueError(f"{label_path}:{line_number}: class id {cls} outside 0..{class_count - 1}")
        counts[cls] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--element-config", required=True, type=project_path)
    parser.add_argument("--output-dir", required=True, type=project_path)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help=(
            "Repeatable. Format: tag:path[:cap]. "
            "Tag examples: starter, manako_sam3, taco, hard_neg. "
            "Path is a YOLO dir (with images/train,labels/train). "
            "cap is optional max images for this source."
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of NON-starter sources to send to val, in addition to starter val anchor.",
    )
    parser.add_argument(
        "--starter-tag",
        default="starter",
        help="Tag whose images are copied to both train and val as the trusted anchor.",
    )
    parser.add_argument(
        "--no-starter-train-copy",
        action="store_true",
        help="Only put starter images in val. Not recommended for 7-image starter packs.",
    )
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()
    if args.val_fraction < 0 or args.val_fraction >= 1:
        raise SystemExit("--val-fraction must be >= 0 and < 1")

    spec = load_element_spec(args.element_config)
    random.seed(args.seed)

    output = args.output_dir
    images_train = output / "images" / "train"
    images_val = output / "images" / "val"
    labels_train = output / "labels" / "train"
    labels_val = output / "labels" / "val"
    for path in (images_train, images_val, labels_train, labels_val):
        path.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {"train": 0, "val": 0}
    source_counts: dict[str, dict[str, int]] = {}
    class_counts: dict[int, int] = {idx: 0 for idx in range(len(spec.objects))}

    for entry in args.source:
        parts = entry.split(":")
        if len(parts) < 2:
            raise SystemExit(f"bad --source value: {entry}")
        tag = parts[0]
        src_path = project_path(parts[1])
        # 3rd field: cap (max images from this source). Use "" to skip cap.
        cap = int(parts[2]) if len(parts) >= 3 and parts[2] else None
        # 4th field: train-only multiplier. e.g. "3x" copies each train pair 3 times.
        # Use this to oversample Score-distribution sources without polluting val.
        mult = 1
        if len(parts) >= 4 and parts[3]:
            mult_raw = parts[3].lower().rstrip("x")
            mult = max(1, int(mult_raw))

        pairs = collect_pairs(src_path)
        if not pairs:
            print(f"[warn] no pairs found for {tag} at {src_path}")
            continue
        random.shuffle(pairs)
        if cap is not None:
            pairs = pairs[:cap]

        is_starter = tag == args.starter_tag
        source_counts.setdefault(tag, {"train": 0, "val": 0})
        for source_index, (image_path, label_path) in enumerate(pairs):
            label_counts = validate_label_file(label_path, len(spec.objects))
            target_splits = ["val"] if is_starter else []
            if is_starter and not args.no_starter_train_copy:
                target_splits.append("train")
            if not is_starter:
                target_splits.append("val" if random.random() < args.val_fraction else "train")

            for split in target_splits:
                dst_img = images_val if split == "val" else images_train
                dst_lbl = labels_val if split == "val" else labels_train
                # When oversampling, only duplicate train; val stays at one copy.
                copies = mult if (split == "train" and mult > 1) else 1
                for copy_idx in range(copies):
                    suffix_tag = "" if copy_idx == 0 else f"_x{copy_idx}"
                    stem = f"{tag}_{source_index:06d}{suffix_tag}_{image_path.stem}"
                    new_img = dst_img / f"{stem}{image_path.suffix.lower()}"
                    new_lbl = dst_lbl / f"{stem}.txt"
                    shutil.copy2(image_path, new_img)
                    shutil.copy2(label_path, new_lbl)

                    for cls, value in label_counts.items():
                        class_counts[cls] = class_counts.get(cls, 0) + value

                    counts[split] += 1
                    source_counts[tag][split] += 1
                    rows.append({
                        "source_tag": tag,
                        "source_image": str(image_path),
                        "image": str(new_img),
                        "label": str(new_lbl),
                        "split": split,
                        "boxes": sum(label_counts.values()),
                        "multiplier_copy": copy_idx,
                    })

    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(spec.objects))
    (output / "data.yaml").write_text("\n".join([
        f"path: {output.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(spec.objects)}",
        "names:",
        names_block,
        "",
    ]))

    manifest = {
        "element_id": spec.element_id,
        "objects": list(spec.objects),
        "counts": counts,
        "class_counts": {spec.objects[k]: v for k, v in class_counts.items()},
        "source_counts": source_counts,
        "sources": list({row["source_tag"] for row in rows}),
        "rows": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
