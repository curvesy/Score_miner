#!/usr/bin/env python3
"""Auto-label images with YOLOWorld for cup/bottle/can.

YOLOWorld is open-vocabulary YOLO, bundled with ultralytics. No extra install.

Output layout (YOLO format, train split only — splitter goes in later):
  <output-dir>/images/train/*.png
  <output-dir>/labels/train/*.txt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLOWorld


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, type=Path,
                   help="dir containing raw images")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="YOLO dataset dir to write into")
    p.add_argument("--classes", default="cup,bottle,can",
                   help="comma-separated class names in YOLO class-id order")
    p.add_argument("--conf", type=float, default=0.10,
                   help="confidence threshold for keeping a box")
    p.add_argument("--model", default="yolov8s-worldv2.pt")
    p.add_argument("--imgsz", type=int, default=960)
    args = p.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    print(f"[autolabel] classes={classes} conf={args.conf} model={args.model}")

    model = YOLOWorld(args.model)
    model.set_classes(classes)

    images_out = args.output_dir / "images" / "train"
    labels_out = args.output_dir / "labels" / "train"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    print(f"[autolabel] labeling {len(image_paths)} images")

    class_counts = {c: 0 for c in classes}
    total_boxes = 0
    for img_path in image_paths:
        results = model.predict(str(img_path), conf=args.conf, imgsz=args.imgsz, verbose=False)
        if not results:
            continue
        result = results[0]
        h, w = result.orig_shape
        lines = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                if cls < 0 or cls >= len(classes):
                    continue
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                if bw <= 0 or bh <= 0:
                    continue
                xc = ((x1 + x2) / 2.0) / w
                yc = ((y1 + y2) / 2.0) / h
                lines.append(f"{cls} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
                class_counts[classes[cls]] += 1
                total_boxes += 1

        target_img = images_out / img_path.name
        target_lbl = labels_out / f"{img_path.stem}.txt"
        if not target_img.exists():
            shutil.copy2(img_path, target_img)
        target_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"  {img_path.name}: {len(lines)} boxes")

    print(f"\n[autolabel] done")
    print(f"  total boxes  : {total_boxes}")
    print(f"  by class     : {class_counts}")
    print(f"  images dir   : {images_out}")
    print(f"  labels dir   : {labels_out}")


if __name__ == "__main__":
    main()
