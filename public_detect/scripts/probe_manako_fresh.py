#!/usr/bin/env python3
"""Fetch FRESH Manako eval frames (validator challenge images) and probe a model.

Workflow:
  1. Re-pull turbo.scoredata.me/manako/index.json
  2. Dedupe to one eval per (hotkey, challenge_id) — same video = same frames
  3. Download every challenge-objects PNG referenced from those evals
  4. Filter out frames you already have locally (--exclude-dir)
  5. Auto-label NEW frames with YOLOWorld at --autolabel-conf (stricter than train conf)
  6. Run --model on the new frames and run the same map50 + fp_score formula
     the validator uses, so the printed score is directly comparable to the live
     console score.

This is the closest "live-distribution proxy" you can build locally without
querying SAM3 yourself.

Example:
  PYTHONPATH=src uv run python scripts/probe_manako_fresh.py \\
    --model runs/beverage/yolo11n_phase4_external_v1_local/weights/best.pt \\
    --exclude-dir data/yolo_candidates/beverage_manako_autolabeled/images/train \\
    --output-dir data/yolo_candidates/beverage_manako_fresh_v1
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


INDEX_URL = "https://turbo.scoredata.me/manako/index.json"
IMG_RE = re.compile(
    r"https://[^\"'\s]+/challenge-objects/[^\"'\s]+\.(?:png|jpg|jpeg)",
    re.IGNORECASE,
)
RESP_RE = re.compile(r'"responses_key"\s*:\s*"([^"]+)"')
REF_RE = re.compile(r"manako/[^\"\s]+\.json")
HEADERS = {"User-Agent": "manako-fresh-probe/1"}


def fetch_text(url: str, timeout: int = 60) -> str | None:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None
            return r.read().decode("utf-8", "replace")
    except Exception as exc:
        print(f"  [fetch] {type(exc).__name__}: {exc} for {url}", flush=True)
        return None


def fetch_bytes(url: str, timeout: int = 60) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None
            return r.read()
    except Exception:
        return None


def absolutize(ref: str) -> str:
    return ref if ref.startswith("http") else f"https://turbo.scoredata.me/{ref.lstrip('/')}"


def step_scrape(args) -> set[str]:
    """Walk Manako, return unique image URLs for Detect-beverage-detect evals."""
    if args.index_file:
        print(f"[scrape] loading index from {args.index_file}", flush=True)
        idx = Path(args.index_file).read_text()
    else:
        print(f"[scrape] fetching {INDEX_URL}", flush=True)
        idx = fetch_text(INDEX_URL, timeout=180)
    if not idx:
        sys.exit("failed to fetch Manako index")

    refs = sorted(set(REF_RE.findall(idx)))
    bev = [r for r in refs if args.element_filter.lower() in r.lower()]

    # one eval per (hotkey, challenge_id), keep newest
    by_key: dict[tuple[str, str], str] = {}
    for ref in bev:
        parts = ref.split("/")
        if len(parts) < 6 or parts[-2] != "evaluation":
            continue
        key = (parts[-4], parts[-3])
        if key not in by_key or ref > by_key[key]:
            by_key[key] = ref
    bev = sorted(by_key.values())
    print(f"[scrape] {len(bev)} unique (hotkey, challenge_id) refs", flush=True)

    full_refs = [absolutize(r) for r in bev]
    eval_texts: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, txt in enumerate(pool.map(fetch_text, full_refs), 1):
            if txt:
                eval_texts.append(txt)
            if i % 500 == 0 or i == len(full_refs):
                print(f"[scrape] eval {i}/{len(full_refs)} hits={len(eval_texts)}", flush=True)

    image_urls: set[str] = set()
    response_refs: set[str] = set()
    for text in eval_texts:
        for m in IMG_RE.findall(text):
            image_urls.add(m)
        for m in RESP_RE.findall(text):
            response_refs.add(absolutize(m))

    response_list = sorted(response_refs)
    if response_list:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, txt in enumerate(pool.map(fetch_text, response_list), 1):
                if txt:
                    for m in IMG_RE.findall(txt):
                        image_urls.add(m)
                if i % 500 == 0 or i == len(response_list):
                    print(f"[scrape] response {i}/{len(response_list)} total_imgs={len(image_urls)}", flush=True)

    print(f"[scrape] total unique image URLs: {len(image_urls)}", flush=True)
    return image_urls


def step_filter_and_download(args, image_urls: set[str]) -> list[Path]:
    """Skip frames already present in --exclude-dir, download the rest."""
    seen: set[str] = set()
    for d in args.exclude_dir:
        if d and Path(d).is_dir():
            for f in Path(d).iterdir():
                seen.add(f.name)
    print(f"[filter] {len(seen)} frames already present in exclude-dirs", flush=True)

    fresh_urls = [u for u in sorted(image_urls) if u.rsplit("/", 1)[-1] not in seen]
    print(f"[filter] {len(fresh_urls)} URLs are NEW", flush=True)

    out_dir = args.output_dir / "images_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    def dl(url: str) -> Path | None:
        name = url.rsplit("/", 1)[-1]
        target = out_dir / name
        if target.exists() and target.stat().st_size > 0:
            return target
        data = fetch_bytes(url)
        if not data:
            return None
        target.write_bytes(data)
        return target

    downloaded: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, path in enumerate(pool.map(dl, fresh_urls), 1):
            if path is not None:
                downloaded.append(path)
            if i % 50 == 0 or i == len(fresh_urls):
                print(f"[download] {i}/{len(fresh_urls)} ok={len(downloaded)}", flush=True)
    return downloaded


def step_autolabel(args, image_paths: list[Path]) -> tuple[Path, Path]:
    """YOLOWorld auto-label fresh frames to YOLO images/labels/train layout."""
    from ultralytics import YOLOWorld

    classes = ["cup", "bottle", "can"]
    print(f"[autolabel] model=yolov8s-worldv2.pt conf={args.autolabel_conf} classes={classes}", flush=True)

    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(classes)

    images_out = args.output_dir / "images" / "train"
    labels_out = args.output_dir / "labels" / "train"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    n_boxes = 0
    n_imgs_with_box = 0
    for img_path in image_paths:
        results = model.predict(str(img_path), conf=args.autolabel_conf, imgsz=960, verbose=False)
        if not results:
            continue
        r = results[0]
        h, w = r.orig_shape
        lines: list[str] = []
        if r.boxes is not None:
            for box in r.boxes:
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

        dst_img = images_out / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        (labels_out / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if lines:
            n_imgs_with_box += 1
            n_boxes += len(lines)

    # write minimal data.yaml so existing scorers can read it
    (args.output_dir / "data.yaml").write_text(
        "\n".join([
            f"path: {args.output_dir.resolve()}",
            "train: images/train",
            "val: images/train",
            "nc: 3",
            "names:",
            "  0: cup",
            "  1: bottle",
            "  2: can",
            "",
        ])
    )
    print(f"[autolabel] {n_imgs_with_box}/{len(image_paths)} imgs have boxes, total boxes={n_boxes}", flush=True)
    print(f"[autolabel] data.yaml: {args.output_dir / 'data.yaml'}")
    return images_out, args.output_dir / "data.yaml"


def step_score(args, data_yaml: Path) -> None:
    """Compute the same map50+fp_score the validator uses."""
    from public_detect.score_eval import (
        Box,
        evaluate_score,
        load_yolo_dataset,
        load_yolo_ground_truth,
    )
    from ultralytics import YOLO

    image_paths, class_names = load_yolo_dataset(data_yaml)
    gt_boxes, _ = load_yolo_ground_truth(data_yaml)
    print(f"[score] images={len(image_paths)} gt_boxes={len(gt_boxes)}", flush=True)

    yolo = YOLO(str(args.model))
    results = yolo.predict(
        source=[str(p) for p in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.5,
        max_det=100,
        verbose=False,
    )
    pred_boxes: list[Box] = []
    for img_path, r in zip(image_paths, results, strict=True):
        if r.boxes is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        for coords, score, c in zip(xyxy, conf, cls, strict=True):
            pred_boxes.append(Box(
                image_id=img_path.stem,
                cls=int(c),
                xyxy=tuple(float(v) for v in coords),
                conf=float(score),
            ))
    metrics = evaluate_score(gt_boxes, pred_boxes, class_names)

    print()
    print("=" * 60)
    print(f"  FRESH MANAKO PROBE — model={args.model.name}")
    print("=" * 60)
    print(f"  images          : {len(image_paths)}")
    print(f"  gt_boxes        : {metrics.gt}     (pseudo-GT from YOLOWorld @ {args.autolabel_conf})")
    print(f"  pred_boxes      : {metrics.predictions}")
    print()
    print(f"  map50           : {metrics.map50:.4f}")
    print(f"  precision       : {metrics.precision:.4f}")
    print(f"  recall          : {metrics.recall:.4f}")
    print(f"  false_positive  : {metrics.fp_score:.4f}")
    print()
    score = 0.6 * metrics.map50 + 0.4 * metrics.fp_score
    print(f"  manifest score  : {score:.4f}   (0.6*map50 + 0.4*fp_score)")
    print()
    print(f"  per-class:")
    for c in metrics.classes:
        print(f"    {c.name:8s}: ap50={c.ap50:.3f} prec={c.precision:.3f} "
              f"rec={c.recall:.3f}  tp={c.tp} fp={c.fp} fn={c.fn}")
    print()
    print("  reference:")
    print("    live top miner : 0.7438")
    print("    live baseline  : 0.1955")
    print("    target         : 0.9400")
    print()
    print("  NOTE: this probe uses YOLOWorld pseudo-GT, NOT SAM3 like the validator.")
    print("        Live SAM3 GT is usually MORE forgiving than YOLOWorld.")
    print("        Expect live score >= this probe number, typically by +0.05.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path,
                   help="path to trained best.pt to evaluate")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="where to write images/train + labels/train + data.yaml")
    p.add_argument("--exclude-dir", action="append", default=[],
                   help="repeatable; skip frames whose basename exists in this dir")
    p.add_argument("--element-filter", default="Detect-beverage")
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--index-file", type=Path, default=None,
                   help="use a local copy of index.json instead of fetching")
    p.add_argument("--autolabel-conf", type=float, default=0.20,
                   help="YOLOWorld confidence floor for pseudo-GT (stricter than 0.10)")
    p.add_argument("--conf", type=float, default=0.10,
                   help="model inference confidence floor")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--skip-scrape", action="store_true",
                   help="reuse output-dir/images_raw instead of rescraping")
    p.add_argument("--skip-autolabel", action="store_true",
                   help="reuse existing images/train+labels/train, only re-score")
    args = p.parse_args()

    if args.skip_autolabel:
        data_yaml = args.output_dir / "data.yaml"
        if not data_yaml.exists():
            sys.exit(f"--skip-autolabel set but {data_yaml} missing")
        step_score(args, data_yaml)
        return

    if args.skip_scrape:
        raw_dir = args.output_dir / "images_raw"
        fresh = sorted(raw_dir.iterdir()) if raw_dir.exists() else []
        print(f"[scrape] skip, reusing {len(fresh)} files from {raw_dir}")
    else:
        urls = step_scrape(args)
        fresh = step_filter_and_download(args, urls)

    if not fresh:
        sys.exit("no fresh frames to probe on")

    _, data_yaml = step_autolabel(args, fresh)
    step_score(args, data_yaml)


if __name__ == "__main__":
    main()
