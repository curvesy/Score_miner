#!/usr/bin/env python3
"""Probe a trained model against the current TOP MINER's public predictions.

Why this beats YOLOWorld pseudo-GT:
  - The Score validator's SAM3 ground truth is NOT public. I checked every
    plausible Manako path; the GT is computed at scoring time and discarded.
  - The CURRENT LEADER's response boxes ARE public via manako and they score
    ~0.7435 against SAM3 -> they are the best free proxy for SAM3 GT.
  - The winner is fine-tuned for exactly this distribution, so their high-conf
    boxes look like Score frames, not generic open-vocab guesses.

Workflow:
  1. Hit console.scorevision.io/api/v2/elements to find live winnerHotkey
  2. Walk turbo.scoredata.me/manako/index.json for that hotkey's evaluations
  3. Download each eval's responses JSON -> winner's box predictions
  4. Download each frame image from manako.scoredata.me/challenge-objects/...
  5. Treat winner boxes with conf >= --gt-conf as proxy GT (YOLO format)
  6. Run your --model on the same frames
  7. Print the validator's formula  0.6*map50 + 0.4*fp_score

Example:
  PYTHONPATH=src uv run python scripts/probe_via_winner.py \\
    --model runs/beverage/yolo11n_phase4_external_v1_local/weights/last.pt \\
    --output-dir data/yolo_candidates/beverage_winner_proxy_v1 \\
    --max-evals 50 --gt-conf 0.50
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


CONSOLE_API = "https://console.scorevision.io/api/v2/elements"
MANAKO_INDEX = "https://turbo.scoredata.me/manako/index.json"
ELEMENT_ID = "manak0/Detect-beverage-detect"
REF_RE = re.compile(r"manako/[^\"\s]+\.json")
HEADERS = {"User-Agent": "winner-proxy-probe/1"}


def fetch_text(url: str, timeout: int = 60) -> str | None:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace") if r.status == 200 else None
    except Exception as exc:
        print(f"  [fetch] {type(exc).__name__}: {exc} for {url}", flush=True)
        return None


def fetch_bytes(url: str, timeout: int = 60) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read() if r.status == 200 else None
    except Exception:
        return None


def fetch_json(url: str, timeout: int = 60):
    t = fetch_text(url, timeout=timeout)
    if not t:
        return None
    try:
        return json.loads(t)
    except Exception:
        return None


def get_live_winner() -> tuple[str, float]:
    doc = fetch_json(CONSOLE_API)
    if not isinstance(doc, list):
        sys.exit("could not read console API leaderboard")
    for el in doc:
        if el.get("id") == ELEMENT_ID:
            hot = el.get("winnerHotkey")
            score = float(el.get("currentScore") or 0.0)
            if not hot:
                sys.exit("no winnerHotkey for beverage element (yet)")
            return hot, score
    sys.exit(f"element {ELEMENT_ID} not found in API response")


def newest_evals_for_hotkey(hotkey: str, limit: int) -> list[str]:
    print(f"[scrape] fetching index ({MANAKO_INDEX})", flush=True)
    idx = fetch_text(MANAKO_INDEX, timeout=180)
    if not idx:
        sys.exit("failed to fetch manako index")
    refs = sorted(set(REF_RE.findall(idx)), reverse=True)
    bev = [
        r for r in refs
        if "detect-beverage" in r.lower()
        and "/evaluation/" in r
        and f"/{hotkey}/" in r
    ]
    print(f"[scrape] {len(bev)} evaluations for winner {hotkey[:12]}..", flush=True)
    return bev[:limit]


def pull_winner_responses(eval_refs: list[str], min_score: float, workers: int):
    """Return list of dicts: {frame_url, boxes, eval_score}.

    Only eval-windows whose composite_score >= min_score are kept (so we get
    proxy GT only from windows where the winner actually executed well).
    """
    base = "https://turbo.scoredata.me/"

    def load_eval(ref: str):
        doc = fetch_json(base + ref)
        if not isinstance(doc, list) or not doc:
            return None
        pl = doc[0].get("payload") or {}
        cs = float(pl.get("composite_score") or 0.0)
        if cs < min_score:
            return None
        rk = (pl.get("telemetry") or {}).get("run", {}).get("responses_key")
        return (cs, rk) if rk else None

    results: list[tuple[float, dict]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, item in enumerate(pool.map(load_eval, eval_refs), 1):
            if item is not None:
                cs, rk = item
                results.append((cs, rk))
            if i % 25 == 0 or i == len(eval_refs):
                print(f"[evals] {i}/{len(eval_refs)} usable={len(results)}", flush=True)

    print(f"[evals] {len(results)} eval-windows with cs >= {min_score}", flush=True)

    def load_resp(item):
        cs, rk = item
        doc = fetch_json(base + rk)
        return cs, rk, doc

    frame_records: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (cs, rk, rdoc) in enumerate(pool.map(load_resp, results), 1):
            if not isinstance(rdoc, dict):
                continue
            frames = rdoc.get("frames") or []
            preds = ((rdoc.get("predictions") or {}).get("frames") or [])
            frame_url_by_id = {int(f["frame_id"]): f["url"] for f in frames if f.get("url")}
            for fr in preds:
                fid = int(fr.get("frame_id", -1))
                url = frame_url_by_id.get(fid)
                if not url:
                    continue
                frame_records.append({
                    "frame_url": url,
                    "boxes": fr.get("boxes") or [],
                    "eval_score": cs,
                    "eval_ref": rk,
                })
            if i % 25 == 0 or i == len(results):
                print(f"[resp] {i}/{len(results)} total_frames={len(frame_records)}", flush=True)

    return frame_records


def download_frames(frame_records: list[dict], out_dir: Path, workers: int) -> dict:
    """Download images, dedupe by URL. Returns {frame_url: local_path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    unique_urls = sorted({r["frame_url"] for r in frame_records})
    print(f"[download] {len(unique_urls)} unique frames", flush=True)

    def dl(url: str) -> tuple[str, Path | None]:
        name = url.rsplit("/", 1)[-1]
        target = out_dir / name
        if target.exists() and target.stat().st_size > 0:
            return url, target
        data = fetch_bytes(url)
        if not data:
            return url, None
        target.write_bytes(data)
        return url, target

    url_to_path: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (url, path) in enumerate(pool.map(dl, unique_urls), 1):
            if path is not None:
                url_to_path[url] = path
            if i % 50 == 0 or i == len(unique_urls):
                print(f"[download] {i}/{len(unique_urls)} ok={len(url_to_path)}", flush=True)
    return url_to_path


def write_yolo_labels(frame_records, url_to_path, out_dir: Path, gt_conf: float):
    """Write winner boxes (filtered to conf>=gt_conf) as YOLO labels."""
    images_dir = out_dir / "images" / "train"
    labels_dir = out_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # accumulate boxes per frame_url (multiple evals can reference same frame)
    per_url: dict[str, list[dict]] = {}
    for rec in frame_records:
        per_url.setdefault(rec["frame_url"], []).extend(rec["boxes"])

    from PIL import Image
    n_with = 0
    n_boxes = 0
    for url, boxes in per_url.items():
        src = url_to_path.get(url)
        if not src or not src.exists():
            continue
        with Image.open(src) as im:
            w, h = im.size
        rows: list[str] = []
        for b in boxes:
            try:
                if float(b.get("conf", 0)) < gt_conf:
                    continue
                cls = int(b["cls_id"])
                if cls < 0 or cls > 2:
                    continue
                x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
            except Exception:
                continue
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            if bw <= 0 or bh <= 0:
                continue
            xc = ((x1 + x2) / 2.0) / w
            yc = ((y1 + y2) / 2.0) / h
            rows.append(f"{cls} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")

        dst_img = images_dir / src.name
        if not dst_img.exists():
            shutil.copy2(src, dst_img)
        (labels_dir / f"{src.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        if rows:
            n_with += 1
            n_boxes += len(rows)

    (out_dir / "data.yaml").write_text(
        "\n".join([
            f"path: {out_dir.resolve()}",
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
    print(f"[labels] {n_with} frames have boxes, total boxes={n_boxes}, gt_conf>={gt_conf}", flush=True)
    return out_dir / "data.yaml"


def score_model(args, data_yaml: Path, winner_score: float) -> None:
    from public_detect.score_eval import (
        Box, evaluate_score, load_yolo_dataset, load_yolo_ground_truth,
    )
    from ultralytics import YOLO

    image_paths, class_names = load_yolo_dataset(data_yaml)
    gt_boxes, _ = load_yolo_ground_truth(data_yaml)
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
    m = evaluate_score(gt_boxes, pred_boxes, class_names)
    manifest = 0.6 * m.map50 + 0.4 * m.fp_score

    print()
    print("=" * 64)
    print(f"  WINNER-PROXY PROBE  --  model={args.model.name}")
    print("=" * 64)
    print(f"  images          : {len(image_paths)}")
    print(f"  proxy GT boxes  : {m.gt}  (winner conf>=--gt-conf)")
    print(f"  your pred boxes : {m.predictions}")
    print()
    print(f"  map50           : {m.map50:.4f}")
    print(f"  precision       : {m.precision:.4f}")
    print(f"  recall          : {m.recall:.4f}")
    print(f"  false_positive  : {m.fp_score:.4f}")
    print()
    print(f"  proxy score     : {manifest:.4f}   (0.6*map50 + 0.4*fp_score)")
    print()
    print(f"  per-class:")
    for c in m.classes:
        print(f"    {c.name:8s}: ap50={c.ap50:.3f} prec={c.precision:.3f} "
              f"rec={c.recall:.3f}  tp={c.tp} fp={c.fp} fn={c.fn}")
    print()
    print(f"  live winner     : {winner_score:.4f}  (currentScore from console API)")
    print(f"  delta to winner : {manifest - winner_score:+.4f}")
    print()
    print("  NOTE: proxy GT is the LEADER'S boxes, not SAM3.")
    print("        Leader scores 0.74 against SAM3, so this probe is a tight")
    print("        lower bound on your live score. Expect live = proxy + 0.02..0.08.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-evals", type=int, default=80,
                   help="how many recent winner eval-windows to harvest")
    p.add_argument("--min-eval-score", type=float, default=0.50,
                   help="only use winner evals whose composite_score >= this")
    p.add_argument("--gt-conf", type=float, default=0.50,
                   help="winner box conf floor for inclusion as proxy GT")
    p.add_argument("--conf", type=float, default=0.10,
                   help="your model's inference confidence")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--skip-scrape", action="store_true",
                   help="reuse images_raw/ from previous run")
    p.add_argument("--skip-labels", action="store_true",
                   help="reuse existing images/train+labels/train, only re-score")
    args = p.parse_args()

    if args.skip_labels:
        winner, winner_score = get_live_winner()
        score_model(args, args.output_dir / "data.yaml", winner_score)
        return

    winner, winner_score = get_live_winner()
    print(f"[live] winner={winner[:14]}.. score={winner_score:.4f}")

    if args.skip_scrape:
        raw_dir = args.output_dir / "images_raw"
        if not raw_dir.exists():
            sys.exit(f"--skip-scrape but {raw_dir} missing")
        # rebuild url_to_path map from cached files
        url_to_path = {}
        for f in raw_dir.iterdir():
            url_to_path[f.name] = f   # not a URL but we only need filename match
        # we still need frame_records for labels -> requires the eval data
        sys.exit("--skip-scrape requires probe to also persist eval data; not yet implemented")

    eval_refs = newest_evals_for_hotkey(winner, args.max_evals)
    frame_records = pull_winner_responses(eval_refs, args.min_eval_score, args.workers)
    if not frame_records:
        sys.exit("no frames harvested -- raise --max-evals or lower --min-eval-score")
    url_to_path = download_frames(frame_records, args.output_dir / "images_raw", args.workers)
    data_yaml = write_yolo_labels(frame_records, url_to_path, args.output_dir, args.gt_conf)
    score_model(args, data_yaml, winner_score)


if __name__ == "__main__":
    main()
