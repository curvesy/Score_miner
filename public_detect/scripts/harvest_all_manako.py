#!/usr/bin/env python3
"""Harvest Detect-beverage training frames from EVERY miner in Manako.

Strategy:
  - Walk the Manako index for all Detect-beverage evals (12k+).
  - Pull every eval JSON in parallel; keep only evals with composite_score >= --min-eval-score.
  - For each kept eval, fetch its responses_key file (miner predictions + frame URLs).
  - Group by frame URL. For each frame, keep the boxes from the SINGLE highest-scoring
    eval that included that frame -- so each frame gets its best-available proxy GT.
  - Download every unique frame, write YOLO labels.

Why this beats probe_via_winner.py:
  - probe_via_winner.py only pulls the current LEADER's evals (small, ~150 frames).
  - Manako has predictions from every miner ever scored on this element. By pulling
    from anyone with composite_score >= 0.4 we get the same frames the leader sees,
    just labeled by whichever miner happened to be evaluated on each frame. The
    leader is the gold standard, but a 0.55 miner's boxes on a frame the leader
    was NEVER evaluated on are still 55% aligned with SAM3 -- training signal.

Example:
  PYTHONPATH=src uv run python scripts/harvest_all_manako.py \\
      --output-dir data/yolo_candidates/beverage_all_miners_v1 \\
      --min-eval-score 0.40 --gt-conf 0.50 --workers 60
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


MANAKO_INDEX = "https://turbo.scoredata.me/manako/index.json"
REF_RE = re.compile(r"manako/[^\"\s]+\.json")
HEADERS = {"User-Agent": "all-miners-harvest/1"}


def fetch_text(url, timeout=60):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace") if r.status == 200 else None
    except Exception:
        return None


def fetch_bytes(url, timeout=60):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read() if r.status == 200 else None
    except Exception:
        return None


def fetch_json(url, timeout=60):
    t = fetch_text(url, timeout=timeout)
    if not t:
        return None
    try:
        return json.loads(t)
    except Exception:
        return None


def absolutize(ref):
    return ref if ref.startswith("http") else f"https://turbo.scoredata.me/{ref.lstrip('/')}"


def _eval_block(ref: str) -> int | None:
    name = ref.rsplit("/", 1)[-1]
    match = re.match(r"(\d+)-", name)
    return int(match.group(1)) if match else None


def step_collect_eval_refs(args):
    if args.index_file:
        print(f"[index] loading {args.index_file}", flush=True)
        idx = args.index_file.read_text()
    else:
        print(f"[index] fetching {MANAKO_INDEX}", flush=True)
        idx = fetch_text(MANAKO_INDEX, timeout=180)
    if not idx:
        sys.exit("failed to fetch manako index")
    refs = sorted(set(REF_RE.findall(idx)), reverse=True)
    bev = [
        r for r in refs
        if "detect-beverage" in r.lower() and "/evaluation/" in r
    ]
    print(f"[index] total beverage evals: {len(bev)}", flush=True)
    blocks = [b for r in bev if (b := _eval_block(r)) is not None]
    if args.last_blocks:
        if not blocks:
            sys.exit("could not infer eval blocks from refs")
        latest = max(blocks)
        args.min_eval_block = max(args.min_eval_block or 0, latest - args.last_blocks)
        print(
            f"[index] latest eval block={latest}; filtering to >= {args.min_eval_block} "
            f"(last_blocks={args.last_blocks})",
            flush=True,
        )
    if args.min_eval_block:
        before = len(bev)
        bev = [r for r in bev if (_eval_block(r) or -1) >= args.min_eval_block]
        print(f"[index] block-filtered evals: {len(bev)} of {before}", flush=True)
    if args.max_evals:
        bev = bev[: args.max_evals]
        print(f"[index] capped to {len(bev)} newest evals", flush=True)
    return bev


def step_load_evals(args, eval_refs):
    """Fetch each eval JSON in parallel. Return list of (cs, hotkey, responses_key)."""
    base = "https://turbo.scoredata.me/"

    def load(ref):
        doc = fetch_json(base + ref)
        if not isinstance(doc, list) or not doc:
            return None
        d = doc[0]
        pl = d.get("payload") or {}
        cs = float(pl.get("composite_score") or 0.0)
        if cs < args.min_eval_score:
            return None
        rk = (pl.get("telemetry") or {}).get("run", {}).get("responses_key")
        if not rk:
            return None
        hot = d.get("hotkey", "?")
        return (cs, hot, rk)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, item in enumerate(pool.map(load, eval_refs), 1):
            if item is not None:
                results.append(item)
            if i % 250 == 0 or i == len(eval_refs):
                print(f"[evals] {i}/{len(eval_refs)} usable={len(results)}", flush=True)
    # show miner distribution
    from collections import Counter
    by_hot = Counter(h for _, h, _ in results)
    print(f"[evals] usable={len(results)} from {len(by_hot)} unique miners")
    for hk, n in by_hot.most_common(8):
        print(f"  {hk[:16]}.. {n} evals")
    return results


def step_pull_responses(args, evals):
    """Fetch responses files. Return {frame_url: (cs, list[box])} where
    each frame is labeled by the SINGLE highest-scoring eval that included it."""
    base = "https://turbo.scoredata.me/"

    def load_resp(item):
        cs, hot, rk = item
        doc = fetch_json(base + rk)
        return cs, hot, rk, doc

    best_per_url = {}
    n_with_boxes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (cs, hot, rk, rdoc) in enumerate(pool.map(load_resp, evals), 1):
            if isinstance(rdoc, dict):
                frames = rdoc.get("frames") or []
                preds = ((rdoc.get("predictions") or {}).get("frames") or [])
                url_by_id = {int(f["frame_id"]): f["url"] for f in frames if f.get("url")}
                for fr in preds:
                    fid = int(fr.get("frame_id", -1))
                    url = url_by_id.get(fid)
                    if not url:
                        continue
                    boxes = fr.get("boxes") or []
                    if not boxes:
                        continue
                    prev = best_per_url.get(url)
                    if prev is None or cs > prev[0]:
                        best_per_url[url] = (cs, boxes)
                    n_with_boxes += 1
            if i % 100 == 0 or i == len(evals):
                print(f"[resp] {i}/{len(evals)}  unique_frames={len(best_per_url)}", flush=True)
    print(f"[resp] frames-with-boxes encounters: {n_with_boxes}", flush=True)
    print(f"[resp] unique frames after best-per-frame: {len(best_per_url)}", flush=True)
    return best_per_url


def step_download(args, url_to_meta):
    out_dir = args.output_dir / "images_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = sorted(url_to_meta.keys())

    def dl(url):
        name = url.rsplit("/", 1)[-1]
        target = out_dir / name
        if target.exists() and target.stat().st_size > 0:
            return url, target
        data = fetch_bytes(url)
        if not data:
            return url, None
        target.write_bytes(data)
        return url, target

    url_to_path = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (url, path) in enumerate(pool.map(dl, urls), 1):
            if path is not None:
                url_to_path[url] = path
            if i % 100 == 0 or i == len(urls):
                print(f"[download] {i}/{len(urls)} ok={len(url_to_path)}", flush=True)
    return url_to_path


def step_write_labels(args, url_to_meta, url_to_path):
    from PIL import Image
    images_dir = args.output_dir / "images" / "train"
    labels_dir = args.output_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    n_with = 0
    n_boxes = 0
    score_hist = {}
    for url, (cs, boxes) in url_to_meta.items():
        src = url_to_path.get(url)
        if not src or not src.exists():
            continue
        with Image.open(src) as im:
            w, h = im.size
        rows = []
        for b in boxes:
            try:
                if float(b.get("conf", 0)) < args.gt_conf:
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

        if not rows:
            continue

        dst_img = images_dir / src.name
        if not dst_img.exists():
            shutil.copy2(src, dst_img)
        (labels_dir / f"{src.stem}.txt").write_text("\n".join(rows) + "\n")
        n_with += 1
        n_boxes += len(rows)
        bucket = round(cs, 1)
        score_hist[bucket] = score_hist.get(bucket, 0) + 1

    (args.output_dir / "data.yaml").write_text("\n".join([
        f"path: {args.output_dir.resolve()}",
        "train: images/train",
        "val: images/train",
        "nc: 3",
        "names:",
        "  0: cup",
        "  1: bottle",
        "  2: can",
        "",
    ]))
    print(f"\n[final] {n_with} frames labeled, {n_boxes} total boxes (conf>={args.gt_conf})")
    print(f"[final] miner composite_score histogram of source labels:")
    for bucket in sorted(score_hist):
        print(f"  cs~{bucket:.1f}: {score_hist[bucket]} frames")
    print(f"[final] output: {args.output_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--min-eval-score", type=float, default=0.40,
                   help="only use evals with composite_score >= this")
    p.add_argument("--gt-conf", type=float, default=0.50,
                   help="miner box conf floor for inclusion as proxy GT")
    p.add_argument("--max-evals", type=int, default=None,
                   help="cap newest N evals to scan (default: all 12k)")
    p.add_argument("--index-file", type=Path, default=None,
                   help="load Manako index from a local file instead of fetching")
    p.add_argument("--min-eval-block", type=int, default=None,
                   help="only scan eval refs with block prefix >= this value")
    p.add_argument("--last-blocks", type=int, default=None,
                   help="only scan eval refs within N blocks of the newest eval block")
    p.add_argument("--workers", type=int, default=60)
    p.add_argument("--skip-scrape", action="store_true",
                   help="reuse output-dir/images_raw + cached_meta.json")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_cache = args.output_dir / "harvest_meta.json"

    if args.skip_scrape and meta_cache.exists():
        print(f"[scrape] reusing {meta_cache}")
        cache = json.loads(meta_cache.read_text())
        url_to_meta = {url: (cs, boxes) for url, (cs, boxes) in cache.items()}
    else:
        eval_refs = step_collect_eval_refs(args)
        evals = step_load_evals(args, eval_refs)
        if not evals:
            sys.exit("no usable evals -- lower --min-eval-score")
        url_to_meta = step_pull_responses(args, evals)
        if not url_to_meta:
            sys.exit("no frames harvested")
        meta_cache.write_text(json.dumps(
            {url: list(v) for url, v in url_to_meta.items()}
        ))

    url_to_path = step_download(args, url_to_meta)
    step_write_labels(args, url_to_meta, url_to_path)


if __name__ == "__main__":
    main()
