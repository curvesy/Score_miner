#!/usr/bin/env python3
"""Fast parallel Manako image scraper.

Walks the manako index, filters refs by element, fetches every eval JSON +
linked responses_key JSON in parallel, extracts every challenge-objects PNG URL,
deduplicates, and downloads them.

Stdlib only. ThreadPoolExecutor with ~40 workers.
"""

from __future__ import annotations

import argparse
import re
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
HEADERS = {"User-Agent": "manako-fast-scraper/1"}


def fetch_text(url: str, timeout: int = 30, *, verbose: bool = False) -> str | None:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                if verbose:
                    print(f"  [fetch] HTTP {r.status} for {url}", flush=True)
                return None
            return r.read().decode("utf-8", "replace")
    except Exception as exc:
        if verbose:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--element-filter", required=True,
                   help="substring match, e.g. Detect-beverage")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-refs", type=int, default=None,
                   help="cap number of eval refs scanned (debug)")
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--unique-per-challenge", action="store_true",
                   help="keep only one eval per (hotkey, challenge_id) — same video = same frames, so this is what you want")
    p.add_argument("--index-file", type=Path, default=None,
                   help="load index.json from a local file instead of fetching")
    args = p.parse_args()

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if args.index_file:
        print(f"[scrape] loading index from {args.index_file}", flush=True)
        idx = args.index_file.read_text()
    else:
        print(f"[scrape] fetching index {INDEX_URL}", flush=True)
        idx = fetch_text(INDEX_URL, timeout=180, verbose=True)
    if not idx:
        sys.exit("failed to fetch index — try: curl -sS 'https://turbo.scoredata.me/manako/index.json' > /tmp/idx.json  and rerun with --index-file /tmp/idx.json")

    refs = sorted(set(REF_RE.findall(idx)))
    bev = [r for r in refs if args.element_filter.lower() in r.lower()]

    if args.unique_per_challenge:
        # ref shape: manako/manak0_Detect-X/<hotkey>/<challenge_id>/evaluation/<eval>.json
        # keep highest eval block per (hotkey, challenge_id) — same video = same frames
        by_key: dict[tuple[str, str], str] = {}
        for ref in bev:
            parts = ref.split("/")
            if len(parts) < 6 or parts[-2] != "evaluation":
                continue
            key = (parts[-4], parts[-3])  # hotkey, challenge_id
            if key not in by_key or ref > by_key[key]:
                by_key[key] = ref
        bev = sorted(by_key.values())
        print(f"[scrape] deduped to {len(bev)} unique (hotkey, challenge_id) refs", flush=True)

    full_refs = [absolutize(r) for r in bev]
    if args.max_refs:
        full_refs = full_refs[: args.max_refs]
    print(f"[scrape] {len(full_refs)} '{args.element_filter}' refs to scan", flush=True)

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
    print(f"[scrape] direct image URLs from evals: {len(image_urls)}", flush=True)
    print(f"[scrape] responses_key refs to follow: {len(response_refs)}", flush=True)

    response_list = sorted(response_refs)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, txt in enumerate(pool.map(fetch_text, response_list), 1):
            if txt:
                for m in IMG_RE.findall(txt):
                    image_urls.add(m)
            if i % 500 == 0 or i == len(response_list):
                print(f"[scrape] response {i}/{len(response_list)} total_imgs={len(image_urls)}", flush=True)

    print(f"[scrape] total unique image URLs: {len(image_urls)}", flush=True)

    def download(url: str) -> bool:
        name = url.rsplit("/", 1)[-1]
        target = images_dir / name
        if target.exists() and target.stat().st_size > 0:
            return True
        data = fetch_bytes(url)
        if not data:
            return False
        target.write_bytes(data)
        return True

    image_list = sorted(image_urls)
    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, success in enumerate(pool.map(download, image_list), 1):
            if success:
                ok += 1
            if i % 200 == 0 or i == len(image_list):
                print(f"[scrape] downloaded {ok}/{i}/{len(image_list)}", flush=True)

    print(f"[scrape] DONE: {ok}/{len(image_list)} images in {images_dir}", flush=True)


if __name__ == "__main__":
    main()
