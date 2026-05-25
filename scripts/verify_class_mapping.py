#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import yaml


DEFAULT_ELEMENT_ID = "PlayerDetect_v1@1.0"
DEFAULT_PUBLIC_INDEX = "https://turbo.scoredata.me/manifest/index.json"
DEFAULT_LOCAL_MANIFEST = "../turbovision/tests/test_data/manifests/example_manifest.yml"


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args)
    if args.list:
        print(json.dumps(summarize_elements(manifest), indent=2, sort_keys=True))
        return

    element = find_element(manifest, args.element_id)
    if element is None:
        element_ids = [str(item.get("id")) for item in manifest.get("elements", []) if item.get("id")]
        raise SystemExit(
            f"Element {args.element_id!r} not found. Available elements: {', '.join(element_ids)}"
        )

    objects = element.get("objects") or []
    pillars = ((element.get("metrics") or {}).get("pillars") or {})
    report = {
        "source": manifest.get("_source"),
        "manifest_window_id": manifest.get("window_id"),
        "manifest_version": manifest.get("version"),
        "element_id": args.element_id,
        "element_weight": element.get("weight"),
        "baseline_theta": element.get("baseline_theta"),
        "latency_p95_ms": element.get("latency_p95_ms"),
        "service_rate_fps": element.get("service_rate_fps"),
        "objects": objects,
        "class_ids": {name: idx for idx, name in enumerate(objects)},
        "pillars": pillars,
        "keypoints_weighted": "keypoints_iou" in pillars,
        "ball_cls_id": class_id_for(objects, "ball"),
        "player_cls_id": class_id_for(objects, "player"),
        "goalkeeper_cls_id": class_id_for(objects, "goalkeeper"),
        "referee_cls_id": class_id_for(objects, "referee"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.strict:
        validate_strict(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify ScoreVision element object order and pillar weights without R2 credentials."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path, help="Local manifest YAML path.")
    source.add_argument("--url", default=None, help="Public manifest index URL.")
    parser.add_argument("--element-id", default=DEFAULT_ELEMENT_ID)
    parser.add_argument("--block", type=int, default=None)
    parser.add_argument("--list", action="store_true", help="List all elements in the selected manifest.")
    parser.add_argument("--strict", action="store_true", help="Fail if PlayerDetect lacks player class.")
    return parser.parse_args()


def load_manifest(args: argparse.Namespace) -> dict:
    if args.manifest is not None:
        path = args.manifest
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit(f"Manifest did not decode to an object: {path}")
        data["_source"] = str(path)
        return data

    index_url = args.url or DEFAULT_PUBLIC_INDEX
    index = fetch_json(index_url)
    urls = manifest_urls_from_index(index_url, index)
    picked = pick_manifest_url(urls, args.block)
    if picked is None:
        raise SystemExit(f"No block-prefixed manifest entries found in {index_url}")
    _block, manifest_url = picked
    data = fetch_yaml(manifest_url)
    if not isinstance(data, dict):
        raise SystemExit(f"Manifest did not decode to an object: {manifest_url}")
    data["_source"] = manifest_url
    return data


def fetch_json(url: str) -> object:
    with urlopen(http_request(url), timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_yaml(url: str) -> object:
    with urlopen(http_request(url), timeout=30) as response:
        return yaml.safe_load(response.read().decode("utf-8"))


def http_request(url: str) -> Request:
    return Request(
        url,
        headers={
            "Accept": "application/json,text/yaml,text/plain,*/*",
            "User-Agent": "score-miner-manifest-verifier/0.1",
        },
    )


def manifest_urls_from_index(index_url: str, index: object) -> list[str]:
    keys: list[str] = []
    if isinstance(index, list):
        keys = [item for item in index if isinstance(item, str)]
    elif isinstance(index, dict) and isinstance(index.get("entries"), list):
        keys = [item["path"] for item in index["entries"] if isinstance(item.get("path"), str)]
    return [join_key_to_base(index_url, key) for key in keys]


def join_key_to_base(index_url: str, key_or_url: str) -> str:
    key = key_or_url.strip()
    if key.startswith(("http://", "https://")):
        return key
    parsed = urlparse(index_url)
    bucket_base = f"{parsed.scheme}://{parsed.netloc}/"
    if key.startswith("/"):
        return bucket_base + key.lstrip("/")
    if key.startswith(("manifest/", "manako/")):
        return bucket_base + key
    return urljoin(index_url.rsplit("/", 1)[0] + "/", key)


def pick_manifest_url(urls: list[str], block: int | None) -> tuple[int, str] | None:
    pairs: list[tuple[int, str]] = []
    for url in urls:
        name = Path(urlparse(url).path).name
        try:
            pairs.append((int(name.split("-", 1)[0]), url))
        except Exception:
            continue
    if not pairs:
        return None
    pairs.sort(key=lambda pair: pair[0])
    if block is None:
        return pairs[-1]
    eligible = [pair for pair in pairs if pair[0] <= block]
    return eligible[-1] if eligible else pairs[-1]


def find_element(manifest: dict, element_id: str) -> dict | None:
    for element in manifest.get("elements") or []:
        if isinstance(element, dict) and element.get("id") == element_id:
            return element
    return None


def summarize_elements(manifest: dict) -> dict:
    items = []
    for element in manifest.get("elements") or []:
        if not isinstance(element, dict):
            continue
        objects = element.get("objects") or []
        pillars = ((element.get("metrics") or {}).get("pillars") or {})
        items.append(
            {
                "id": element.get("id"),
                "track": element.get("track"),
                "weight": element.get("weight"),
                "baseline_theta": element.get("baseline_theta"),
                "objects": objects,
                "class_ids": {name: idx for idx, name in enumerate(objects)},
                "pillars": pillars,
                "latency_p95_ms": element.get("latency_p95_ms"),
                "service_rate_fps": element.get("service_rate_fps"),
                "ground_truth": element.get("ground_truth", False),
                "challenge_type_version": element.get("challenge_type_version"),
            }
        )
    return {
        "source": manifest.get("_source"),
        "manifest_window_id": manifest.get("window_id"),
        "manifest_version": manifest.get("version"),
        "elements": items,
    }


def class_id_for(objects: list[str], label: str) -> int | None:
    normalized = [item.strip().lower().replace("_", " ") for item in objects]
    aliases = {
        "player": {"player", "person"},
        "ball": {"ball", "sports ball", "soccer ball", "football"},
        "goalkeeper": {"goalkeeper", "keeper", "goalie"},
        "referee": {"referee", "ref"},
    }
    for idx, name in enumerate(normalized):
        if name in aliases.get(label, {label}):
            return idx
    return None


def validate_strict(report: dict) -> None:
    if str(report["element_id"]).startswith("PlayerDetect") and report["player_cls_id"] is None:
        raise SystemExit("PlayerDetect element has no player/person class in objects.")


if __name__ == "__main__":
    main()
