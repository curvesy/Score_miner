#!/usr/bin/env python3
"""Load a deploy repo miner.py and run predict_batch on local images."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=project_path)
    parser.add_argument("--images", required=True, type=project_path)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--n-keypoints", type=int, default=32)
    args = parser.parse_args()

    miner_cls = load_miner(args.repo / "miner.py")
    miner = miner_cls(args.repo)
    image_paths = sorted(
        item
        for item in args.images.iterdir()
        if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )[: args.limit]
    images = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"could not read image {path}")
        images.append(image)
    results = miner.predict_batch(images, offset=0, n_keypoints=args.n_keypoints)
    payload = {
        "repo": str(args.repo),
        "images": [str(path) for path in image_paths],
        "miner": repr(miner),
        "results": [
            result.model_dump() if hasattr(result, "model_dump") else result.dict()
            for result in results
        ],
    }
    print(json.dumps(payload, indent=2))


def load_miner(path: Path):
    spec = importlib.util.spec_from_file_location("deploy_miner_smoke", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Miner


if __name__ == "__main__":
    main()
