from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import cv2

from score_miner_core.validator_sim.replay_loader import load_json_object


def export_labelstudio_tasks(
    *,
    pgt_path: Path,
    review_manifest_path: Path,
    output_path: Path,
    image_prefix: str,
) -> list[dict]:
    pgt = load_json_object(pgt_path)
    review_manifest = load_json_object(review_manifest_path)
    annotations = pgt.get("annotations")
    exported = review_manifest.get("exported")
    if not isinstance(annotations, list):
        raise ValueError("PGT JSON must contain annotations: [...]")
    if not isinstance(exported, list):
        raise ValueError("Review manifest must contain exported: [...]")

    image_by_frame = {
        int(item["frame_id"]): str(item["path"])
        for item in exported
        if isinstance(item, dict) and "frame_id" in item and "path" in item
    }
    annotations_by_frame: dict[int, list[dict]] = {}
    for item in annotations:
        if not isinstance(item, dict):
            continue
        frame_id = item.get("frame_id", item.get("frame_idx", item.get("frame_number")))
        if isinstance(frame_id, str) and frame_id.isdigit():
            frame_id = int(frame_id)
        if not isinstance(frame_id, int):
            continue
        annotations_by_frame.setdefault(frame_id, []).append(item)

    tasks: list[dict] = []
    for frame_id, image_path in sorted(image_by_frame.items()):
        width, height = _image_size(Path(image_path))
        result = [
            _labelstudio_rectangle(annotation, width=width, height=height)
            for annotation in annotations_by_frame.get(frame_id, [])
        ]
        tasks.append(
            {
                "data": {
                    "image": f"{image_prefix.rstrip('/')}/{Path(image_path).name}",
                    "frame_id": frame_id,
                },
                "predictions": [
                    {
                        "model_version": "score-miner-rfdetr-bootstrap",
                        "score": 1.0,
                        "result": result,
                    }
                ],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2, sort_keys=True), encoding="utf-8")
    return tasks


def _labelstudio_rectangle(annotation: dict, *, width: int, height: int) -> dict:
    x1, y1, x2, y2 = [float(v) for v in annotation["bbox"]]
    return {
        "id": str(uuid4())[:10],
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "value": {
            "x": 100.0 * x1 / width,
            "y": 100.0 * y1 / height,
            "width": 100.0 * max(0.0, x2 - x1) / width,
            "height": 100.0 * max(0.0, y2 - y1) / height,
            "rectanglelabels": [str(annotation.get("label", "player"))],
        },
    }


def _image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export bootstrap PGT as Label Studio prediction tasks.")
    parser.add_argument("--pgt", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--image-prefix",
        default="/data/local-files/?d=rfdetr_m_local_chute_smoke",
        help="Prefix Label Studio should use to access exported review images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = export_labelstudio_tasks(
        pgt_path=args.pgt,
        review_manifest_path=args.review_manifest,
        output_path=args.output,
        image_prefix=args.image_prefix,
    )
    print(json.dumps({"output": str(args.output), "tasks": len(tasks)}, indent=2))


if __name__ == "__main__":
    main()

