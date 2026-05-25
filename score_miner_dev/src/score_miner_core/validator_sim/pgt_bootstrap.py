from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from score_miner_core.benchmark.prediction_summary import ChutePredictResponse
from score_miner_core.validator_sim.replay_loader import load_json_object, load_replay_response


DEFAULT_FRAME_IDS = "0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,749"


def build_pgt_bootstrap(
    *,
    replay_dir: Path,
    output_path: Path,
    frame_ids: list[int],
    min_confidence: float,
    max_boxes_per_frame: int | None,
    label: str,
) -> dict[str, Any]:
    response = ChutePredictResponse.model_validate(load_replay_response(replay_dir))
    if not response.success or response.predictions is None:
        raise ValueError(f"Replay has no successful predictions: {response.error}")

    requested = set(frame_ids)
    annotations: list[dict[str, Any]] = []
    selected_frames: list[int] = []
    for frame in response.predictions.frames:
        if frame.frame_id not in requested:
            continue
        selected_frames.append(frame.frame_id)
        boxes = [box for box in frame.boxes if box.conf >= min_confidence]
        boxes = sorted(boxes, key=lambda box: box.conf, reverse=True)
        if max_boxes_per_frame is not None:
            boxes = boxes[:max_boxes_per_frame]
        for box in boxes:
            annotations.append(
                {
                    "frame_id": frame.frame_id,
                    "bbox": [box.x1, box.y1, box.x2, box.y2],
                    "label": label,
                    "score": round(float(box.conf), 6),
                    "review_status": "needs_manual_review",
                    "source": "model_prediction_bootstrap",
                }
            )

    missing_requested_frames = sorted(requested.difference(selected_frames))
    payload = {
        "video_name": replay_dir.name,
        "source_replay": str(replay_dir),
        "review_required": True,
        "warning": (
            "This file is bootstrapped from model predictions. Correct boxes manually "
            "before using it as ground truth for score decisions."
        ),
        "selection": {
            "frame_ids": sorted(frame_ids),
            "missing_requested_frames": missing_requested_frames,
            "min_confidence": min_confidence,
            "max_boxes_per_frame": max_boxes_per_frame,
            "label": label,
        },
        "annotations": annotations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def parse_frame_ids(value: str) -> list[int]:
    frame_ids = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        frame_ids.append(int(raw))
    if not frame_ids:
        raise ValueError("At least one frame id is required")
    return sorted(set(frame_ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap a review-required PGT JSON from replay predictions."
    )
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-ids", default=DEFAULT_FRAME_IDS)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-boxes-per-frame", type=int, default=25)
    parser.add_argument("--label", default="player")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_pgt_bootstrap(
        replay_dir=args.replay_dir,
        output_path=args.output,
        frame_ids=parse_frame_ids(args.frame_ids),
        min_confidence=args.min_confidence,
        max_boxes_per_frame=args.max_boxes_per_frame,
        label=args.label,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "annotations": len(payload["annotations"]),
                "frames": len(set(item["frame_id"] for item in payload["annotations"])),
                "review_required": payload["review_required"],
                "missing_requested_frames": payload["selection"]["missing_requested_frames"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

