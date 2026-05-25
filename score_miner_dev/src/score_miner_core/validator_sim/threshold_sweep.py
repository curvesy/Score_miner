from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from score_miner_core.benchmark.prediction_summary import (
    ChutePredictResponse,
    summarize_chute_response,
)
from score_miner_core.validator_sim.replay_loader import load_replay_response


class ThresholdRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold: float
    frames: int
    boxes_total: int
    empty_frames: int
    boxes_per_frame_mean: float | None
    boxes_per_frame_p50: float | None
    boxes_per_frame_p95: float | None
    confidence_min: float | None
    confidence_mean: float | None
    confidence_p50: float | None
    confidence_p95: float | None


class ThresholdSweepReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replay_dir: str
    expected_frame_count: int | None
    n_keypoints: int
    rows: list[ThresholdRow]
    recommended_for_review: ThresholdRow | None


def run_threshold_sweep(
    *,
    replay_dir: Path,
    thresholds: list[float],
    expected_frame_count: int | None,
    n_keypoints: int,
    target_boxes_per_frame_min: float,
    target_boxes_per_frame_max: float,
) -> ThresholdSweepReport:
    response = ChutePredictResponse.model_validate(load_replay_response(replay_dir))
    if not response.success or response.predictions is None:
        raise ValueError(f"Replay has no successful predictions: {response.error}")

    rows: list[ThresholdRow] = []
    for threshold in thresholds:
        filtered_response = _filter_response(response, threshold=threshold)
        summary = summarize_chute_response(
            filtered_response.model_dump(mode="json"),
            n_keypoints=n_keypoints,
            expected_frame_count=expected_frame_count,
        )
        rows.append(
            ThresholdRow(
                threshold=round(float(threshold), 4),
                frames=summary.frames_returned,
                boxes_total=summary.boxes_total,
                empty_frames=summary.empty_frames,
                boxes_per_frame_mean=summary.boxes_per_frame.mean,
                boxes_per_frame_p50=summary.boxes_per_frame.p50,
                boxes_per_frame_p95=summary.boxes_per_frame.p95,
                confidence_min=summary.confidence.min,
                confidence_mean=summary.confidence.mean,
                confidence_p50=summary.confidence.p50,
                confidence_p95=summary.confidence.p95,
            )
        )

    recommended = _recommend_threshold(
        rows,
        target_boxes_per_frame_min=target_boxes_per_frame_min,
        target_boxes_per_frame_max=target_boxes_per_frame_max,
    )
    return ThresholdSweepReport(
        replay_dir=str(replay_dir),
        expected_frame_count=expected_frame_count,
        n_keypoints=n_keypoints,
        rows=rows,
        recommended_for_review=recommended,
    )


def _filter_response(response: ChutePredictResponse, *, threshold: float) -> ChutePredictResponse:
    assert response.predictions is not None
    payload = response.model_dump(mode="json")
    for frame in payload["predictions"]["frames"]:
        frame["boxes"] = [
            box for box in frame.get("boxes", []) if float(box.get("conf", 0.0)) >= threshold
        ]
    return ChutePredictResponse.model_validate(payload)


def _recommend_threshold(
    rows: list[ThresholdRow],
    *,
    target_boxes_per_frame_min: float,
    target_boxes_per_frame_max: float,
) -> ThresholdRow | None:
    candidates = [
        row
        for row in rows
        if row.boxes_per_frame_mean is not None
        and target_boxes_per_frame_min <= row.boxes_per_frame_mean <= target_boxes_per_frame_max
        and row.empty_frames == 0
    ]
    if candidates:
        return max(candidates, key=lambda row: row.threshold)
    nonempty = [row for row in rows if row.empty_frames == 0]
    if nonempty:
        return min(
            nonempty,
            key=lambda row: abs((row.boxes_per_frame_mean or 0.0) - target_boxes_per_frame_max),
        )
    return rows[0] if rows else None


def parse_thresholds(value: str) -> list[float]:
    thresholds = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        thresholds.append(float(raw))
    if not thresholds:
        raise ValueError("At least one threshold is required")
    return sorted(set(thresholds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds over saved replay predictions.")
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        default="0.35,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
    )
    parser.add_argument("--expected-frame-count", type=int, default=None)
    parser.add_argument("--n-keypoints", type=int, default=32)
    parser.add_argument("--target-boxes-per-frame-min", type=float, default=8.0)
    parser.add_argument("--target-boxes-per-frame-max", type=float, default=14.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_threshold_sweep(
        replay_dir=args.replay_dir,
        thresholds=parse_thresholds(args.thresholds),
        expected_frame_count=args.expected_frame_count,
        n_keypoints=args.n_keypoints,
        target_boxes_per_frame_min=args.target_boxes_per_frame_min,
        target_boxes_per_frame_max=args.target_boxes_per_frame_max,
    )
    payload = report.model_dump(mode="json")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

