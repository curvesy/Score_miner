from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import supervision as sv

from score_miner_core.benchmark.latency import latency_summary
from score_miner_core.benchmark.schema_check import SchemaCheckResult, validate_frame_results
from score_miner_core.detector.detector_router import create_detector
from score_miner_core.runtime.memory_budget import MemoryBudget
from score_miner_core.runtime.miner_runtime import MinerRuntime


DEFAULT_VIDEO = "turbovision/tests/test_data/videos/example_football.mp4"
DEFAULT_NOTES = "score_miner_dev/notes/benchmark_results.md"


class VideoSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    sampled_frames: int
    stride: int
    start: int
    end: int | None = None


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video: VideoSummary
    batch_size: int = Field(gt=0)
    n_keypoints: int = Field(gt=0)
    batches: int
    frames_processed: int
    boxes_total: int
    boxes_per_frame_mean: float
    boxes_per_frame_max: int
    latency: dict[str, float]
    memory_before: dict[str, Any]
    memory_after_load: dict[str, Any]
    memory_after_predict: dict[str, Any]
    schema_check: SchemaCheckResult
    detector: str
    threshold: float
    optimize_for_inference: bool


def load_video_frames(
    video_path: Path,
    *,
    max_frames: int,
    stride: int,
    start: int = 0,
) -> tuple[list[np.ndarray], VideoSummary]:
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    end = None
    if max_frames > 0:
        end = start + max_frames * stride

    frames = list(
        sv.get_video_frames_generator(
            source_path=str(video_path),
            stride=stride,
            start=start,
            end=end,
        )
    )
    if max_frames > 0:
        frames = frames[:max_frames]

    summary = VideoSummary(
        path=str(video_path),
        width=int(video_info.width),
        height=int(video_info.height),
        fps=float(video_info.fps),
        total_frames=int(video_info.total_frames),
        sampled_frames=len(frames),
        stride=stride,
        start=start,
        end=end,
    )
    return frames, summary


def run_benchmark(
    *,
    video_path: Path,
    max_frames: int,
    batch_size: int,
    n_keypoints: int,
    stride: int,
    start: int,
    path_hf_repo: Path,
    detector_name: str,
    threshold: float,
    player_cls_id: int,
    ball_cls_id: int | None,
    optimize_for_inference: bool,
    input_color_space: str,
) -> BenchmarkReport:
    memory_budget = MemoryBudget()
    memory_before = memory_budget.status()
    detector = create_detector(
        detector_name,
        threshold=threshold,
        player_cls_id=player_cls_id,
        ball_cls_id=ball_cls_id,
        optimize_for_inference=optimize_for_inference,
    )
    miner = MinerRuntime(path_hf_repo, detector=detector, input_color_space=input_color_space)
    memory_after_load = memory_budget.status()

    frames, video_summary = load_video_frames(
        video_path,
        max_frames=max_frames,
        stride=stride,
        start=start,
    )

    all_results: list[Any] = []
    batch_latencies_ms: list[float] = []
    for offset, batch in _iter_batches(frames, batch_size=batch_size, start_frame_id=start):
        started_ns = perf_counter_ns()
        batch_results = miner.predict_batch(batch, offset=offset, n_keypoints=n_keypoints)
        elapsed_ns = perf_counter_ns() - started_ns
        batch_latencies_ms.append(elapsed_ns / 1_000_000)
        all_results.extend(batch_results)

    memory_after_predict = memory_budget.status()
    schema_check = validate_frame_results(
        all_results,
        expected_frame_count=len(frames),
        n_keypoints=n_keypoints,
    )
    box_counts = [
        len(result.boxes if not hasattr(result, "model_dump") else result.model_dump()["boxes"])
        for result in all_results
    ]

    return BenchmarkReport(
        video=video_summary,
        batch_size=batch_size,
        n_keypoints=n_keypoints,
        batches=len(batch_latencies_ms),
        frames_processed=len(frames),
        boxes_total=sum(box_counts),
        boxes_per_frame_mean=round(sum(box_counts) / len(box_counts), 4) if box_counts else 0.0,
        boxes_per_frame_max=max(box_counts) if box_counts else 0,
        latency=latency_summary(batch_latencies_ms),
        memory_before=memory_before,
        memory_after_load=memory_after_load,
        memory_after_predict=memory_after_predict,
        schema_check=schema_check,
        detector=detector_name,
        threshold=threshold,
        optimize_for_inference=optimize_for_inference,
    )


def write_markdown_report(report: BenchmarkReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = report.model_dump(mode="json")
    lines = [
        "# Benchmark Results",
        "",
        "```json",
        json.dumps(data, indent=2, sort_keys=True),
        "```",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _iter_batches(
    frames: list[np.ndarray],
    *,
    batch_size: int,
    start_frame_id: int,
):
    for idx in range(0, len(frames), batch_size):
        yield start_frame_id + idx, frames[idx : idx + batch_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local miner benchmark on a video.")
    parser.add_argument("--video", type=Path, default=Path(DEFAULT_VIDEO))
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-keypoints", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--path-hf-repo", type=Path, default=Path("score_miner"))
    parser.add_argument("--detector", default="empty")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--player-cls-id", type=int, default=0)
    parser.add_argument("--ball-cls-id", type=int, default=None)
    parser.add_argument(
        "--input-color-space",
        choices=("rgb", "bgr"),
        default="rgb",
        help="Color order produced by this local frame loader. Chutes runtime defaults to bgr.",
    )
    parser.add_argument(
        "--no-optimize-for-inference",
        action="store_true",
        help="Disable detector-specific inference optimization.",
    )
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_NOTES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_benchmark(
        video_path=args.video,
        max_frames=args.frames,
        batch_size=args.batch_size,
        n_keypoints=args.n_keypoints,
        stride=args.stride,
        start=args.start,
        path_hf_repo=args.path_hf_repo,
        detector_name=args.detector,
        threshold=args.threshold,
        player_cls_id=args.player_cls_id,
        ball_cls_id=args.ball_cls_id,
        optimize_for_inference=not args.no_optimize_for_inference,
        input_color_space=args.input_color_space,
    )
    write_markdown_report(report, args.output)
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
