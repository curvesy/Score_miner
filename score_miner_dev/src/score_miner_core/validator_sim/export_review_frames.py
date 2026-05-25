from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

import cv2
import supervision as sv

from score_miner_core.benchmark.prediction_summary import ChutePredictResponse
from score_miner_core.validator_sim.pgt_bootstrap import DEFAULT_FRAME_IDS, parse_frame_ids
from score_miner_core.validator_sim.replay_loader import load_replay_response


def export_review_frames(
    *,
    replay_dir: Path,
    video: str,
    output_dir: Path,
    frame_ids: list[int],
    min_confidence: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = _ensure_local_video(video)
    response = ChutePredictResponse.model_validate(load_replay_response(replay_dir))
    if not response.success or response.predictions is None:
        raise ValueError(f"Replay has no successful predictions: {response.error}")

    by_frame = {frame.frame_id: frame for frame in response.predictions.frames}
    requested = set(frame_ids)
    exported: list[dict[str, object]] = []
    for frame_id, frame_image in _iter_selected_frames(video_path, frame_ids):
        prediction = by_frame.get(frame_id)
        if prediction is None:
            continue
        boxes = [box for box in prediction.boxes if box.conf >= min_confidence]
        annotated = frame_image.copy()
        for box in boxes:
            color = _box_color(box.team_id)
            cv2.rectangle(
                annotated,
                (int(box.x1), int(box.y1)),
                (int(box.x2), int(box.y2)),
                color,
                2,
            )
            team = f" t{box.team_id}" if box.team_id is not None else ""
            cv2.putText(
                annotated,
                f"p{team} {box.conf:.2f}",
                (int(box.x1), max(0, int(box.y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        output_path = output_dir / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(output_path), annotated)
        exported.append(
            {
                "frame_id": frame_id,
                "path": str(output_path),
                "boxes": len(boxes),
            }
        )

    report = {
        "replay_dir": str(replay_dir),
        "video": video,
        "output_dir": str(output_dir),
        "requested_frame_ids": sorted(frame_ids),
        "exported": exported,
        "missing_frame_ids": sorted(requested.difference(item["frame_id"] for item in exported)),
        "min_confidence": min_confidence,
    }
    (output_dir / "review_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _ensure_local_video(video: str) -> Path:
    if video.startswith(("http://", "https://")):
        suffix = Path(video.split("?", 1)[0]).suffix or ".mp4"
        tmp = NamedTemporaryFile(prefix="score-miner-review-", suffix=suffix, delete=False)
        request = Request(
            video,
            headers={
                "User-Agent": "score-miner-review/0.1",
                "Accept": "video/mp4,video/*;q=0.9,*/*;q=0.8",
            },
        )
        with tmp:
            with urlopen(request, timeout=300) as response:
                tmp.write(response.read())
        return Path(tmp.name)
    return Path(video)


def _box_color(team_id: int | str | None) -> tuple[int, int, int]:
    if str(team_id) == "1":
        return (255, 80, 80)
    if str(team_id) == "2":
        return (80, 180, 255)
    return (0, 255, 0)


def _iter_selected_frames(video_path: Path, frame_ids: list[int]):
    requested = set(frame_ids)
    if not requested:
        return
    end = max(requested) + 1
    for idx, frame in enumerate(
        sv.get_video_frames_generator(source_path=str(video_path), start=0, end=end)
    ):
        if idx in requested:
            yield idx, frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export replay frames with predicted boxes for manual PGT review.")
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-ids", default=DEFAULT_FRAME_IDS)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_review_frames(
        replay_dir=args.replay_dir,
        video=args.video,
        output_dir=args.output_dir,
        frame_ids=parse_frame_ids(args.frame_ids),
        min_confidence=args.min_confidence,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
