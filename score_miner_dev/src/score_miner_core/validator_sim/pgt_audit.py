from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from score_miner_core.validator_sim.replay_loader import load_json_object


class PGTAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pgt_path: str
    score_ready: bool
    review_required: bool
    annotations: int
    frames: int
    label_counts: dict[str, int]
    review_status_counts: dict[str, int]
    boxes_per_frame: dict[str, float | int | None]
    invalid_annotations: list[str]
    warnings: list[str]


def audit_pgt(path: Path, *, allow_unreviewed: bool = False) -> PGTAuditReport:
    payload = load_json_object(path)
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("PGT JSON must contain annotations: [...]")

    invalid: list[str] = []
    warnings: list[str] = []
    label_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    frame_counts: dict[int, int] = defaultdict(int)

    for idx, item in enumerate(annotations):
        if not isinstance(item, dict):
            invalid.append(f"annotations[{idx}] is not an object")
            continue
        frame_id = item.get("frame_id", item.get("frame_idx", item.get("frame_number")))
        if isinstance(frame_id, str) and frame_id.isdigit():
            frame_id = int(frame_id)
        if not isinstance(frame_id, int) or frame_id < 0:
            invalid.append(f"annotations[{idx}] invalid frame_id={frame_id!r}")
            continue
        raw_bbox = item.get("bbox", item.get("bbox_2d"))
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            invalid.append(f"annotations[{idx}] invalid bbox={raw_bbox!r}")
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in raw_bbox]
        except (TypeError, ValueError):
            invalid.append(f"annotations[{idx}] non-integer bbox={raw_bbox!r}")
            continue
        if x2 < x1 or y2 < y1:
            invalid.append(f"annotations[{idx}] invalid bbox geometry={raw_bbox!r}")
            continue

        label = str(item.get("label", item.get("class", ""))).strip()
        if not label:
            invalid.append(f"annotations[{idx}] missing label")
            continue
        label_counts[label] += 1
        review_status = str(item.get("review_status", "missing")).strip() or "missing"
        review_counts[review_status] += 1
        frame_counts[frame_id] += 1

    review_required = bool(payload.get("review_required")) or any(
        status in {"needs_manual_review", "missing"} for status in review_counts
    )
    if review_required and not allow_unreviewed:
        warnings.append("PGT still contains review-required or missing review_status labels.")
    if len(label_counts) == 1 and "player" in label_counts:
        warnings.append("PGT only contains player labels; role/team/ball scoring remains incomplete.")

    box_counts = list(frame_counts.values())
    score_ready = not invalid and (allow_unreviewed or not review_required)
    return PGTAuditReport(
        pgt_path=str(path),
        score_ready=score_ready,
        review_required=review_required,
        annotations=len(annotations),
        frames=len(frame_counts),
        label_counts=dict(sorted(label_counts.items())),
        review_status_counts=dict(sorted(review_counts.items())),
        boxes_per_frame=_box_stats(box_counts),
        invalid_annotations=invalid,
        warnings=warnings,
    )


def _box_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PGT JSON before validator_sim scoring.")
    parser.add_argument("--pgt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-unreviewed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_pgt(args.pgt, allow_unreviewed=args.allow_unreviewed)
    payload = report.model_dump(mode="json")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

