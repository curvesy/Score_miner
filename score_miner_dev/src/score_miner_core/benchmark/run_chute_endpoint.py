from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from score_miner_core.benchmark.prediction_summary import (
    PredictionSummary,
    summarize_chute_response,
)


DEFAULT_ENDPOINT = "http://localhost:8000/predict"
DEFAULT_VIDEO = "https://scoredata.me/2025_03_14/35ae7a/h1_0f2ca0.mp4"
DEFAULT_OUTPUT_ROOT = "runs/replays"


class EndpointRunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint_url: str
    video_url: str
    output_dir: str
    request_payload: dict[str, Any]
    http_status: int
    elapsed_seconds: float = Field(ge=0)
    summary: PredictionSummary


def run_endpoint_replay(
    *,
    endpoint_url: str,
    video_url: str,
    output_dir: Path,
    timeout_seconds: float,
    n_keypoints: int,
    expected_frame_count: int | None,
    meta: dict[str, Any],
    wrap_data: bool,
) -> EndpointRunReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"url": video_url, "meta": meta}
    request_payload = {"data": payload} if wrap_data else payload

    started = perf_counter()
    http_status, raw_response = _post_json(
        endpoint_url,
        request_payload,
        timeout_seconds=timeout_seconds,
    )
    elapsed_seconds = perf_counter() - started

    summary = summarize_chute_response(
        raw_response,
        n_keypoints=n_keypoints,
        expected_frame_count=expected_frame_count,
    )
    report = EndpointRunReport(
        endpoint_url=endpoint_url,
        video_url=video_url,
        output_dir=str(output_dir),
        request_payload=request_payload,
        http_status=http_status,
        elapsed_seconds=round(elapsed_seconds, 6),
        summary=summary,
    )

    _write_json(output_dir / "request.json", request_payload)
    _write_json(output_dir / "response.json", raw_response)
    _write_json(output_dir / "summary.json", summary.model_dump(mode="json"))
    _write_json(output_dir / "report.json", report.model_dump(mode="json"))
    _write_markdown(output_dir / "report.md", report)
    return report


def _post_json(
    endpoint_url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        endpoint_url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
            status = int(response.status)
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
    except URLError as exc:
        raise RuntimeError(f"Failed to call {endpoint_url}: {exc}") from exc

    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Endpoint returned non-JSON response with status={status}: {response_body[:500]}"
        ) from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"Endpoint returned JSON {type(decoded).__name__}, expected object.")
    return status, decoded


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(path: Path, report: EndpointRunReport) -> None:
    summary = report.summary
    lines = [
        "# Chute Endpoint Replay",
        "",
        f"- endpoint: `{report.endpoint_url}`",
        f"- video: `{report.video_url}`",
        f"- http_status: `{report.http_status}`",
        f"- elapsed_seconds: `{report.elapsed_seconds}`",
        f"- success: `{summary.success}`",
        f"- schema_valid: `{summary.schema_check.valid}`",
        f"- frames_returned: `{summary.frames_returned}`",
        f"- boxes_total: `{summary.boxes_total}`",
        f"- empty_frames: `{summary.empty_frames}`",
        f"- boxes/frame mean: `{summary.boxes_per_frame.mean}`",
        f"- boxes/frame p95: `{summary.boxes_per_frame.p95}`",
        f"- confidence p50: `{summary.confidence.p50}`",
        f"- confidence p95: `{summary.confidence.p95}`",
        "",
        "## Class Counts",
        "",
        "```json",
        json.dumps(summary.class_counts, indent=2, sort_keys=True),
        "```",
        "",
        "## Schema Errors",
        "",
        "```json",
        json.dumps(summary.schema_check.errors, indent=2, sort_keys=True),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call a Chutes /predict endpoint and save replay artifacts.")
    parser.add_argument("--url", default=DEFAULT_ENDPOINT, help="Chutes predict endpoint URL.")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Challenge video URL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Replay output directory. Defaults to runs/replays/<timestamp>.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--n-keypoints", type=int, default=32)
    parser.add_argument("--expected-frame-count", type=int, default=None)
    parser.add_argument("--meta-json", default="{}", help="JSON object added as payload meta.")
    parser.add_argument(
        "--wrap-data",
        action="store_true",
        help="Send {'data': payload}; default sends payload directly, matching current TurboVision curl path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = json.loads(args.meta_json)
    if not isinstance(meta, dict):
        raise SystemExit("--meta-json must decode to a JSON object")
    output_dir = args.output or Path(DEFAULT_OUTPUT_ROOT) / _timestamp_slug()
    report = run_endpoint_replay(
        endpoint_url=args.url,
        video_url=args.video,
        output_dir=output_dir,
        timeout_seconds=args.timeout_seconds,
        n_keypoints=args.n_keypoints,
        expected_frame_count=args.expected_frame_count,
        meta=meta,
        wrap_data=args.wrap_data,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))


def _timestamp_slug() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    main()

