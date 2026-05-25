from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from score_miner_core.validator_sim.replay_loader import write_json


class GroundTruthFetchReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_url: str
    challenge_id: str
    output: str
    raw_output: str | None = None
    http_status: int
    attempts: int
    annotations: int
    converted: bool


def fetch_ground_truth_payload(
    *,
    api_url: str,
    challenge_id: str,
    auth_token: str,
    timeout_seconds: float = 30.0,
    max_attempts: int = 3,
    retry_sleep_seconds: float = 2.0,
) -> tuple[int, int, dict[str, Any]]:
    if not api_url:
        raise ValueError("api_url is required")
    if not challenge_id:
        raise ValueError("challenge_id is required")
    if not auth_token:
        raise ValueError("auth_token is required")

    url = f"{api_url.rstrip('/')}/api/private-track/ground-truth/{challenge_id}"
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        request = Request(
            url,
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError("Ground truth API returned non-object JSON")
                return int(response.status), attempt, data
        except HTTPError as exc:
            if exc.code in {401, 403, 404}:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Ground truth API returned {exc.code}: {body[:500]}") from exc
            last_error = exc
        except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
        if attempt < max_attempts:
            sleep(retry_sleep_seconds * attempt)
    raise RuntimeError(f"Failed to fetch ground truth after {max_attempts} attempts: {last_error}")


def convert_ground_truth_payload(
    payload: dict[str, Any],
    *,
    video_name: str,
    source: str,
) -> dict[str, Any]:
    raw_gt = payload.get("ground_truth", payload)
    if isinstance(raw_gt, dict) and isinstance(raw_gt.get("annotations"), list):
        annotations = raw_gt["annotations"]
    elif isinstance(raw_gt, list):
        annotations = raw_gt
    else:
        raise ValueError("Ground truth payload must contain ground_truth list or annotations list")

    converted: list[dict[str, Any]] = []
    for idx, item in enumerate(annotations):
        if not isinstance(item, dict):
            continue
        frame_id = item.get("frame_id", item.get("frame_idx", item.get("frame_number")))
        if isinstance(frame_id, str) and frame_id.isdigit():
            frame_id = int(frame_id)
        if not isinstance(frame_id, int):
            frame_id = 0

        bbox = item.get("bbox", item.get("bbox_2d"))
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        label = item.get("label", item.get("class", "player"))
        converted.append(
            {
                "frame_id": frame_id,
                "bbox": [int(v) for v in bbox],
                "label": str(label),
                "score": float(item.get("score", 1.0)),
                "review_status": "trusted_api",
                "source": source,
            }
        )

    return {
        "video_name": video_name,
        "review_required": False,
        "source": source,
        "annotations": converted,
    }


def fetch_and_save_ground_truth(
    *,
    api_url: str,
    challenge_id: str,
    auth_token: str,
    output: Path,
    raw_output: Path | None,
    video_name: str,
    timeout_seconds: float,
    max_attempts: int,
) -> GroundTruthFetchReport:
    status, attempts, payload = fetch_ground_truth_payload(
        api_url=api_url,
        challenge_id=challenge_id,
        auth_token=auth_token,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )
    if raw_output is not None:
        write_json(raw_output, payload)
    converted = convert_ground_truth_payload(
        payload,
        video_name=video_name,
        source=f"{api_url.rstrip('/')}/api/private-track/ground-truth/{challenge_id}",
    )
    write_json(output, converted)
    return GroundTruthFetchReport(
        api_url=api_url,
        challenge_id=challenge_id,
        output=str(output),
        raw_output=str(raw_output) if raw_output else None,
        http_status=status,
        attempts=attempts,
        annotations=len(converted["annotations"]),
        converted=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch private-track ground truth and convert to validator_sim PGT JSON.")
    parser.add_argument("--api-url", default=os.getenv("SCORE_GT_API_URL", ""))
    parser.add_argument("--challenge-id", default=os.getenv("SCORE_GT_CHALLENGE_ID", ""))
    parser.add_argument("--auth-token", default=os.getenv("SCORE_GT_AUTH_TOKEN", ""))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, default=None)
    parser.add_argument("--video-name", default="api_ground_truth")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = fetch_and_save_ground_truth(
        api_url=args.api_url,
        challenge_id=args.challenge_id,
        auth_token=args.auth_token,
        output=args.output,
        raw_output=args.raw_output,
        video_name=args.video_name,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

