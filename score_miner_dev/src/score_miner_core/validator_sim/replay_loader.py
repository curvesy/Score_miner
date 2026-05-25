from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_replay_response(replay_dir: Path) -> dict[str, Any]:
    response_path = replay_dir / "response.json"
    if not response_path.is_file():
        raise FileNotFoundError(f"Missing replay response: {response_path}")
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Replay response must be a JSON object: {response_path}")
    return payload


def load_predictions_from_response(response: dict[str, Any]) -> dict[str, Any] | None:
    predictions = response.get("predictions")
    if predictions is None:
        return None
    if not isinstance(predictions, dict):
        raise ValueError("response.predictions must be an object or null")
    frames = predictions.get("frames")
    if frames is None:
        raise ValueError("response.predictions.frames missing")
    if not isinstance(frames, list):
        raise ValueError("response.predictions.frames must be a list")
    return predictions


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

