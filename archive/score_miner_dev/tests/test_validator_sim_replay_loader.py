import json

from score_miner_core.validator_sim.replay_loader import (
    load_predictions_from_response,
    load_replay_response,
)


def test_load_replay_response_reads_json_object(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    (replay_dir / "response.json").write_text(
        json.dumps({"success": True, "predictions": {"frames": []}, "error": None}),
        encoding="utf-8",
    )

    response = load_replay_response(replay_dir)

    assert response["success"] is True


def test_load_predictions_from_response_requires_frames() -> None:
    predictions = load_predictions_from_response(
        {"success": True, "predictions": {"frames": [{"frame_id": 0}]}}
    )

    assert predictions == {"frames": [{"frame_id": 0}]}

