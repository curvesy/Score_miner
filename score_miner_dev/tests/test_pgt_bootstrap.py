import json

from score_miner_core.validator_sim.pgt_bootstrap import (
    build_pgt_bootstrap,
    parse_frame_ids,
)


def test_parse_frame_ids_sorts_and_deduplicates() -> None:
    assert parse_frame_ids("5,1,5,2") == [1, 2, 5]


def test_build_pgt_bootstrap_marks_review_required(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    (replay_dir / "response.json").write_text(
        json.dumps(
            {
                "success": True,
                "predictions": {
                    "frames": [
                        {
                            "frame_id": 10,
                            "boxes": [
                                {
                                    "x1": 1,
                                    "y1": 2,
                                    "x2": 3,
                                    "y2": 4,
                                    "cls_id": 0,
                                    "conf": 0.9,
                                    "team_id": None,
                                    "cluster_id": None,
                                },
                                {
                                    "x1": 5,
                                    "y1": 6,
                                    "x2": 7,
                                    "y2": 8,
                                    "cls_id": 0,
                                    "conf": 0.2,
                                    "team_id": None,
                                    "cluster_id": None,
                                },
                            ],
                            "keypoints": [],
                        }
                    ]
                },
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "pgt.json"
    payload = build_pgt_bootstrap(
        replay_dir=replay_dir,
        output_path=output,
        frame_ids=[10],
        min_confidence=0.5,
        max_boxes_per_frame=10,
        label="player",
    )

    assert output.is_file()
    assert payload["review_required"] is True
    assert len(payload["annotations"]) == 1
    assert payload["annotations"][0]["review_status"] == "needs_manual_review"

