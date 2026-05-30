import json

from score_miner_core.validator_sim.threshold_sweep import run_threshold_sweep


def test_threshold_sweep_filters_boxes_and_recommends_highest_in_range(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    (replay_dir / "response.json").write_text(
        json.dumps(
            {
                "success": True,
                "predictions": {
                    "frames": [
                        {
                            "frame_id": 0,
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
                                    "conf": 0.4,
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

    report = run_threshold_sweep(
        replay_dir=replay_dir,
        thresholds=[0.35, 0.8],
        expected_frame_count=1,
        n_keypoints=0,
        target_boxes_per_frame_min=1,
        target_boxes_per_frame_max=1,
    )

    assert report.rows[0].boxes_total == 2
    assert report.rows[1].boxes_total == 1
    assert report.recommended_for_review is not None
    assert report.recommended_for_review.threshold == 0.8

