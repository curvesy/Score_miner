from score_miner_core.benchmark.prediction_summary import summarize_chute_response


def test_summarize_chute_response_counts_boxes_and_frames() -> None:
    response = {
        "success": True,
        "predictions": {
            "frames": [
                {
                    "frame_id": 0,
                    "boxes": [
                        {
                            "x1": 1,
                            "y1": 2,
                            "x2": 10,
                            "y2": 20,
                            "cls_id": 0,
                            "conf": 0.9,
                            "team_id": "home",
                            "cluster_id": 7,
                        }
                    ],
                    "keypoints": [(0, 0), (5, 5)],
                },
                {
                    "frame_id": 1,
                    "boxes": [],
                    "keypoints": [(0, 0), (0, 0)],
                },
            ]
        },
        "error": None,
    }

    summary = summarize_chute_response(response, n_keypoints=2)

    assert summary.success is True
    assert summary.schema_check.valid is True
    assert summary.frames_returned == 2
    assert summary.frame_ids_contiguous is True
    assert summary.boxes_total == 1
    assert summary.empty_frames == 1
    assert summary.class_counts == {"0": 1}
    assert summary.team_id_counts == {"home": 1}
    assert summary.cluster_id_counts == {"7": 1}
    assert summary.valid_keypoints_total == 1


def test_summarize_chute_response_reports_schema_errors() -> None:
    response = {
        "success": True,
        "predictions": {
            "frames": [
                {
                    "frame_id": 0,
                    "boxes": [
                        {
                            "x1": 10,
                            "y1": 2,
                            "x2": 1,
                            "y2": 20,
                            "cls_id": 0,
                            "conf": 0.9,
                        }
                    ],
                    "keypoints": [(0, 0)],
                }
            ]
        },
        "error": None,
    }

    summary = summarize_chute_response(response, n_keypoints=2)

    assert summary.schema_check.valid is False
    assert summary.frames_returned == 0
    assert summary.boxes_total == 0
    assert summary.schema_check.errors

