from score_miner_core.validator_sim.ground_truth_client import convert_ground_truth_payload


def test_convert_ground_truth_payload_accepts_ground_truth_list() -> None:
    payload = {
        "ground_truth": [
            {
                "frame_idx": 5,
                "bbox": [1, 2, 3, 4],
                "class": "player",
            }
        ]
    }

    converted = convert_ground_truth_payload(
        payload,
        video_name="clip",
        source="test",
    )

    assert converted["review_required"] is False
    assert converted["annotations"] == [
        {
            "frame_id": 5,
            "bbox": [1, 2, 3, 4],
            "label": "player",
            "score": 1.0,
            "review_status": "trusted_api",
            "source": "test",
        }
    ]

