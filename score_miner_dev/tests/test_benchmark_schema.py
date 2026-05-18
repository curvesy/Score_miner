from score_miner_core.benchmark.schema_check import validate_frame_results
from score_miner_core.runtime.miner_runtime import TVFrameResult


def test_schema_check_accepts_valid_runtime_frame() -> None:
    result = TVFrameResult(frame_id=0, boxes=[], keypoints=[(0, 0), (1, 1)])

    check = validate_frame_results([result], expected_frame_count=1, n_keypoints=2)

    assert check.valid
    assert check.errors == []


def test_schema_check_rejects_wrong_keypoint_count() -> None:
    result = TVFrameResult(frame_id=0, boxes=[], keypoints=[(0, 0)])

    check = validate_frame_results([result], expected_frame_count=1, n_keypoints=2)

    assert not check.valid
    assert "expected 2 keypoints" in check.errors[0]
