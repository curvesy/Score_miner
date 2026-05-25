from score_miner_core.runtime.postprocess import PostprocessConfig


def test_postprocess_config_defaults_match_runtime_decision() -> None:
    config = PostprocessConfig()

    assert config.confidence_threshold == 0.75
    assert config.max_boxes_per_frame == 18
    assert config.min_box_area == 0.0

