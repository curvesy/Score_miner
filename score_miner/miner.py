from pathlib import Path
from os import getenv

from score_miner_core.detector.detector_router import create_detector
from score_miner_core.runtime.miner_runtime import MinerRuntime
from score_miner_core.runtime.postprocess import PostprocessConfig
from score_miner_core.runtime.role_cleanup import RoleCleanupConfig
from score_miner_core.runtime.team_color import TeamColorConfig
from score_miner_core.runtime.tracking import TrackingConfig


class Miner:
    def __init__(self, path_hf_repo: Path) -> None:
        detector_name = getenv("SCORE_MINER_DETECTOR", "rfdetr_m")
        threshold = float(getenv("SCORE_MINER_THRESHOLD", "0.75"))
        player_cls_id = int(getenv("SCORE_MINER_PLAYER_CLS_ID", "0"))
        ball_cls_id_raw = getenv("SCORE_MINER_BALL_CLS_ID")
        ball_cls_id = int(ball_cls_id_raw) if ball_cls_id_raw else None
        input_color_space = getenv("SCORE_MINER_INPUT_COLOR_SPACE", "bgr")
        postprocess_config = PostprocessConfig.from_env()
        team_color_config = TeamColorConfig.from_env()
        tracking_config = TrackingConfig.from_env()
        role_cleanup_config = RoleCleanupConfig.from_env()
        detector = create_detector(
            detector_name,
            threshold=threshold,
            player_cls_id=player_cls_id,
            ball_cls_id=ball_cls_id,
        )
        self.runtime = MinerRuntime(
            path_hf_repo,
            detector=detector,
            postprocess_config=postprocess_config,
            team_color_config=team_color_config,
            tracking_config=tracking_config,
            role_cleanup_config=role_cleanup_config,
            input_color_space=input_color_space,
        )

    def __repr__(self) -> str:
        return repr(self.runtime)

    def predict_batch(self, batch_images, offset: int, n_keypoints: int):
        return self.runtime.predict_batch(batch_images, offset, n_keypoints)
