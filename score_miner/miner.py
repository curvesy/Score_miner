from pathlib import Path

from score_miner_core.runtime.miner_runtime import MinerRuntime


class Miner:
    def __init__(self, path_hf_repo: Path) -> None:
        self.runtime = MinerRuntime(path_hf_repo)

    def __repr__(self) -> str:
        return repr(self.runtime)

    def predict_batch(self, batch_images, offset: int, n_keypoints: int):
        return self.runtime.predict_batch(batch_images, offset, n_keypoints)
