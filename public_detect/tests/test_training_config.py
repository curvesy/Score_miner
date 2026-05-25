from __future__ import annotations

from pathlib import Path

import yaml


def test_training_configs_reference_existing_data() -> None:
    root = Path(__file__).resolve().parents[1]
    configs = sorted((root / "configs/training").glob("*.yaml"))
    assert configs, "expected training configs"
    for path in configs:
        data = yaml.safe_load(path.read_text())
        assert (root / data["element_config"]).exists()
        assert (root / data["data"]).exists()
        assert data["model"].endswith(".pt")
        assert int(data["epochs"]) > 0
        assert int(data["imgsz"]) > 0
        assert int(data["batch"]) > 0
