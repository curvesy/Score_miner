from pathlib import Path

from public_detect.deploy_repo import build_deploy_repo, render_miner


def test_render_miner_contains_required_interface() -> None:
    text = render_miner(
        class_names=["cup", "bottle", "can"],
        input_size=768,
        conf_thresholds=[0.1, 0.2, 0.3],
        max_det=20,
        iou_thres=0.4,
        cross_iou_thres=0.7,
        min_side=4,
        min_box_area=16,
        max_aspect_ratio=12,
    )

    assert "class Miner" in text
    assert "def predict_batch" in text
    assert "weights.onnx" in text
    assert "['cup', 'bottle', 'can']" in text


def test_build_deploy_repo_writes_miner_and_size_report(tmp_path: Path) -> None:
    weights = tmp_path / "source.onnx"
    weights.write_bytes(b"onnx")

    report = build_deploy_repo(
        weights=weights,
        output_dir=tmp_path / "repo",
        class_names=["cup", "bottle", "can"],
        input_size=768,
        conf_thresholds=[0.1, 0.2, 0.3],
    )

    assert report["passes"] is True
    assert (tmp_path / "repo" / "miner.py").exists()
    assert (tmp_path / "repo" / "weights.onnx").read_bytes() == b"onnx"
    assert (tmp_path / "repo" / "miner_config.json").exists()
    assert (tmp_path / "repo" / "size_report.json").exists()
