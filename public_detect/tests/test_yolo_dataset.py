from __future__ import annotations

from pathlib import Path

from PIL import Image

from public_detect.elements import ElementSpec
from public_detect.score_api import StarterAnnotation, StarterAsset
from public_detect.yolo_dataset import write_yolo_dataset, xyxy_to_yolo


def test_xyxy_to_yolo_normalizes_center_and_size() -> None:
    assert xyxy_to_yolo((10, 20, 30, 60), width=100, height=100) == (
        0.2,
        0.4,
        0.2,
        0.4,
    )


def test_element_class_order_guard_rejects_wrong_order() -> None:
    spec = ElementSpec(
        element_id="example",
        slug="example",
        objects=("cup", "bottle", "can"),
        max_model_size_mb=30,
    )
    spec.assert_objects_match(["cup", "bottle", "can"])
    try:
        spec.assert_objects_match(["bottle", "cup", "can"])
    except ValueError as exc:
        assert "object order mismatch" in str(exc)
    else:
        raise AssertionError("expected object order mismatch")


def test_write_yolo_dataset_maps_class_ids(tmp_path: Path) -> None:
    spec = ElementSpec(
        element_id="manak0/Detect-beverage-detect",
        slug="beverage",
        objects=("cup", "bottle", "can"),
        max_model_size_mb=30,
    )
    image_dir = tmp_path / "images_in"
    image_dir.mkdir()
    Image.new("RGB", (200, 100), "white").save(image_dir / "asset1.png")
    asset = StarterAsset(
        asset_id="asset1",
        image_url="https://example.test/asset1.png",
        annotation_url=None,
        frame_index=1,
        objects=("cup", "bottle", "can"),
        annotations=(
            StarterAnnotation("bottle", (50, 20, 150, 60)),
            StarterAnnotation("can", (0, 0, 20, 20)),
        ),
    )
    summary = write_yolo_dataset([asset], spec, image_dir, tmp_path / "yolo")
    label = (tmp_path / "yolo" / "labels" / "train" / "asset1.txt").read_text().splitlines()
    assert label[0].startswith("1 ")
    assert label[1].startswith("2 ")
    assert summary["images"] == 1
