import json
from pathlib import Path

from PIL import Image

from public_detect.ingest import ingest_coco_detection, load_coco_ingest_config


def test_ingest_coco_detection_maps_classes_and_hard_negatives(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    element = project / "beverage.yaml"
    element.write_text(
        "\n".join(
            [
                "element_id: manak0/Detect-beverage-detect",
                "slug: beverage",
                "max_model_size_mb: 30",
                "objects:",
                "  - cup",
                "  - bottle",
                "  - can",
                "",
            ]
        )
    )
    config_path = project / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "source_id: test_source",
                "source_type: taco_contextual_trash",
                "element_config: beverage.yaml",
                "review_status: needs_review",
                "class_map:",
                "  cup:",
                "    - paper cup",
                "  bottle:",
                "    - plastic bottle",
                "  can:",
                "    - drink can",
                "hard_negative_categories:",
                "  - jar",
                "filters:",
                "  include_images_with_mapped_labels: true",
                "  include_hard_negative_only_images: true",
                "",
            ]
        )
    )
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (100, 100), "white").save(images / "a.jpg")
    Image.new("RGB", (100, 100), "white").save(images / "b.jpg")
    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "categories": [
            {"id": 10, "name": "paper cup"},
            {"id": 11, "name": "jar"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 10, "bbox": [10, 20, 30, 40]},
            {"id": 2, "image_id": 2, "category_id": 11, "bbox": [1, 2, 3, 4]},
        ],
    }
    coco_path = tmp_path / "coco.json"
    coco_path.write_text(json.dumps(coco))

    config = load_coco_ingest_config(config_path, project)
    manifest = ingest_coco_detection(
        coco_json=coco_path,
        image_root=images,
        output_dir=tmp_path / "out",
        config=config,
    )

    assert manifest["images"] == 2
    assert manifest["boxes"] == 1
    assert (tmp_path / "out/data.yaml").exists()
    labels = sorted((tmp_path / "out/labels/train").glob("*.txt"))
    assert labels[0].read_text().startswith("0 ")
    assert labels[1].read_text() == ""
