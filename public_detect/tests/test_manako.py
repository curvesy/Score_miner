from pathlib import Path

from PIL import Image

from public_detect.manako import parse_manako_frames, render_prediction_overlay


def test_parse_manako_frames_extracts_predictions() -> None:
    data = {
        "frames": [{"frame_id": 0, "url": "https://x/challenge-objects/c1/images/a.png"}],
        "predictions": {
            "frames": [
                {
                    "frame_id": 0,
                    "boxes": [
                        {"x1": 1, "y1": 2, "x2": 3, "y2": 4, "cls_id": 2, "conf": 0.7}
                    ],
                }
            ]
        },
    }

    frames = parse_manako_frames(data)

    assert len(frames) == 1
    assert frames[0].challenge_id == "c1"
    assert frames[0].boxes[0]["cls_id"] == 2


def test_render_prediction_overlay_writes_image(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "overlay.jpg"
    Image.new("RGB", (20, 20), "white").save(source)

    render_prediction_overlay(
        image_path=source,
        output_path=target,
        boxes=({"x1": 2, "y1": 2, "x2": 10, "y2": 10, "cls_id": 1, "conf": 0.5},),
    )

    assert target.exists()
