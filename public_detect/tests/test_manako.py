from pathlib import Path

from PIL import Image

from public_detect.manako import (
    load_manako_frames_from_index,
    parse_manako_frames,
    render_prediction_overlay,
)


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


def test_load_manako_frames_from_index_follows_yaml_refs(monkeypatch) -> None:
    payload = {
        "frames": [{"frame_id": 0, "url": "https://x/challenge-objects/c2/images/a.png"}],
        "predictions": {"frames": [{"frame_id": 0, "boxes": []}]},
    }

    def fake_fetch(url: str):
        assert url == "https://turbo.scoredata.me/manako/abc.yaml"
        return payload

    monkeypatch.setattr("public_detect.manako.fetch_manako_payload", fake_fetch)

    frames = load_manako_frames_from_index(
        index_data={"items": ["abc.yaml"]},
        index_url="https://turbo.scoredata.me/manako/index.json",
    )

    assert len(frames) == 1
    assert frames[0].challenge_id == "c2"


def test_load_manako_frames_from_index_joins_manako_refs_at_origin(monkeypatch) -> None:
    payload = {
        "frames": [{"frame_id": 0, "url": "https://x/challenge-objects/c4/images/a.png"}],
        "predictions": {"frames": [{"frame_id": 0, "boxes": []}]},
    }

    def fake_fetch(url: str):
        assert url == "https://turbo.scoredata.me/manako/path/file.json"
        return payload

    monkeypatch.setattr("public_detect.manako.fetch_manako_payload", fake_fetch)

    frames = load_manako_frames_from_index(
        index_data={"items": ["manako/path/file.json"]},
        index_url="https://turbo.scoredata.me/manako/index.json",
    )

    assert len(frames) == 1


def test_load_manako_frames_from_index_filters_element_refs(monkeypatch) -> None:
    fetched = []

    def fake_fetch(url: str):
        fetched.append(url)
        return {
            "frames": [{"frame_id": 0, "url": "https://x/challenge-objects/c5/images/a.png"}],
            "predictions": {"frames": [{"frame_id": 0, "boxes": []}]},
        }

    monkeypatch.setattr("public_detect.manako.fetch_manako_payload", fake_fetch)

    frames = load_manako_frames_from_index(
        index_data=[
            "manako/manak0_Detect-petrol-station-1-0/hotkey/evaluation/a.json",
            "manako/manak0_Detect-beverage-detect/hotkey/evaluation/b.json",
        ],
        index_url="https://turbo.scoredata.me/manako/index.json",
        element_filters=("Detect-beverage",),
    )

    assert len(frames) == 1
    assert fetched == ["https://turbo.scoredata.me/manako/manak0_Detect-beverage-detect/hotkey/evaluation/b.json"]


def test_load_manako_frames_from_index_follows_responses_key(monkeypatch) -> None:
    def fake_fetch(url: str):
        if "/evaluation/" in url:
            return [
                {
                    "payload": {
                        "telemetry": {
                            "run": {
                                "responses_key": "manako/manak0_Detect-beverage-detect/hotkey/responses/r.json"
                            }
                        }
                    }
                }
            ]
        assert url == "https://turbo.scoredata.me/manako/manak0_Detect-beverage-detect/hotkey/responses/r.json"
        return {
            "frames": [{"frame_id": 0, "url": "https://x/challenge-objects/c7/images/a.png"}],
            "predictions": {"frames": [{"frame_id": 0, "boxes": []}]},
        }

    monkeypatch.setattr("public_detect.manako.fetch_manako_payload", fake_fetch)

    frames = load_manako_frames_from_index(
        index_data=["manako/manak0_Detect-beverage-detect/hotkey/evaluation/e.json"],
        index_url="https://turbo.scoredata.me/manako/index.json",
        element_filters=("Detect-beverage",),
    )

    assert len(frames) == 1
    assert frames[0].challenge_id == "c7"


def test_parse_manako_frames_finds_nested_entries() -> None:
    data = {
        "result": {
            "payload": {
                "frames": [{"frame_id": 0, "url": "https://x/challenge-objects/c6/images/a.png"}],
                "predictions": {"frames": [{"frame_id": 0, "boxes": []}]},
            }
        }
    }

    frames = parse_manako_frames(data)

    assert len(frames) == 1
    assert frames[0].challenge_id == "c6"


def test_parse_manako_frames_dedupes_same_challenge_with_url_variants() -> None:
    data = [
        {"frames": [{"frame_id": 0, "url": "https://x/challenge-objects/c8/images/a.png?miner=1"}]},
        {"frames": [{"frame_id": 0, "url": "https://x/challenge-objects/c8/images/a.png?miner=2"}]},
    ]

    frames = parse_manako_frames(data)

    assert len(frames) == 1


def test_load_manako_frames_from_index_skips_dead_refs(monkeypatch) -> None:
    payload = {
        "frames": [{"frame_id": 0, "url": "https://x/challenge-objects/c3/images/a.png"}],
        "predictions": {"frames": [{"frame_id": 0, "boxes": []}]},
    }

    def fake_fetch(url: str):
        if url.endswith("missing.yaml"):
            raise RuntimeError("404")
        return payload

    monkeypatch.setattr("public_detect.manako.fetch_manako_payload", fake_fetch)

    frames, skipped = load_manako_frames_from_index(
        index_data={"items": ["missing.yaml", "ok.yaml"]},
        index_url="https://turbo.scoredata.me/manako/index.json",
        return_skipped=True,
    )

    assert len(frames) == 1
    assert len(skipped) == 1


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
