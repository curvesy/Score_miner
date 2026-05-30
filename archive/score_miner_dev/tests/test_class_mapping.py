import pytest

from score_miner_core.runtime.class_mapping import ClassMapping, ClassMappingError


def test_class_mapping_resolves_manifest_ids() -> None:
    mapping = ClassMapping.from_manifest(["player", "goalkeeper", "referee", "ball"])

    assert mapping.cls_id_for_detector_label("player") == 0
    assert mapping.cls_id_for_detector_label("ball") == 3
    assert mapping.cls_id_for_detector_label("unknown") is None


def test_class_mapping_rejects_missing_targets() -> None:
    with pytest.raises(ClassMappingError):
        ClassMapping.from_manifest(
            ["player", "ball"],
            detector_to_manifest={"goalkeeper": "goalkeeper"},
        )
