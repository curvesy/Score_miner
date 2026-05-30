import numpy as np
import supervision as sv

from score_miner_core.detector.class_id_mapper import DetectionClassIdMapper


def test_coco_to_turbovision_mapper_keeps_person_and_ball() -> None:
    detections = sv.Detections(
        xyxy=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=float),
        confidence=np.array([0.9, 0.8, 0.7], dtype=float),
        class_id=np.array([1, 37, 2], dtype=int),
    )
    mapper = DetectionClassIdMapper.coco_to_turbovision(player_cls_id=0, ball_cls_id=3)

    mapped = mapper.apply(detections)

    assert mapped.class_id.tolist() == [0, 3]
    assert mapped.xyxy.shape == (2, 4)
