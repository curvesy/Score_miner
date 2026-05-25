import numpy as np

from score_miner_core.runtime.miner_runtime import BoundingBox
from score_miner_core.runtime.role_cleanup import RoleCleanupConfig, cleanup_roles_by_color
from score_miner_core.runtime.team_color import TeamColorConfig


def test_role_cleanup_is_noop_without_referee_class_id() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    boxes = [BoundingBox(x1=0, y1=0, x2=20, y2=70, cls_id=0, conf=0.99, team_id=1)]

    cleaned = cleanup_roles_by_color(
        image,
        boxes,
        role_config=RoleCleanupConfig(referee_cls_id=None),
        team_config=TeamColorConfig(min_players_per_frame=2),
    )

    assert cleaned[0].cls_id == 0
    assert cleaned[0].team_id == 1


def test_role_cleanup_marks_isolated_referee_color_when_enabled() -> None:
    image = np.zeros((80, 150, 3), dtype=np.uint8)
    boxes = [
        BoundingBox(x1=0, y1=10, x2=20, y2=70, cls_id=0, conf=0.95, team_id=1),
        BoundingBox(x1=25, y1=10, x2=45, y2=70, cls_id=0, conf=0.95, team_id=1),
        BoundingBox(x1=55, y1=10, x2=75, y2=70, cls_id=0, conf=0.95, team_id=2),
        BoundingBox(x1=80, y1=10, x2=100, y2=70, cls_id=0, conf=0.95, team_id=2),
        BoundingBox(x1=120, y1=10, x2=140, y2=70, cls_id=0, conf=0.96, team_id=1),
    ]
    for box in boxes[:2]:
        image[box.y1 : box.y2, box.x1 : box.x2] = [220, 20, 20]
    for box in boxes[2:4]:
        image[box.y1 : box.y2, box.x1 : box.x2] = [20, 40, 220]
    image[boxes[4].y1 : boxes[4].y2, boxes[4].x1 : boxes[4].x2] = [30, 30, 30]

    cleaned = cleanup_roles_by_color(
        image,
        boxes,
        role_config=RoleCleanupConfig(
            referee_cls_id=3,
            referee_min_confidence=0.9,
            referee_min_team_distance=20,
            referee_margin=100,
            referee_max_per_frame=1,
        ),
        team_config=TeamColorConfig(min_players_per_frame=2, min_crop_pixels=4),
    )

    assert cleaned[4].cls_id == 3
    assert cleaned[4].team_id is None
    assert [box.cls_id for box in cleaned[:4]] == [0, 0, 0, 0]
