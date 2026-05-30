import numpy as np

from score_miner_core.runtime.miner_runtime import BoundingBox
from score_miner_core.runtime.team_color import TeamColorConfig, TeamColorMemory, assign_team_ids_by_color


def test_assign_team_ids_by_torso_color() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    boxes = [
        BoundingBox(x1=5, y1=10, x2=25, y2=70, cls_id=0, conf=0.9),
        BoundingBox(x1=30, y1=10, x2=50, y2=70, cls_id=0, conf=0.9),
        BoundingBox(x1=65, y1=10, x2=85, y2=70, cls_id=0, conf=0.9),
        BoundingBox(x1=90, y1=10, x2=110, y2=70, cls_id=0, conf=0.9),
    ]

    for box in boxes[:2]:
        image[box.y1 : box.y2, box.x1 : box.x2] = [220, 20, 20]
    for box in boxes[2:]:
        image[box.y1 : box.y2, box.x1 : box.x2] = [20, 40, 220]

    assigned = assign_team_ids_by_color(
        image,
        boxes,
        TeamColorConfig(min_players_per_frame=2, min_crop_pixels=4),
    )

    assert assigned[0].team_id == assigned[1].team_id
    assert assigned[2].team_id == assigned[3].team_id
    assert assigned[0].team_id != assigned[2].team_id
    assert {box.team_id for box in assigned} == {1, 2}


def test_team_color_ignores_non_player_boxes() -> None:
    image = np.full((40, 40, 3), [200, 0, 0], dtype=np.uint8)
    boxes = [
        BoundingBox(x1=0, y1=0, x2=20, y2=30, cls_id=0, conf=0.9),
        BoundingBox(x1=20, y1=0, x2=39, y2=30, cls_id=1, conf=0.9),
    ]

    assigned = assign_team_ids_by_color(
        image,
        boxes,
        TeamColorConfig(min_players_per_frame=2, min_crop_pixels=4),
    )

    assert assigned[0].team_id is None
    assert assigned[1].team_id is None


def test_team_color_memory_stabilizes_track_votes() -> None:
    memory = TeamColorMemory(TeamColorConfig(track_memory_min_votes=2))
    boxes = [BoundingBox(x1=0, y1=0, x2=10, y2=10, cls_id=0, conf=0.9, team_id=1, track_id=42)]

    memory.stabilize(boxes)
    memory.stabilize(boxes)
    boxes[0].team_id = 2
    memory.stabilize(boxes)

    assert boxes[0].team_id == 1
