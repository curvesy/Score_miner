from score_miner_core.runtime.team_color import TeamColorConfig, TeamColorMemory


class FakeBox:
    def __init__(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        *,
        cls_id: int,
        track_id: int | None = None,
        team_id: int | str | None = None,
    ) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.cls_id = cls_id
        self.track_id = track_id
        self.team_id = team_id


def test_track_with_history_keeps_team_id_when_frame_fails_to_cluster() -> None:
    memory = TeamColorMemory(TeamColorConfig(track_memory_min_votes=1))
    for _ in range(3):
        memory.stabilize([FakeBox(0, 0, 10, 10, cls_id=0, track_id=7, team_id=1)])

    boxes = [FakeBox(0, 0, 10, 10, cls_id=0, track_id=7, team_id=None)]

    out = memory.stabilize(boxes)

    assert out[0].team_id == 1


def test_untracked_player_uses_nearest_same_frame_team_assignment() -> None:
    memory = TeamColorMemory(TeamColorConfig())
    boxes = [
        FakeBox(0, 0, 10, 10, cls_id=0, team_id=1),
        FakeBox(100, 0, 110, 10, cls_id=0, team_id=2),
        FakeBox(2, 0, 12, 10, cls_id=0, team_id=None),
    ]

    out = memory.stabilize(boxes)

    assert out[2].team_id == 1


def test_untracked_player_stays_unassigned_without_same_frame_evidence() -> None:
    memory = TeamColorMemory(TeamColorConfig())
    boxes = [FakeBox(2, 0, 12, 10, cls_id=0, team_id=None)]

    out = memory.stabilize(boxes)

    assert out[0].team_id is None
