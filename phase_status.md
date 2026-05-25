# Phase Status

Date: 2026-05-21

## Current Position

Phase 4 replay tooling is complete for local, non-private development.

The honest scoring oracle is not complete yet. It is not permanently externally blocked: Week 1 Task 7 should stand up local SAM3 PGT scaffolding so validator_sim can score against the same PGT recipe family the validator uses. We should not pretend bootstrap PGT is a real score.

## Completed Phases

```text
Phase 0 - Recon / scoring spec: done
Phase 1 - Repo skeleton, class mapping, memory budget: done
Phase 2 - Local benchmark harness: done
Phase 3 - RF-DETR-M smoke deploy: done
Phase 4 - Replay/validator_sim harness: replay tooling done; SAM3 scoring oracle not done
```

## Phase 4 Deliverables

```text
endpoint replay runner
prediction summary
schema checker
replay artifact writer
review-frame exporter
PGT bootstrap helper
PGT audit helper
threshold sweep helper
TurboVision score_runner wrapper
private GT client wrapper, optional only
```

## Latest Good Replay

```text
runs/replays/rfdetr_m_t075_team_tracking_role_guard_v1
```

Expected files:

```text
request.json
response.json
summary.json
report.json
report.md
```

Latest summary:

```text
success: true
schema valid: true
frames_returned: 750
missing_frame_ids: []
empty_frames: 0
boxes_total: 6231
boxes/frame mean: 8.308
confidence min: 0.750405
team_id 1: 3269
team_id 2: 2779
team_id null: 183
class_counts: {"0": 6231}
valid_keypoints_total: 0
```

## Current Runtime Baseline

```text
RF-DETR-M
threshold 0.75
max_boxes_per_frame 18
OpenCV Lab/k-means team color
internal Supervision ByteTrack team memory
guarded role cleanup, no relabel until class ID is known
dummy keypoints
```

## Explicit Non-Goals For Phase 4

```text
No fake leaderboard score from unreviewed bootstrap PGT.
No private API dependency.
No role relabel without live manifest object order.
No detector head-to-head yet.
No keypoint/homography implementation yet.
```

## Next Phase Options

Recommended next work package:

```text
Implementation Spec Week 1, Tasks 1-7
```

Immediate order:

```text
Task 1 - BGR/RGB fix
Task 2 - ball_cls_id + env toggles
Task 3 - team_id carry-forward / null team leak fix
Task 4 - live manifest class mapping validation
Task 6 - real Chutes measured baseline
Task 7 - SAM3 PGT scaffold
```

Decision rule:

```text
Pillar weights and baseline_theta decide priority, not phase numbers.
Do not build pitch keypoints for PlayerDetect while keypoints_iou weight is absent.
Do not run detector head-to-head before silent baseline bugs are fixed.
```
