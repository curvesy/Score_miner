# Scoring Spec

Date: 2026-05-19

This is the Phase 0 contract extracted from the local TurboVision repo.

## Manifest Example

Source: `turbovision/tests/test_data/manifests/example_manifest.yml`

### PlayerDetect_v1@1.0

- element weight: `0.6`
- preproc fps: `5`
- resize_long: `1280`
- latency_p95_ms: `200`
- service_rate_fps: `25`
- pillars:
  - `iou`: `0.35`
  - `count`: `0.2`
  - `palette`: `0.15`
  - `smoothness`: `0.15`
  - `role`: `0.15`

### BallDetect_v1@1.0

- element weight: `0.4`
- preproc fps: `5`
- resize_long: `1280`
- latency_p95_ms: `200`
- service_rate_fps: `25`
- pillars:
  - `iou`: `0.45`
  - `count`: `0.35`
  - `smoothness`: `0.2`

## Output Contract

The Chute imports a root-level `miner.py` and expects:

```python
class Miner:
    def __init__(self, path_hf_repo): ...
    def predict_batch(self, batch_images, offset, n_keypoints): ...
```

Each result frame must include:

```text
frame_id
boxes: [{x1, y1, x2, y2, cls_id, conf, optional team_id/cluster_id}]
keypoints: exactly n_keypoints pairs, with (0, 0) for missing points
```

## Scoring Code Findings

Source files:

- `turbovision/scorevision/vlm_pipeline/non_vlm_scoring/objects.py`
- `turbovision/scorevision/vlm_pipeline/non_vlm_scoring/keypoints.py`
- `turbovision/scorevision/vlm_pipeline/non_vlm_scoring/smoothness.py`
- `turbovision/scorevision/utils/evaluate.py`

### Objects

- `iou` is label-agnostic Hungarian AUC-F1 at IoU thresholds `0.3` and `0.5`.
- `count` is label-agnostic Hungarian F1 at IoU `0.3`.
- `palette` is players-only team assignment using `team_id` or `cluster_id`.
- TEAM1/TEAM2 orientation can flip; the scorer tests both mappings and keeps the better one.
- `role` is average of object-label score and team-label score.
- Optional metrics exist for `map50`, `precision`, `recall`, and `false_positive`.

### Keypoints

- Missing points must be `(0, 0)`.
- Fewer than four valid keypoints means zero homography score.
- Invalid projected masks, bowtie projections, huge spreads, or poor field-line overlap score zero.
- Good homography consistency matters more than simply filling all keypoints.

### Smoothness

- Smoothness is frame-to-frame bbox mask IoU divided by jerkiness.
- Boxes are grouped by `bbox.label` and `cluster_id`.
- Team or cluster flips can reduce smoothness even if geometry is stable.

## Latency / Memory

- RTF formula from local code: `(p95_latency_ms / 10000) * (service_rate_fps / 5)`.
- With `service_rate_fps=25`, hard gate is around `p95 > 2000 ms`.
- Competitive target remains much lower: under `200 ms` p95 when possible.
- Chute template sets `MAX_LOADED_MODEL_SIZE_GB = 5.0`.
- Treat `4.5 GB` as warning and `5.0 GB` as failure.

## Immediate Engineering Consequences

1. Build class mapping guard before model race.
2. Track memory from the first smoke test.
3. Use TurboVision scoring utilities inside `validator_sim`; do not reimplement scoring.
4. Run RF-DETR-L, DEIMv2-L, and D-FINE-L through the same adapter and scorer.
5. Optimize final score, not COCO AP.
