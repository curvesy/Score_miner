# Score/TurboVision 2026 Miner Plan

Date: 2026-05-18

Goal: build a top-tier TurboVision football miner from the cloned `turbovision/` repo. The strategy is a **chunk-aware football perception system**, not just a single-frame detector swap.

## 1. Decision

Use this stack:

```text
RF-DETR-M smoke test
+ RF-DETR-L vs DEIMv2-L vs D-FINE-L head-to-head before final detector choice
+ sn-gamestate / TrackLab reference pipeline for soccer GSR modules
+ day-one memory budget under Chutes 5 GB limit
+ day-one class mapping guard
+ VideoState world model
+ adaptive chunk/frame scheduler
+ validator_sim + replay mining + optimizer_core closed loop
+ temporal association / occlusion memory / smoothing
+ ball-specific high-resolution pass
+ team/role temporal memory, ReID, and optional jersey-number evidence
+ pitch keypoints now, TrackLab/TVCalib/SegFormer/AuxFlow-style homography next
+ score-aware confidence + uncertainty calibration
+ FiftyOne light offline review
+ one experiment tracker: MLflow local-first or W&B sweeps
+ Twelve Labs optional fallback only, not a priority
+ SAM3/VLM offline pseudo-labeling only
```

Use RF-DETR-M only as the first smoke-test. Do not crown RF-DETR-L, DEIMv2-L, or D-FINE-L from COCO AP. Run all serious detector candidates through the same TurboVision scorer, same postprocess, same memory budget, and same latency measurement. Use YOLO/YOLO26 for side tasks where speed matters, especially ball-only or keypoint models.

Why this is the right 2026 direction:

- RF-DETR is still valuable because it is designed for target-dataset fine-tuning, but D-FINE-L and DEIMv2-L are real contenders and must be benchmarked.
- `sn-gamestate` / TrackLab is the current SoccerNet-GSR reference framework; use it instead of rebuilding every soccer module from zero.
- SoccerNet-GSR and 2026 sports CV systems are pipeline systems: detection, tracking, homography, role/team identity, and temporal consistency.
- TurboVision scores structured football outputs and latency, not generic COCO AP.
- The Chute may receive a full video URL, so we must handle chunks, not only isolated images.

The non-conservative target is:

```text
video chunk
  -> scene analyzer
  -> adaptive compute scheduler
  -> detector race winner / ball / TrackLab-inspired geometry / team / ReID modules
  -> VideoState fusion
  -> score-aware calibration
  -> TurboVision-compatible output
```

The hidden edge is not one model. It is **persistent football state estimation**.

The leaderboard edge is not another folder. It is a **closed score optimization loop**:

```text
run miner
  -> simulate validator score
  -> archive failures
  -> mine hard replays
  -> mutate thresholds/scheduler/smoothing with Optuna
  -> retrain or recalibrate only when evidence says so
  -> redeploy best measured config
```

This loop must be built before serious fine-tuning. Fine-tuning without a local score loop only improves AP; the challenge rewards weighted TurboVision score.

## 2. What Their Repo Actually Requires

Main files:

```text
turbovision/scorevision/miner/open_source/example_miner/miner.py
turbovision/scorevision/miner/open_source/example_miner/chute_config.yml
turbovision/scorevision/miner/open_source/chute_template/turbovision_chute.py.j2
turbovision/scorevision/utils/evaluate.py
turbovision/scorevision/vlm_pipeline/non_vlm_scoring/objects.py
turbovision/scorevision/vlm_pipeline/non_vlm_scoring/keypoints.py
turbovision/scorevision/vlm_pipeline/non_vlm_scoring/smoothness.py
turbovision/scorevision/vlm_pipeline/vlm_annotator_sam3.py
turbovision/tests/test_data/manifests/example_manifest.yml
turbovision/tests/test_data/videos/example_football.mp4
```

Their example miner is only a template:

```python
self.bbox_model = YOLO(path_hf_repo / "football-player-detection.pt")
self.keypoints_model = YOLO(path_hf_repo / "football-pitch-detection.pt")
```

The Chute template expects:

```python
class Miner:
    def __init__(self, path_hf_repo): ...
    def predict_batch(self, batch_images, offset, n_keypoints): ...
```

It returns:

```text
{
  "frames": [
    {
      "frame_id": int,
      "boxes": [
        {
          "x1": int,
          "y1": int,
          "x2": int,
          "y2": int,
          "cls_id": int,
          "conf": float,
          "team_id": optional int/string,
          "cluster_id": optional int/string
        }
      ],
      "keypoints": [[x, y], ...]
    }
  ]
}
```

Important constraints:

- `cls_id` is mapped through the live manifest's `element.objects`; class order must match the manifest.
- `team_id` or `cluster_id` is accepted for player team assignment.
- The current Chute template loads `miner.py` directly and blocks arbitrary helper-module imports from the HF artifact.
- Imports from installed packages in `site-packages` are allowed by the template whitelist, so our preferred deploy path is to install our own package and keep `miner.py` as a thin entrypoint.
- The template checks loaded miner memory and fails if it estimates more than `5.0 GB`.
- The same Chute template supports either explicit `frames` payloads or a full `video_url`.

## 3. How Chunks Work

The validator/challenge path can produce two payload types.

### Explicit frames payload

`prepare_challenge_payload()` may build:

```text
payload.frames = [base64/url frames]
meta.n_frames_total = number of payload frames
meta.min_frames_required = selected/scored count
meta.batch_size = 64
meta.n_keypoints = 32
```

The Chute loads all payload frames and calls `predict_batch()`.

### Full video URL

If the payload has `url`, the Chute downloads the clip and processes the whole video:

```text
for batch in get_video_frames_in_batches(video, batch_size):
    miner.predict_batch(batch_images=batch, offset=batch_start, n_keypoints=n_keypoints)
```

This means the miner cannot assume it only sees the scored frames. A modern miner should be able to process a whole chunk efficiently and return stable frame-level predictions.

## 4. Scoring Targets

TurboVision scores more than detection:

```text
iou: bbox placement by matching
count: object count F1
palette: team assignment
role: player/referee/goalkeeper + team
smoothness: frame-to-frame bbox mask IoU
map50/precision/recall/false_positive when enabled
keypoints_iou or pitch calibration iou
latency/RTF gate
```

Pitch keypoints are scored by projecting the football pitch template into the frame and checking overlap with actual field lines. This rewards homography stability, not just raw keypoint proximity.

Example manifest values:

```text
fps: 5
resize_long: 1280
latency_p95_ms: 200
service_rate_fps: 25
n_keypoints: 32
```

RTF gate:

```text
RTF = (p95_latency_ms / 10000) * (service_rate_fps / 5)
```

With `service_rate_fps = 25`, hard zero-score is around `p95 > 2000 ms`, but the competitive target is much lower:

```text
target: < 200 ms end-to-end p95
acceptable early: 200-500 ms
risky: 500-1000 ms
bad: > 1000 ms
```

Local scoring-file implications:

- `objects.py` uses Hungarian matching. `iou` is label-agnostic AUC-F1 at IoU thresholds `0.3` and `0.5`; `count` is label-agnostic F1 at IoU `0.3`.
- `palette` is players-only team assignment. It accepts either TEAM1/TEAM2 orientation by testing both mappings, but unstable `cluster_id/team_id` still hurts palette and smoothness.
- `role` is the average of object label score and team score.
- `keypoints.py` scores homography quality by projecting pitch lines and comparing to real field-line edges. Fewer than four valid points, bowtie projections, unrealistic masks, or bad projected line overlap go to zero.
- `smoothness.py` groups by `bbox.label` plus `cluster_id`; team/cluster flips can damage smoothness even if boxes are good.

Therefore the highest-value engineering targets are:

```text
correct class mapping
stable frame count
memory under 5 GB
ball/player box placement
count precision/recall balance
team_id stability
role stability
valid homography/keypoints
smooth frame-to-frame boxes
```

## 5. 2026 Model Choice

### Detector decision

Do not choose the final detector by COCO AP or vendor claims. The detector decision is:

```text
RF-DETR-M smoke test first
then RF-DETR-L vs DEIMv2-L vs D-FINE-L head-to-head
optional SoccerDETR only if code/weights/export are usable
```

Reported comparison:

```text
RF-DETR-M: 54.7 AP, 4.4 ms T4 TensorRT FP16
RF-DETR-L: 56.5 AP, 6.8 ms T4 TensorRT FP16
DEIMv2-L: 56.0 AP, 10.47 ms
DEIMv2-X: 57.8 AP, 13.75 ms
D-FINE-L: strong L-size contender, about 54.0-57.3 AP depending pretraining, about 8.07 ms T4 FP16 TensorRT
YOLO26-L: fast, around 54.x AP class, useful fallback
```

Decision:

- RF-DETR-M: first smoke-test only.
- RF-DETR-L: serious candidate because RF-DETR was built around target-dataset fine-tuning, not because it is automatically best on raw COCO.
- DEIMv2-L: serious candidate because DINOv3 features may help small/occluded soccer objects and OOD scenes.
- D-FINE-L: serious candidate because L-size D-FINE can beat or match RF-DETR-L on some COCO-style benchmarks and has strong TensorRT latency.
- RF-DETR-XL/2XL and DEIMv2-X: not first live deployment because model size/cold-start/memory risk is higher.
- YOLO26: use for speed-sensitive side tasks, not as main 2026 detector unless RF-DETR deployment fails.

Head-to-head protocol:

```text
same SoccerNet/TurboVision validation clips
same resized inputs
same class mapping
same postprocess
same team/keypoint modules disabled for detector-only test
same p50/p95/p99 latency logging
same Chutes-style memory measurement
same validator_sim score breakdown
```

Final detector criterion:

```text
weighted TurboVision score
+ p95 latency
+ memory under 5 GB
+ export reliability
+ fine-tune path
+ failure profile on tiny ball / occluded players / wide shots
```

### Soccer-specific detector research

SoccerDETR is worth watching because it reports strong soccer-specific results in 2026. Do not make it first implementation unless weights/code/export are practical, because TurboVision deployment reliability matters as much as paper performance.

RT-DETRv4 and YOLO26 are also relevant, but they do not change the first serious choice:

- RT-DETRv4-X reports strong real-time AP, but RF-DETR-L has the cleaner current fine-tune/export story for this miner.
- YOLO26 is very useful where predictable latency matters, especially ball/keypoint side models, but it is not the first main detector.
- LW-DETR remains a watch-list ablation. D-FINE-L is upgraded from watch-list to direct head-to-head candidate.
- RF-DETR-XL/2XL should be revisited after the full pipeline works, because their extra AP may be worth it only if Chutes memory/cold-start and p95 stay healthy.

### SoccerNet / TrackLab leverage

Use `sn-gamestate` and TrackLab as the soccer-specific reference system:

```text
sn-gamestate:
  jersey number recognition
  team affiliation
  SoccerNet GSR task glue
  baseline configs

TrackLab:
  detection wrappers
  ReID wrappers
  tracking wrappers
  tracker states
  Hydra module configuration
```

Do not blindly put the full TrackLab stack into live TurboVision first. Use it in three stages:

```text
1. offline reference and benchmark
2. module source for ReID/tracking/calibration/team/jersey logic
3. live wrapper only after memory and latency prove it fits Chutes
```

This keeps the plan aggressive without letting dependency weight break the miner.

### Pitch calibration

Use an AuxFlow-style upgrade:

```text
keypoints on anchor frames
-> homography from reliable anchors
-> optical-flow auxiliary point propagation
-> temporal homography smoothing
```

This matches current SoccerNet-GSR direction and directly fits TurboVision keypoint scoring.

### Tracking and identity

Current TurboVision schema does not directly score track IDs, but smoothness, palette, and role all benefit from internal identity. Do not expose unnecessary identity fields, but do use ReID/tracklets internally.

Use:

```text
TrackLab tracker baselines
BoT-SORT / DeepOCSORT / Deep-EIoU style ablations
IoU + appearance association
camera-motion compensation when available
motion/flow assisted interpolation
EMA/Kalman smoothing with scene-cut reset
separate ball logic
short crop embeddings when needed
```

Add ReID in measured stages:

```text
Stage 1: HSV/Lab team color memory only
Stage 2: OSNet/PRTReID/CLIP-ReIdent crop embeddings from TrackLab-style pipeline
Stage 3: jersey-number evidence on selected high-confidence tracklets
```

Jersey number recognition is not required by TurboVision output today, but it can stabilize player identity, team affiliation, and goalkeeper/player role in hard scenes. Use Koshkina/PARSeq-style jersey number recognition offline first, then selective live only if latency and memory allow it.

### SAM3, Molmo2, Qwen3-VL, C-RADIOv4-H

Use these aggressively offline, not live first:

- SAM3: pseudo-GT alignment, concept segmentation for players/jerseys/ball, hard example mining.
- Molmo2 / Qwen3-VL: hard crop QA, role/team disagreement review, weak labels for rare scenes.
- C-RADIOv4-H: crop embeddings or teacher features for team/role/hard-negative mining.

Live VLM/SAM inference is a later experiment only if a manifest Element explicitly benefits from it and latency allows it.

## 6. Most Important 2026 Upgrade: VideoState

The miner should not be stateless. Build an internal world model for every video/chunk:

```python
class VideoState:
    scene_id
    camera_type
    camera_motion
    homography
    homography_confidence
    player_tracks
    referee_tracks
    goalkeeper_tracks
    team_color_memory
    role_memory
    ball_state
    confidence_history
    occlusion_memory
    last_frame_features
```

Why this matters:

- Smoothness improves because boxes do not jitter frame-to-frame.
- Team labels stop flipping.
- Referee/goalkeeper labels become stable.
- Ball false positives can be rejected by trajectory plausibility.
- Pitch keypoints become stable through homography memory.
- The miner can spend compute only where uncertainty is high.

This is the difference between:

```text
frame detector miner
```

and:

```text
2026 football reconstruction miner
```

## 7. Adaptive Compute Strategy

Do not run identical heavy inference on every frame forever. Use adaptive compute:

| Situation | Compute |
|---|---|
| scene cut | full RF-DETR refresh + keypoint refresh |
| stable wide shot | tracking + sparse detector refresh |
| penalty-box crowd | high-res player/ball pass |
| ball uncertain | ball refiner + trajectory filter |
| homography unstable | keypoint/homography refresh |
| close-up/crowd/replay | suppress pitch assumptions and reduce false positives |
| fast camera pan | optical-flow assisted smoothing |

This improves score and latency together. It is the correct 2026 approach for chunk video.

## 8. Project Structure

Use two folders:

```text
validator_improve/
  turbovision/          # upstream repo, keep mostly unchanged
  score_miner_dev/      # our clean local project
  score_miner/          # deployable TurboVision package
```

Local development:

```text
score_miner_dev/
  pyproject.toml
  score_miner_core/
    __init__.py
  runtime/
    orchestrator.py
    scheduler.py
    video_state.py
    scene_analyzer.py
    memory_budget.py
    class_mapping.py
  detector/
    rfdetr_runner.py
    deim_runner.py
    dfine_runner.py
    soccerdetr_runner.py
    detector_router.py
    detector_benchmark.py
    confidence_calibrator.py
  chunk/
    frame_sampler.py
    temporal_cache.py
    video_reader.py
  tracking/
    association.py
    motion_model.py
    occlusion_memory.py
    tracklab_adapter.py
    reid_embeddings.py
  team/
    jersey_cluster.py
    jersey_number_evidence.py
    role_cleanup.py
    temporal_palette_memory.py
  keypoints/
    homography_filter.py
    auxflow_propagation.py
    camera_motion.py
    tracklab_calibration_adapter.py
  ball/
    crop_refiner.py
    trajectory_filter.py
    crop_scheduler.py
  benchmark/
    run_local.py
    score_breakdown.py
    replay_runner.py
  validator_sim/
    manifest_loader.py
    schema_checker.py
    pillar_metrics.py
    rtf_gate.py
    report.py
  optimizer_core/
    objective.py
    search_space.py
    optuna_runner.py
    score_graph.py
    post_validator_patch_engine.py
    config_registry.py
  replay/
    failure_store.py
    clip_sampler.py
    hard_case_index.py
    regression_suite.py
  active_learning/
    fiftyone_export.py
    uncertainty_sampler.py
    annotation_queue.py
    hard_negative_builder.py
  external/
    sn_gamestate_notes.md
    tracklab_adapter_notes.md
  telemetry/
    wandb_logger.py
    mlflow_logger.py
    run_manifest.py
  experiments/
    configs/
    results/
    leaderboards/
  datasets/
    raw/
    curated/
    hard_negatives/
    calibration/
  export/
    onnx/
    tensorrt/
  training/
    pseudo_labeling/
    hard_negative_mining/
    distillation/
  deploy/
    bundle_builder.py
    miner_flatten.py
    healthcheck.py
  miner_entry.py
```

Deploy package:

```text
score_miner/
  miner.py              # thin entrypoint importing score_miner_core
  chute_config.yml
  README.md
  dist/
    score_miner_core-*.whl
  models/
    rfdetr_l_checkpoint.pth
    rfdetr_m_checkpoint.pth
    deimv2_l_checkpoint.pth
    dfine_l_checkpoint.pth
    football-pitch-detection.pt
    optional_ball_detector.pt
  notes/
    class_mapping.md
    benchmark_results.md
    optimizer_results.md
    replay_failures.md
    calibration.md
```

Why two folders:

- `score_miner_dev/` lets us work like real engineers.
- `score_miner/` stays compatible with their current Chute template.
- Primary deploy strategy: build `score_miner_core` as a wheel, install it into `site-packages` from `chute_config.yml`, and keep `score_miner/miner.py` thin.
- Fallback deploy strategy: if Chutes/TurboVision rejects package imports, flatten the runtime into one deployable `miner.py`.

Do not patch the Chute template first. Validators compare deployed Chute code against the expected template. Template changes may trigger integrity failure unless the protocol accepts them.

## 9. Closed Score Optimization Loop

This is now a higher priority than fine-tuning. A top miner should have an automatic loop that learns which config maximizes the actual validator-style score.

Build:

```text
validator_sim/
  -> reproduces schema checks, frame-count checks, pillar scores, and RTF gate

replay/
  -> stores clips/frames where score drops, latency spikes, class mapping fails, ball false positives appear, team flips happen, or homography breaks

optimizer_core/
  -> uses Optuna to mutate config and maximize weighted score

post_validator_patch_engine/
  -> diagnoses score breakdown and proposes safe config changes

telemetry/
  -> logs every config, model hash, dataset split, score vector, latency vector, and failure bucket
```

The optimizer should tune the things that actually move TurboVision score:

```text
detector_candidate: rfdetr_l / deimv2_l / dfine_l
player_conf
ball_conf
referee_conf
goalkeeper_conf
max_boxes_per_frame
per-scene threshold multipliers
tracker IoU threshold
tracker inertia / Kalman noise
box smoothing alpha
team clustering temperature
team flip hysteresis
ball temporal confirmation window
ball crop scale
keypoint confidence threshold
homography smoothing weight
homography invalidation threshold
scene-cut sensitivity
RF-DETR refresh cadence
high-res crop trigger threshold
reid_weight
team_hysteresis_frames
memory_budget_mode
```

Use Optuna first. Nevergrad/CMA-ES are backup choices if the search space becomes more discrete or non-smooth.

Objective:

```text
maximize:
  weighted_validator_score
  - latency_penalty
  - memory_penalty
  - frame_count_failure_penalty
  - nondeterminism_penalty
```

The score graph must persist:

```text
config hash
model hash
dataset split
video/chunk ids
score total
pillar scores
p50/p95/p99 latency
memory GB
failure tags
artifact paths
```

This is the system that tells us whether RF-DETR-L, DEIMv2-L, D-FINE-L, SoccerDETR ideas, YOLO26 ball, TrackLab modules, TensorRT, INT8, or a threshold change actually helps.

## 10. Miner Runtime Architecture

### V1: valid smoke miner

Inside deployable `miner.py`, keep only the entrypoint:

```python
from score_miner_core.runtime import MinerRuntime

class Miner:
    def __init__(self, path_hf_repo):
        self.runtime = MinerRuntime(path_hf_repo)

    def predict_batch(self, batch_images, offset, n_keypoints):
        return self.runtime.predict_batch(batch_images, offset, n_keypoints)
```

Inside `score_miner_core`:

```text
load:
  RF-DETR-M smoke detector first
  candidate detector interface for RF-DETR-L / DEIMv2-L / D-FINE-L
  YOLO pitch keypoint model

predict_batch:
  decode/receive BGR frames
  resize/preprocess for detector
  run selected detector batch
  map labels to manifest class ids
  run pitch keypoint model
  add team_id clustering
  clean referee/goalkeeper roles
  apply light temporal smoothing inside the batch
  return valid TVFrameResult objects
```

V1 must prioritize:

- valid schema
- correct frame IDs
- correct class mapping
- memory under Chutes 5 GB limit
- nonzero score
- low p95 latency
- stable Chute health

### V2: VideoState chunk system

Add:

```text
scene analyzer
adaptive scheduler
VideoState object
temporal cache across predict_batch calls when possible
scene-cut detection
IoU/flow association
player box interpolation for skipped/weak frames
ball crop refiner
team color memory
homography smoothing
confidence calibration
score-aware output shaping
```

If the Chute instance keeps state between calls, cache should be per-video/task and bounded. If not, do smoothing inside each batch.

### V3: trained leaderboard miner

Add:

```text
winner detector fine-tuned on football objects
ball-specific model trained on hard tiny-ball frames
TrackLab-inspired ReID/tracking/team modules
optional jersey-number evidence for stable identity/team/role
AuxFlow/TVCalib/SegFormer-style keypoint/homography module
TensorRT engine or ONNX runtime if Python inference is too slow
SAM3/DEIMv2/D-FINE pseudo-label teacher pipeline
calibration dataset for validator-score thresholds
```

## 11. Score-Aware Calibration And Uncertainty

This is a crucial edge. Do not maximize detector recall blindly. Maximize final TurboVision score.

Tune thresholds per class and scene:

```text
player: balanced recall/count
ball: high precision first, recall second
referee: high precision, avoid team pollution
goalkeeper: temporal stability before relabeling
keypoints: suppress impossible homographies
```

The output layer should be score-aware:

```text
raw detections
  -> class-specific thresholding
  -> temporal consistency checks
  -> false-positive suppression
  -> calibrated confidence
  -> final TurboVision boxes/keypoints
```

Examples:

- A weak ball detection in one frame should not be output unless trajectory/history supports it.
- A team assignment should not flip because one crop is shadowed.
- A homography that creates bowtie/spread errors should be replaced by propagated state or `(0,0)` missing points.
- Extra low-confidence boxes may improve recall but reduce count, precision, smoothness, and role score.

Add conformal/uncertainty calibration after the first replay set exists:

```text
detection confidence
  -> calibrated per class and scene
  -> uncertainty bucket
  -> action: output / suppress / track-only / run expensive refiner
```

Use split calibration data from replay clips. Calibrate separately for:

```text
player/referee/goalkeeper/ball
wide shot vs close-up
motion blur vs stable camera
crowd/penalty-box vs open field
day/night or strong shadow scenes
```

Do not use uncertainty as decoration. It must drive decisions:

- uncertain ball: crop refiner or suppress unless trajectory supports it
- uncertain team color: keep previous team_id through hysteresis
- uncertain homography: propagate previous valid homography instead of emitting unstable keypoints
- uncertain scene: full refresh instead of sparse tracking

## 12. Data Review, Active Learning, And Experiment Tracking

Use tools as part of the score loop, not as live miner dependencies.

### FiftyOne

Use offline for visual failure mining:

```text
load replay frames/videos
attach predictions and pseudo-GT
tag false positives, misses, duplicates, bad team labels, invalid homographies
find near-duplicate easy frames and remove them from training emphasis
find hard clusters for annotation or pseudo-label cleanup
export hard-negative and calibration splits
```

FiftyOne is especially useful for:

- tiny ball misses
- referee/player confusion
- goalkeeper/team mistakes
- ad-board/person false positives
- close-up/replay scenes that should be suppressed
- frame groups where smoothing helps or hurts

### MLflow or W&B

Use one tracking backend from day one. Do not build both unless there is a clear reason. MLflow is easier local-first; W&B is better if sweeps/remote dashboards matter. The tracked unit is not just a training run. Track every deployed miner run:

```text
model version
config version
dataset/replay split
validator_sim score vector
latency vector
memory
failure buckets
export type: PyTorch / ONNX / TensorRT FP16 / TensorRT INT8
```

### Twelve Labs

Twelve Labs is optional, not a core dependency. Prefer SoccerNet labels and replay mining first because they are cheaper, already domain-specific, and closer to the validation distribution.

Use Twelve Labs only if SoccerNet/replay data is not enough to find rare cases:

```text
find penalty-box crowd scenes
find corner kicks / free kicks
find goalkeeper close-ups
find broadcast replays and non-field shots
find ball-in-air or fast camera pan clips
find rare lighting / shadows / snow / rain clips
```

Do not put Twelve Labs in live inference. It is a mining/indexing fallback for better replay and training data.

## 13. Exact Build Steps

### Step 1 - Inspect live manifest

Need:

```text
element_id
element.objects class order
pillar weights
latency_p95_ms
service_rate_fps
preproc resize/fps
ground_truth or pseudo-GT
```

Commands:

```bash
cd turbovision
uv sync
sv elements list
sv manifest current
```

If CLI names differ, inspect `scorevision/cli/manifest.py` and `scorevision/cli/elements.py`.

### Step 2 - Add class mapping guard and memory budget

Create:

```text
score_miner_dev/score_miner_core/runtime/class_mapping.py
score_miner_dev/score_miner_core/runtime/memory_budget.py
score_miner/notes/class_mapping.md
score_miner/notes/memory_budget.md
```

This must happen before serious model work:

```text
manifest class order
detector label -> manifest cls_id
team_id/cluster_id normalization
frame_id contract
estimated model memory
runtime memory from /health
cache limits for VideoState/replay
```

The Chute template has a `5.0 GB` loaded-model memory check. Treat `4.5 GB` as the warning line and `5.0 GB` as failure.

### Step 3 - Build local benchmark first

Create:

```text
score_miner_dev/benchmark/run_local.py
```

It should:

- load test video `turbovision/tests/test_data/videos/example_football.mp4`
- sample frames and whole chunks
- call our miner
- validate schema
- measure p50/p95/p99
- measure per-module latency
- replay repeated runs to catch nondeterminism
- record memory before/after model load and after prediction
- write `score_miner/notes/benchmark_results.md`

### Step 4 - Create deployable RF-DETR-M smoke test

Create:

```text
score_miner_dev/score_miner_core/
score_miner_dev/pyproject.toml
score_miner/miner.py
score_miner/chute_config.yml
score_miner/README.md
score_miner/dist/score_miner_core-*.whl
```

Use RF-DETR-M first only to prove:

- package installs
- model loads
- thin `miner.py` can import `score_miner_core` from `site-packages`
- output schema works
- frame IDs are correct
- Chute health passes
- memory check passes

If the installed-package path fails, use `deploy/miner_flatten.py` to generate a single-file `score_miner/miner.py` fallback.

### Step 5 - Build validator_sim

Create:

```text
score_miner_dev/validator_sim/
score_miner/notes/validator_sim.md
```

It must reproduce the challenge failure modes before we optimize:

```text
schema validation
frame_id / frame_count validation
manifest class mapping
pillar score breakdown
RTF gate
latency p50/p95/p99
memory snapshot
```

The first version must call TurboVision's local scoring utilities where possible. Do not rewrite scoring logic by hand if the repo already exposes it.

### Step 6 - Detector head-to-head

Before building around one detector, benchmark:

```text
RF-DETR-L
DEIMv2-L
D-FINE-L
optional SoccerDETR if usable
```

Use the same adapter and postprocess for all candidates:

```text
same validation clips
same frame sampling
same class mapping
same confidence threshold search
same output JSON schema
same validator_sim score breakdown
same p50/p95/p99 latency
same memory-budget report
```

Pick the winner by TurboVision score, latency, memory, and deploy reliability. If no large model fits, keep RF-DETR-M or the best medium candidate and spend effort on ball/team/keypoints.

### Step 7 - Build replay mining

Create:

```text
score_miner_dev/replay/
score_miner/notes/replay_failures.md
```

Every benchmark/dry-run should save:

```text
input video/chunk id
sampled frames
predictions JSON
score breakdown
latency trace
failure tags
thumbnail/contact sheet path
```

Failure tags:

```text
ball_miss
ball_false_positive
player_count_low
player_count_high
team_flip
role_confusion
homography_invalid
smoothness_drop
scene_cut_oversmooth
latency_spike
frame_count_error
```

### Step 8 - Add optimizer_core with Optuna

Create:

```text
score_miner_dev/optimizer_core/
score_miner/notes/optimizer_results.md
```

The Optuna objective must run:

```text
config sample
  -> miner on replay/local videos
  -> validator_sim
  -> weighted score
  -> latency/memory penalties
  -> persisted trial result
```

Start with config-only optimization. Do not retrain inside early Optuna loops; retraining makes each trial too slow. Retraining is triggered only after replay mining proves a repeatable data gap.

### Step 9 - Clone/read sn-gamestate and TrackLab, then adapt modules

Create:

```text
score_miner_dev/external/sn_gamestate_notes.md
score_miner_dev/external/tracklab_adapter_notes.md
score_miner_dev/tracking/tracklab_adapter.py
score_miner_dev/keypoints/tracklab_calibration_adapter.py
```

Use `sn-gamestate` / TrackLab in this order:

```text
offline reference run
module inventory
tracker/ReID/calibration/team ideas
TurboVision schema adapter
live wrapper only after memory/latency test
```

Do not copy the full framework into the live miner until the Chutes memory and dependency budget prove it works.

### Step 10 - Add FiftyOne review and hard-negative mining

Create:

```text
score_miner_dev/active_learning/fiftyone_export.py
score_miner_dev/active_learning/hard_negative_builder.py
```

Use it to turn replay failures into curated training/calibration splits:

```text
hard_negatives/
calibration/
tiny_ball/
role_confusion/
bad_homography/
closeup_suppress/
```

### Step 11 - Add MLflow or W&B tracking

Create:

```text
score_miner_dev/telemetry/
```

At minimum log to local JSONL first, then exactly one of MLflow or W&B:

```text
run_id
git sha
model hash
config hash
score vector
latency vector
memory
failure buckets
artifact paths
```

### Step 12 - Optional Twelve Labs offline chunk search

Use only if SoccerNet labels plus replay mining do not produce enough rare hard cases. Output should be a list of candidate clips/segments to download/review, not runtime predictions.

Priority searches:

```text
penalty box crowd
corner kick
free kick
goalkeeper save
ball in air
fast camera pan
broadcast replay
close-up interview/crowd
rain/snow/shadow
```

### Step 13 - Add team/role logic

Implement in `miner.py` first:

```text
player upper-body crop
grass mask removal
HSV/Lab color feature
two-centroid team clustering
temporal centroid matching
team_id 1/2 output
referee/goalkeeper high-confidence cleanup
optional ReID embedding support
optional jersey-number evidence for stable tracklets
```

This is a high-value improvement because TurboVision has `palette` and `role` pillars.

### Step 14 - Add VideoState and scene analyzer

Implement:

```text
scene-cut detection
camera motion estimate
short track memory
team color memory
ball state
homography state
confidence history
```

Start simple but keep the structure. Even simple state beats stateless frame output.

### Step 15 - Add adaptive scheduler

Implement routing:

```text
full detector refresh on scene cut
normal detector on sampled cadence
ball refiner only when ball uncertain or near action
keypoint refresh only when homography confidence drops
high-res pass only on crowded/penalty-box frames
```

Keep deterministic fallbacks for Chute reliability.

### Step 16 - Add ball specialist

Start conservative:

```text
high confidence threshold
small high-res crop/tile pass
temporal confirmation if adjacent frames exist
weak smoothing only
```

Do not flood false positives. Ball recall matters, but false positives can damage precision/count/false-positive pillars.

### Step 17 - Add pitch/homography upgrade

V1:

```text
existing YOLO pitch keypoints
exactly 32 points
(0,0) for missing
clamp/reject invalid points
```

V2:

```text
TrackLab/TVCalib/SegFormer calibration reference
anchor frame selection
homography validation
flow-guided keypoint propagation
temporal smoothing
scene-cut reset
```

This is where AuxFlow-style 2026 systems beat naive per-frame keypoints.

### Step 18 - Add score-aware calibration and uncertainty

Create a small calibration report:

```text
score_miner/notes/calibration.md
```

Track best thresholds for:

```text
player_conf
ball_conf
referee_conf
goalkeeper_conf
max_boxes_per_frame
ball_temporal_confirmation
homography_confidence_min
uncertainty_action_thresholds
```

Tune against local score breakdown and dry-run feedback.

Then add split conformal-style calibration only where it affects action choices:

```text
suppress
keep previous state
run refiner
emit output
```

### Step 19 - Dry-run deploy

Use no commit:

```bash
cd turbovision
sv -v deploy-os-miner --model-path ../score_miner --element-id <ELEMENT_ID> --no-commit
```

Then:

```bash
curl -X POST https://<CHUTE_SLUG>.chutes.ai/health \
  -d '{}' \
  -H "Authorization: Bearer $CHUTES_API_KEY"

curl -X POST https://<CHUTE_SLUG>.chutes.ai/predict \
  -d '{"url":"https://scoredata.me/2025_03_14/35ae7a/h1_0f2ca0.mp4","meta":{}}' \
  -H "Authorization: Bearer $CHUTES_API_KEY"

sv -vv run-once
```

Only commit after:

- health clean
- memory under 5 GB, with 4.5 GB treated as warning
- predict valid
- score nonzero
- p95 acceptable
- class mapping correct
- output frame count accepted

### Step 20 - Training pipeline

Data:

```text
SoccerNet-GSR
SoccerNet Tracking
SoccerNet camera calibration
Soccana / soccer detection data if accessible
SoccerTrack v2 for role/team/track-style supervision
public broadcast clips with manual cleanup
```

Teacher labels:

```text
SAM3 for pseudo-GT alignment
DEIMv2-L/X and D-FINE-L as detector teachers
sn-gamestate/TrackLab outputs as soccer-specific pseudo-labels where useful
Qwen/Molmo only for hard crop QA or role/team disagreements
C-RADIOv4-H for crop embeddings/teacher features if useful
```

Train:

```text
winner detector football fine-tune
ball-only detector/refiner
role/team crop classifier
optional ReID/jersey-number helpers
pitch keypoint model or homography confidence model
```

Oversample:

```text
tiny ball
crowded penalty boxes
motion blur
goalkeeper/referee confusion
wide broadcast views
ad-board false positives
partial field-line visibility
replay/close-up non-field false positives
hard negatives from FiftyOne review
optimizer-discovered weak scenes
```

Export:

```text
winning detector ONNX first
TensorRT FP16 after stable
INT8 only after calibration data proves no score loss
```

## 14. Evaluation Protocol

Compare these variants:

```text
A. baseline YOLO example
B. RF-DETR-M + YOLO keypoints
C. RF-DETR-L detector-only
D. DEIMv2-L detector-only
E. D-FINE-L detector-only
F. best detector + YOLO keypoints
G. best detector + team/role logic
H. best detector + TrackLab-inspired ReID/tracking
I. best detector + VideoState
J. best detector + ball refiner
K. best detector + TrackLab/TVCalib/AuxFlow homography smoothing
L. best detector + score-aware calibration
M. best detector + validator_sim/replay/Optuna tuned config
N. best detector + uncertainty calibration
O. best detector fine-tuned on replay-mined data
P. best detector TensorRT FP16
Q. best detector TensorRT INT8 only if score holds
R. SoccerDETR-inspired/soccer-specific detector ablation only if deployable
```

Track:

```text
score total
iou
count
role
palette
smoothness
keypoints
precision/recall/false_positive if present
p50/p95/p99 latency
memory GB from /health
peak loaded model memory
cold start failures
frame-count failures
module latency
detector candidate
scene-cut failures
team flip rate
track ID switch proxy
ReID/team evidence usefulness
ball false-positive rate
homography invalid rate
optimizer trial score
replay failure recurrence
calibration expected-vs-observed confidence
dataset split leakage
```

Decision rule:

- Use the detector that wins validator_sim under the same postprocess, memory, and latency constraints.
- Use RF-DETR-M or another medium model if large candidates break Chutes or memory.
- Use TrackLab/sn-gamestate modules when they improve score per millisecond and fit deployment constraints.
- Do not judge by COCO AP alone.
- Do not fine-tune until replay and validator_sim can prove what failure we are training for.

## 15. What Not To Do

Do not:

- build only a detector miner
- choose RF-DETR-L only because of headline COCO AP
- dismiss DEIMv2-L or D-FINE-L without a head-to-head on soccer clips
- run full TrackLab live before measuring dependency, memory, and p95 cost
- run SAM3 or Qwen/Molmo live in the first miner
- patch the Chute template first
- commit on-chain before dry-run score is nonzero
- assume manifest class order
- return too few frames
- over-smooth through camera cuts
- let team labels flip frame-to-frame
- chase ball recall with many false positives
- smooth pitch keypoints into invalid homographies
- chase XL/2XL before the full football pipeline works
- optimize TensorRT before correctness and score breakdown are known
- train blindly before replay mining proves the failure distribution
- use FiftyOne/W&B/MLflow as vanity tooling without tying them to score deltas
- run both MLflow and W&B unless one is clearly needed for a different workflow
- use Twelve Labs live in the miner
- make Twelve Labs a core dependency before exploiting SoccerNet labels and replay mining
- let Optuna optimize on an easy or leaky validation split

## 16. Sources Checked

- TurboVision repo files listed above.
- RF-DETR docs/model info: https://roboflow-rf-detr.mintlify.app/
- RF-DETR export docs: https://rfdetr.roboflow.com/develop/learn/export/
- RF-DETR Hugging Face Transformers docs: https://huggingface.co/docs/transformers/main/model_doc/rf_detr
- RF-DETR repo: https://github.com/roboflow/rf-detr
- DEIMv2 repo/model zoo: https://github.com/Intellindust-AI-Lab/DEIMv2
- D-FINE repo: https://github.com/Peterande/D-FINE
- SoccerNet GSR: https://www.soccer-net.org/tasks/game-state-reconstruction
- sn-gamestate / TrackLab baseline: https://github.com/SoccerNet/sn-gamestate
- TrackLab docs: https://trackinglaboratory.github.io/tracklab/
- SoccerNet GSR paper: https://arxiv.org/abs/2404.11335
- SoccerNet 2025 results: https://arxiv.org/abs/2508.19182
- Broadcast to Minimap: https://arxiv.org/abs/2504.06357
- Broadcast2Pitch WACV 2026: https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf
- AuxFlow 2026: https://www.sciencedirect.com/science/article/pii/S1077314226000299
- SoccerDETR 2026: https://www.mdpi.com/2227-7080/14/3/142
- SoccerTrack v2: https://atomscott.github.io/SoccerTrack-v2/
- SAM3 ICLR 2026: https://openreview.net/forum?id=r35clVtGzw
- Molmo2 2026: https://arxiv.org/abs/2601.10611
- C-RADIOv4-H: https://huggingface.co/nvidia/C-RADIOv4-H
- FiftyOne docs: https://docs.voxel51.com/
- Optuna docs/repo: https://optuna.org/ and https://github.com/optuna/optuna
- MLflow tracking docs: https://www.mlflow.org/docs/latest/ml/tracking
- W&B/YOLO experiment tracking docs: https://docs.ultralytics.com/integrations/weights-biases/
- Twelve Labs API docs: https://twelve-labs-api-doc.readme.io/
- TensorRT docs: https://docs.nvidia.com/deeplearning/tensorrt/index.html
- Conformal object detection 2026: https://arxiv.org/abs/2605.07549
- YOLO26 paper: https://arxiv.org/abs/2601.12882
- LW-DETR repo: https://github.com/atten4vis/lw-detr
- D-FINE overview: https://www.papercodex.com/d-fine-real-time-object-detection-with-detr-level-accuracy-and-no-inference-overhead/
- Koshkina jersey-number pipeline: https://github.com/mkoshkina/jersey-number-pipeline
- Jersey number recognition paper: https://arxiv.org/abs/2405.13896

## 17. Final Build Order

```text
1. Create score_miner_dev and score_miner.
2. Inspect live manifest and scoring files.
3. Add class mapping guard and memory budget.
4. Build local benchmark.
5. Deploy RF-DETR-M smoke test.
6. Build validator_sim using TurboVision scoring utilities.
7. Run RF-DETR-L vs DEIMv2-L vs D-FINE-L head-to-head.
8. Build replay mining.
9. Add optimizer_core with Optuna.
10. Clone/read sn-gamestate and TrackLab; document usable modules.
11. Add TrackLab-inspired adapter pieces only where they fit memory/latency.
12. Add FiftyOne light dataset review and hard-negative mining.
13. Add exactly one experiment tracker: MLflow local-first or W&B sweeps.
14. Keep Twelve Labs optional only if SoccerNet/replay mining lacks rare chunks.
15. Add team_id / role cleanup with ReID and optional jersey evidence.
16. Add VideoState and scene analyzer.
17. Add adaptive scheduler.
18. Add ball refiner.
19. Add TrackLab/TVCalib/AuxFlow-style homography smoothing.
20. Add score-aware calibration and uncertainty/conformal calibration.
21. Run Optuna score tuning on replay/local held-out splits.
22. Fine-tune the winning detector only from replay-proven failures.
23. Export ONNX/TensorRT if Python inference is the bottleneck.
24. Test SoccerDETR/YOLO26/LW-DETR/RF-DETR-XL/DEIMv2-X only as measured ablations.
25. Commit only after dry-run is stable, nonzero, memory-safe, and class-correct.
```
