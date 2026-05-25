# Score Public Detect Rules

Current live public Detect rule summary, verified from manifest and local validator code.

## Active Public Elements

```text
manak0/Detect-beverage-detect
manak0/Detect-crime
manak0/Detect-fire
manak0/Detect-car-wash
```

## Size Rule

```text
Full Hugging Face model repository revision <= max_model_size_mb.
Current public cap: 30MB.
```

Validator enforcement path:

```text
turbovision/scorevision/validator/central/open_source/runner.py
  passes element.max_model_size_mb to get_miners_from_registry()

turbovision/scorevision/utils/miner_registry.py
  sums HfApi.list_repo_tree(... recursive=True, expand=True)
  skips miners whose full repo size exceeds cap
```

This means:

```text
No extra checkpoints.
No training logs.
No dataset files.
No duplicate fp32/fp16 artifacts.
Keep the HF repo minimal.
```

## Class Order

Use manifest order exactly.

### Beverage

```text
0 cup
1 bottle
2 can
```

### Car-wash

```text
0 broom
1 drainage gate
2 nozzle
3 track
```

### Crime

```text
0 balaclava
1 hoodie
2 glove
3 bat
4 spray paint
5 graffiti
```

### Fire

```text
0 fire
1 smoke
2 fire extinguisher
```

## Strategy

```text
Use Score starterPack/proof frames as validation anchor.
Control false positives.
Sweep confidence thresholds.
Deploy correct baseline early.
Improve with diversity-first synthetic data and hard negatives.
```

## Non-Skippable Win Rule

For a top-3 attempt, do not choose models by normal Ultralytics mAP alone.

Use:

```text
Score-style validation = map50 approximation + false-positive pressure
```

Then tune:

```text
global confidence threshold
per-class thresholds
max detections
image size
SAHI/TTA only after baseline
```

Reason:

```text
High recall with many junk boxes can look good in normal training metrics but lose on SN44 because false positives are part of the public Detect score.
```

## Local Phase 3 Scorer

The local scorer is implemented in:

```text
src/public_detect/score_eval.py
scripts/score_threshold_sweep.py
```

It does:

```text
1. Run YOLO prediction once at low confidence.
2. Cache raw detections.
3. Sweep global thresholds.
4. Sweep per-class thresholds.
5. Sweep max_det values.
6. Score by local approximation:
   0.6 * AP50 + 0.4 * false-positive score
7. Save missed boxes and false positives for Phase 4 data planning.
```

This is not the private validator. It is the local decision instrument for
choosing checkpoints and thresholds before deployment.

Current best starter-pack threshold configs:

```text
Car-wash:
  checkpoint: runs/car_wash/yolo11n_starter/weights/best.pt
  max_det: 20
  thresholds:
    broom: 0.5
    drainage gate: 0.1
    nozzle: 0.1
    track: 0.1

Beverage:
  checkpoint: runs/beverage/yolo11n_starter/weights/best.pt
  max_det: 50
  thresholds:
    cup: 0.2
    bottle: 0.1
    can: 0.1
```

Use these as starting points only. They must be re-swept after Phase 4 data.

## Required Preflight

Before choosing an element or deploying a new model:

```bash
python3 score_miner_project/scripts/competition_radar.py --overview --leaderboards --top 8
```

Record:

```text
leader score
top-5 cutoff
participants
target score
gap-to-target
whether the element is already over target
```

## Deployment Path

Current TurboVision public miner docs use:

```bash
sv -v deploy-os-miner --model-path <path_to_model_assets> --element-id <element_id>
```

Useful dry-run flags:

```text
--no-deploy
--no-commit
--revision <sha-or-branch>
```

Required environment:

```text
BITTENSOR_WALLET_COLD
BITTENSOR_WALLET_HOT
CHUTES_API_KEY
HF_USER
HF_TOKEN
SCOREVISION_NETUID
```

## Competitive Engineering Rules

Reusable from the old football plan:

```text
closed score loop
replay/failure archive
class mapping guard
memory/latency checks
deployment reproducibility
competitor-output inspection
threshold optimization by actual Score-style metric
```

Do not reuse football-only pieces for public Detect:

```text
TrackLab live stack
sn-gamestate live stack
team color
pitch keypoints
player/ball role logic
video identity state
large RF-DETR/DEIM/D-FINE models that cannot fit the 30MB repo cap
```

## Offline Teacher Use

Foundation/teacher models are useful for labeling and review, not first live inference:

```text
SAM3: promptable concept segmentation / pseudo-labeling
GroundingDINO: open-vocabulary pseudo boxes
Supervision: normalize/visualize detections and annotations
CVAT/FiftyOne: manual review and hard-negative mining
```

Teacher labels must be reviewed against Score starter/proof style before training. Do not let a teacher model create a large noisy dataset without human spot checks.

## Phase 4 Data Priorities

Build data from measured failures, not from random volume.

```text
Beverage hard negatives:
  jars, glassware, hands, labels, boxes, trash, plates, shelves, reflections

Car-wash hard negatives:
  hose, pipe, floor seams, tires, shadows, water reflections, wall stripes, rail-like lines
```

Synthetic data rule:

```text
diversity first:
  camera angle
  object scale
  lighting
  clutter
  occlusion
  compression
  realistic placement

volume second
```
