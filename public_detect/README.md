# SN44 Public Detect Workspace

This folder is the clean workspace for current SN44 public Detect tasks:

```text
manak0/Detect-car-wash
manak0/Detect-beverage-detect
manak0/Detect-crime
manak0/Detect-fire
```

Do not mix this with the old football/player-detection work. Public Detect is a different problem:

```text
small object detector
full Hugging Face repo <= 30MB
class order from manifest
Score starterPack/proof frames as validation anchor
Chutes deployment
false-positive control
```

## Current Execution Target

```text
1. Car-wash
2. Beverage
3. Next NEW public element from radar
```

## 2026 GPU Setup

Use `uv` for rented GPU machines. This keeps setup reproducible and avoids
ad-hoc package installs:

```bash
cd score_miner_project/public_detect
./scripts/setup_gpu_env.sh
```

The setup script uses:

```text
uv project environment
Python 3.12
UV_TORCH_BACKEND=auto
official PyTorch CUDA wheel selection
latest compatible Ultralytics from the lock/solve
CUDA + YOLO weight availability check before training
```

Current lock, resolved 2026-05-25:

```text
torch 2.12.0
torchvision 0.27.0
ultralytics 8.4.54
```

Run Phase 1B baselines with:

```bash
./scripts/run_phase1b_baselines.sh
```

If a 16GB GPU runs out of memory:

```bash
TRAIN_BATCH=2 TRAIN_IMGSZ=768 ./scripts/run_phase1b_baselines.sh
```

Before starting or switching elements, run:

```bash
python3 score_miner_project/scripts/competition_radar.py --overview --leaderboards --top 8
```

## Model Bake-Off

Start with measured baselines, not model hype:

```text
YOLO11n
YOLO26n
YOLO26s FP16/ONNX if full repo <= 30MB
YOLOv8n control
RT-DETR tiny only after baseline, only if exported repo <= 30MB
```

Any model family is allowed. The protocol does not require YOLO. The gate is:

```text
correct output schema
class IDs in manifest order
full Hugging Face repo revision <= 30MB
latency/health pass
better Score-style validation and live score
```

## Data Priority

Use Score data first:

```text
1. Score starterPack images + annotations
2. Score proof/latest challenge frames
3. Score competitor prediction inspection
4. Diverse synthetic images matched to Score style
5. Public datasets only as filler
```

Do not pick checkpoints by synthetic mAP alone.

## Competitive Loop

The reusable idea from the old football architecture is the score loop, not the
football-specific tracking stack:

```text
radar -> starter/proof download -> train baseline -> export/size gate
-> local Score-style validation -> threshold sweep -> deploy
-> live score + competitor prediction mining -> hard negatives
-> retrain -> redeploy
```

Keep each iteration reproducible:

```text
model weights
element config
training data manifest
threshold config
size-gate report
local score report
live score snapshot
```

## Win Path

Do not skip the Score-style scorer. Normal YOLO mAP is not enough for SN44
public Detect because false positives matter. The winning loop is:

```text
1. Train YOLO11n / YOLO26n baselines.
2. Build local Score-style validation:
   map50 approximation + false-positive pressure.
3. Pull starter/proof/latest challenge frames.
4. Sweep confidence, per-class thresholds, max_det, and imgsz.
5. Add Phase 4 data only from evidence:
   missed objects, duplicate boxes, wrong classes, background false positives.
6. Retrain.
7. Re-sweep thresholds.
8. Export and pass the 30MB full-HF-repo gate.
9. Deploy.
```

For winning, Phase 3 and Phase 4 are not optional. A fast baseline deploy can
skip them, but a top-3 attempt should not.

## Folder Map

```text
public_detect/
  README.md
  TODO.md
  docs/
    research_sources.md
    score_rules.md
  configs/
    elements/
    training/
    export/
  data/
    raw/
    starter_packs/
    proof_frames/
    yolo/
    synthetic/
    hard_negatives/
  runs/
    car_wash/
    beverage/
  models/
    car_wash/
    beverage/
  reports/
    score_sweeps/
    size_gates/
    live_scores/
  src/
    public_detect/
  scripts/
```

Implementation rule:

```text
Before adding code for any TODO, do a current-doc/source check and record the source in docs/research_sources.md.
```
