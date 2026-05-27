# Current Status And GPU Runbook

Date: 2026-05-27

This is the handoff document for SN44 public Detect Beverage/Car-wash work. It
separates what is proven locally from what still needs real Phase 4 data.

## What Is Proven Locally

The local RTX 3070 machine has proven the plumbing:

```text
starter download works
YOLO dataset conversion works
local train works
Score-style threshold sweep works
SAHI sweep path works
YOLO -> ONNX export works
30MB size gate works
minimal ONNX Runtime miner.py works
deploy repo smoke test works
```

This means a rental GPU session should not be spent debugging setup. It should
be spent collecting/correcting data and training serious Phase 4 models.

## Important Score Interpretation

The local score is an approximation:

```text
local_score = 0.6 * local_map50 + 0.4 * local_false_positive
```

The real Score validator has extra live details/gates. A Manako evaluation
payload showed:

```text
map50 weighted_score
false_positive weighted_score
total_weighted
total_weighted_and_gated
```

So local score is not guaranteed to equal console score. It is still useful for
comparing our checkpoints before deploy.

Do not overread the exact number. Do trust the error counts:

```text
TP, FP, FN, recall, class breakdown
```

If a model has bottle TP = 0 and many misses, the real validator will not make
it a top Beverage miner.

## Best Local Beverage Checkpoint

Best local Beverage checkpoint currently found:

```text
runs/beverage/yolo11n_starter/weights/best.pt
```

Local Score-style audit:

```text
score: 0.4880
map50: 0.2610
fp_score: 0.8286
precision: 0.7857
recall: 0.3826
TP: 44
FP: 12
FN: 71
thresholds: cup 0.4, bottle 0.1, can 0.1
```

Class read:

```text
cup: learned partially
can: learned partially
bottle: not learned, 0 TP / 20 GT
```

This is not deploy-competitive against console Beverage around 74.5%. It is a
baseline. The missing piece is data.

## Why The 50-Epoch Local Starter Run Was Worse

The 50-epoch starter-only model trained on only 7 images. It overfit poorly and
became too quiet:

```text
score: 0.4175
map50: 0.0292
fp_score: 1.0
recall: 0.0435
TP: 5
FP: 0
FN: 110
```

This does not mean the project is dead. It means training longer on 7 images is
not the path.

## Real Target To Beat Beverage 74.5

The path is:

```text
fix bottle
increase recall
preserve false-positive score
```

First serious target:

```text
local score >= 0.60: promising
local score >= 0.70: deploy probe / continue
local score >= 0.75: strong deploy candidate
```

Because local score is approximate, a model near 0.70 may still be worth a live
validator probe. A model near 0.49 with bottle TP = 0 is not.

## Data Sources

Priority order:

```text
1. Score starter labels
2. Score Manako challenge images with manually corrected labels
3. TACO selected cup/bottle/can positives
4. TACO and outside hard negatives
5. Additional Score-like stadium/crowd/CCTV beverage frames
```

Manako warning:

```text
Manako boxes are miner predictions, not ground truth.
Use them as visual hints only.
Do not train on raw Manako labels blindly.
```

Expected useful dataset size for Phase 4 v1:

```text
7 starter images
50-300 corrected Manako frames
300-700 selected TACO positives
100-300 hard negatives
```

This is better than 20,000 wrong-domain images.

## A6000 GPU Setup

Use a machine with:

```text
Ubuntu 22.04/24.04
NVIDIA driver compatible with CUDA 12.8
48GB VRAM
200GB+ disk
```

Setup:

```bash
git clone <YOUR_REPO_URL> Score_miner
cd Score_miner/public_detect

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

sudo apt update
sudo apt install -y ffmpeg git-lfs unzip

uv sync --group inference --group export
PYTHONPATH=src uv run python scripts/check_gpu_env.py
```

If CUDA is not available, stop and fix the GPU environment before downloading
data or training.

## Download Official Starter

```bash
PYTHONPATH=src uv run python scripts/download_starter_pack.py \
  --element-config configs/elements/beverage.yaml
```

Optional car-wash:

```bash
PYTHONPATH=src uv run python scripts/download_starter_pack.py \
  --element-config configs/elements/car_wash.yaml
```

## Download All Beverage Manako Frames

Run the full scan:

```bash
PYTHONPATH=src uv run python scripts/download_manako_challenge_frames.py \
  --output-dir data/proof_frames/manako_challenges_beverage_all \
  --element-filter Detect-beverage \
  --min-score 0.001 \
  --quiet
```

Count:

```bash
find data/proof_frames/manako_challenges_beverage_all/images -type f | wc -l
find data/proof_frames/manako_challenges_beverage_all/overlays -type f | wc -l
```

Make review sheet:

```bash
PYTHONPATH=src uv run python scripts/make_contact_sheet.py \
  --input-dir data/proof_frames/manako_challenges_beverage_all/overlays \
  --output reports/source_reviews/manako_beverage_all_sheet.jpg \
  --limit 300 \
  --thumb-width 260 \
  --columns 5
```

Review and keep Score-style Beverage images: stadium/crowd/CCTV/rain/night,
people holding cups/cans/bottles. Reject wrong task or wrong style.

## Download And Ingest TACO

```bash
mkdir -p data/external
cd data/external
git clone https://github.com/pedropro/TACO.git
cd TACO
PYTHONPATH=/home/ubuntu/Score_miner/public_detect/src uv run python download.py
cd /home/ubuntu/Score_miner/public_detect
```

If the repo path differs, adjust `/home/ubuntu/Score_miner/public_detect`.

Ingest:

```bash
PYTHONPATH=src uv run python scripts/ingest_coco_source.py \
  --config configs/data_sources/beverage_taco.yaml \
  --coco-json data/external/TACO/data/annotations.json \
  --image-root data/external/TACO/data \
  --output-dir data/yolo_candidates/beverage_taco_v1
```

Check class counts:

```bash
find data/yolo_candidates/beverage_taco_v1/images/train -type f | wc -l
find data/yolo_candidates/beverage_taco_v1/labels/train -type f | wc -l

awk '{c[$1]++} END {print "cup", c[0]+0; print "bottle", c[1]+0; print "can", c[2]+0}' \
  data/yolo_candidates/beverage_taco_v1/labels/train/*.txt

find data/yolo_candidates/beverage_taco_v1/labels/train -type f -size 0 | wc -l
```

## Phase 4 Dataset Build

The operational missing piece is a polished merge/review script. Until it is
implemented, build the dataset manually or add a script that creates:

```text
data/yolo/beverage_phase4_v1/
  images/train
  labels/train
  images/val
  labels/val
  data.yaml
  manifest.json
```

Class order must be:

```text
0 cup
1 bottle
2 can
```

Use absolute path in `data.yaml`:

```yaml
path: /home/ubuntu/Score_miner/public_detect/data/yolo/beverage_phase4_v1
train: images/train
val: images/val
nc: 3
names:
  0: cup
  1: bottle
  2: can
```

Validation should include trusted starter images and a held-out set of corrected
Manako/TACO data. Do not judge on synthetic/TACO-only validation.

## Train Serious Beverage Models

Create a Phase 4 config:

```bash
cp configs/training/beverage_yolo11n.yaml configs/training/beverage_yolo11n_phase4.yaml
```

Edit:

```yaml
data: data/yolo/beverage_phase4_v1/data.yaml
model: yolo11n.pt
project: runs/beverage
name: yolo11n_phase4_v1
```

Train 960:

```bash
PYTHONPATH=src uv run python scripts/train_baseline.py \
  --config configs/training/beverage_yolo11n_phase4.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 960 \
  --name-suffix 960
```

Train 1280:

```bash
PYTHONPATH=src uv run python scripts/train_baseline.py \
  --config configs/training/beverage_yolo11n_phase4.yaml \
  --epochs 120 \
  --batch 8 \
  --imgsz 1280 \
  --name-suffix 1280
```

Try YOLO11s only after YOLO11n improves and export size still passes 30MB.

## Score And Compare

Single-pass:

```bash
PYTHONPATH=src uv run python scripts/score_threshold_sweep.py \
  --model runs/beverage/<RUN_NAME>/weights/best.pt \
  --data data/yolo/beverage_phase4_v1/data.yaml \
  --name beverage_phase4_v1_single \
  --base-conf 0.001 \
  --per-class
```

SAHI:

```bash
PYTHONPATH=src uv run python scripts/score_threshold_sweep.py \
  --model runs/beverage/<RUN_NAME>/weights/best.pt \
  --data data/yolo/beverage_phase4_v1/data.yaml \
  --name beverage_phase4_v1_sahi \
  --prediction-mode sahi \
  --device cuda:0 \
  --base-conf 0.001 \
  --sahi-slice-height 640 \
  --sahi-slice-width 640 \
  --sahi-overlap 0.25 \
  --per-class
```

Read:

```bash
cat reports/score_sweeps/beverage_phase4_v1_single/summary.json
cat reports/score_sweeps/beverage_phase4_v1_sahi/summary.json
```

Choose the model by:

```text
score
recall
class breakdown, especially bottle
fp_score
```

## Export And Build Miner

Export:

```bash
PYTHONPATH=src uv run python scripts/export_yolo_onnx.py \
  --model runs/beverage/<RUN_NAME>/weights/best.pt \
  --output-dir artifacts/deploy_repos/beverage_phase4_v1 \
  --imgsz 960
```

Build miner repo with best thresholds from sweep:

```bash
PYTHONPATH=src uv run python scripts/build_deploy_repo.py \
  --weights artifacts/deploy_repos/beverage_phase4_v1/weights.onnx \
  --element-config configs/elements/beverage.yaml \
  --output-dir artifacts/deploy_repos/beverage_phase4_v1_miner \
  --input-size 960 \
  --conf <cup_conf>,<bottle_conf>,<can_conf> \
  --max-det <best_max_det>
```

Smoke:

```bash
PYTHONPATH=src uv run python scripts/smoke_deploy_miner.py \
  --repo artifacts/deploy_repos/beverage_phase4_v1_miner \
  --images data/yolo/beverage_phase4_v1/images/val \
  --limit 5
```

Size:

```bash
PYTHONPATH=src uv run python scripts/check_repo_size.py \
  artifacts/deploy_repos/beverage_phase4_v1_miner
```

Remove pycache before upload:

```bash
find artifacts/deploy_repos/beverage_phase4_v1_miner -type d -name "__pycache__" -prune -exec rm -rf {} +
```

## Deployment Decision

Do not deploy only because training finished.

Deploy if:

```text
local score is near or above 0.70
bottle class is no longer dead
fp_score remains acceptable, ideally > 0.80
deploy repo is <= 30MB
miner smoke passes
```

If local score is 0.55-0.65, consider a live validator probe only if the class
breakdown looks healthy and you want to measure the real validator gap.

If local score stays below 0.55 after corrected Manako + TACO, Beverage is not
ready; switch to Car-wash or a newer element.

## What To Copy Back

Copy reports and deploy artifacts:

```bash
rsync -avP ubuntu@GPU_IP:~/Score_miner/public_detect/runs/beverage/ \
  /home/sina/projects/validator_improve/score_miner_project/public_detect/runs/beverage/

rsync -avP ubuntu@GPU_IP:~/Score_miner/public_detect/reports/score_sweeps/ \
  /home/sina/projects/validator_improve/score_miner_project/public_detect/reports/score_sweeps/

rsync -avP ubuntu@GPU_IP:~/Score_miner/public_detect/artifacts/deploy_repos/ \
  /home/sina/projects/validator_improve/score_miner_project/public_detect/artifacts/deploy_repos/
```

Do not copy huge raw datasets unless needed. Copy manifests, sheets, summaries,
best checkpoints, and deploy repos.

