# What Other Miners Are Probably Doing

Date: 2026-05-28  
Element focus: `manak0/Detect-beverage-detect`  
Goal: understand the live iteration loop used by competitive public ScoreVision miners and turn it into a practical runbook.

This document is intentionally direct. The main lesson is that top miners are usually not winning from one static public dataset. They are running a feedback loop:

```text
train -> deploy -> observe public artifacts -> harvest proxy data -> fine-tune -> redeploy
```

They are not getting hidden ground-truth labels directly. They are exploiting the public surfaces that the competition exposes: current leaderboard, challenge logs, miner repos, miner predictions, public/proof frames, and score traces.

## Current Beverage Scoring Reality

The current public manifest for Beverage uses only two pillars:

```yaml
metrics:
  pillars:
    map50: 0.6
    false_positive: 0.4
```

So the practical score is:

```text
score = 0.6 * map50 + 0.4 * false_positive
```

TurboVision's object metric computes:

```text
false_positive = max(0, 1 - (false_positives_per_image / 10))
false_positives_per_image = global_fp / number_of_scored_frames
```

For Beverage, `precision`, `recall`, `count`, and `iou` are useful diagnostics, but they are not currently weighted into the manifest score. They matter indirectly because bad precision creates false positives and bad recall lowers AP.

Important local implication:

```text
score_threshold_sweep.py is closer to the actual Beverage manifest score
simulate_validator_score.py is a diagnostic tool
```

## Why Other Miners Improve During A Live Race

The live competition is not a one-shot Kaggle-style offline benchmark. It is a repeated public-miner system.

The loop:

1. Train a model on available data.
2. Deploy a miner artifact to Hugging Face / Chutes.
3. Validator sends hidden challenges.
4. Public console / Manako / result surfaces expose partial evaluation artifacts.
5. Miner harvests those artifacts.
6. Miner creates proxy labels from strong predictions and teacher models.
7. Miner fine-tunes on a better Score-distribution dataset.
8. Miner redeploys a new revision.
9. Repeat.

The key is not that they receive hidden labels. The key is that the hidden challenge distribution leaks enough structure over time:

```text
which image style appears
which object classes appear often
what high-scoring miners predict
which thresholds and max_det choices work
which model input size top miners use
which HF repos/revisions are active
```

## Public Surfaces To Harvest

### 1. Console Element API

Endpoint pattern:

```text
https://console.scorevision.io/api/v2/elements/<URL_ENCODED_ELEMENT_ID>
```

For Beverage:

```text
https://console.scorevision.io/api/v2/elements/manak0%2FDetect-beverage-detect
```

Useful fields:

```text
element.currentScore
element.baselineScore
element.targetScore
history
leaderboard
challengeDetails
latestAnnotatedChallenge
starterPack
```

What it gives:

```text
leaderboard miner HF repos
leaderboard revisions
recent challenge scores
per-challenge miner score variance
starter pack URLs and labels
sometimes latest annotated/proof challenge info
```

Why it matters:

```text
leaderboard repos reveal inference code and thresholds
challengeDetails reveal who scored well on what
history reveals how stable the leader is
starterPack is the cleanest public labeled anchor
```

### 2. Manako Public Index

Endpoint pattern:

```text
https://turbo.scoredata.me/manako/index.json
```

This index is not a clean dataset. It is a public result/evaluation index. Some refs lead to evaluation JSONs; some evaluation JSONs point to response payloads; some payloads expose image URLs and predictions.

What we learned:

```text
naive direct frame download may produce only 1 or 35 unique frames
the index is dominated by repeated eval refs and duplicated challenge IDs
newer miners may be stored under paths not reached by the simple downloader
harvesting all evaluation payloads can produce more pseudo-labeled Score-style frames
```

Local scripts:

```bash
PYTHONPATH=src uv run python scripts/download_manako_challenge_frames.py
PYTHONPATH=src uv run python scripts/scrape_manako_fast.py
PYTHONPATH=src uv run python scripts/harvest_all_manako.py
PYTHONPATH=src uv run python scripts/probe_manako_fresh.py
PYTHONPATH=src uv run python scripts/probe_via_winner.py
```

### 3. Hugging Face Miner Repos

Top miners typically publish deploy repos with:

```text
miner.py
weights.onnx or model files
chute_config.yml
inference thresholds
input size
max_det
postprocessing
sometimes TTA or NMS tricks
```

What to extract:

```text
input size
confidence threshold
iou threshold
max_det
letterbox / resize behavior
class order
ONNX runtime settings
TTA / flip / SAHI usage
model size
latency assumptions
```

Do not blindly copy labels from a miner. A high-scoring miner can still make systematic errors. Use their predictions as proxy labels only when:

```text
their challenge score was high
their confidence is high
multiple miners agree
or a teacher/manual review confirms the box
```

### 4. Chutes Deployment Surface

The public miner flow is usually:

```text
HF repo stores model artifact and miner.py
Chutes runs the miner service
validator calls the Chutes endpoint
miner returns frames with boxes/keypoints
```

Practical implication:

```text
deploy must be small, deterministic, fast enough, and schema-correct
```

For this Beverage element:

```text
max model repo size: 30 MB
latency budget is generous enough for careful postprocessing, but not infinite
ONNX export size must be checked before deployment
```

## Why Proxy Labels Work

Score hidden data is distribution-specific. Generic datasets like TACO or Open Images help class diversity, but they do not fully match the target domain.

Proxy labels help because they are drawn from Score-like frames:

```text
same image resolution/style
same class definitions
same object size distribution
same lighting/noise/camera style
same confusing background objects
```

Even if pseudo-labels are imperfect, a moderately clean Score-distribution pseudo-label set can beat a large clean wrong-domain dataset.

The key is filtering.

Good proxy label sources:

```text
high-scoring top-miner predictions
multiple miner consensus
winner-style predictions above confidence threshold
SAM3 / GroundingDINO / YOLOWorld teacher boxes on Score-like images
manual corrections on a small high-value subset
```

Bad proxy label sources:

```text
low-scoring miner predictions
all predictions at very low confidence
single-source labels with no review
product photos
trash/litter data over-weighted into validation
wrong class mapping
duplicates leaking train into val
```

## Tooling Stack In 2026

### YOLO / Ultralytics

Use for final deployable detector.

Recommended deployment families:

```text
YOLO11n / YOLO26n / small enough YOLO26s only if ONNX < 30 MB
```

Main constraints:

```text
30 MB repo/model limit
latency p95 limit
small objects require high imgsz or tiled inference
class order must be exactly: cup, bottle, can
```

### YOLOWorld

Use for fast open-vocabulary pseudo-labeling, not as final ground truth.

Prompt/classes:

```text
cup,bottle,can
```

Risk:

```text
low conf creates many false labels
class confusion is common
can may be under-detected
```

Use thresholds:

```text
0.10 for recall exploration
0.25-0.35 for cleaner pseudo-labels
```

### SAM3

SAM3 is useful because it supports concept-prompt segmentation: "cup", "bottle", "can". In 2026, Ultralytics documents SAM3 concept segmentation and text-prompt workflows.

Use case:

```text
teacher-label Score-like frames
turn masks into boxes
review hard examples
generate cleaner pseudo-labels than YOLOWorld in some scenes
```

Risk:

```text
requires newer ultralytics than this repo currently pins
heavier than YOLOWorld
not automatically equal to hidden validator labels
must be reviewed
```

### GroundingDINO

Use for open-set box proposals:

```text
"cup . bottle . can ."
```

Good for:

```text
teacher labels
finding missed classes
cross-checking YOLOWorld
```

Risk:

```text
slower setup
threshold-sensitive
still needs manual/visual review
```

### SAHI

Use for small-object inference after the model is decent.

Why:

```text
Score objects are often small
tiled inference increases apparent object scale
latency budget may allow extra passes
```

Risk:

```text
SAHI increases false positives if the base model is noisy
must threshold-sweep with SAHI separately
```

## Data Source Ranking For Beverage

Best to worst:

1. Score starter labels.
2. Manako / Score challenge frames with reviewed labels.
3. High-scoring miner prediction proxies on Score frames.
4. Multi-miner consensus labels on Score frames.
5. SAM3 / GroundingDINO / YOLOWorld labels on Score-like frames, reviewed.
6. Person-holding-drink Roboflow-style datasets.
7. Open Images filtered for person + bottle/cup/can.
8. Beverage Containers / cans / cups contextual datasets.
9. TACO, as can/bottle/cup variety plus hard negatives.
10. Product/studio photos, only as tiny filler if they help a class and do not hurt Score-style validation.

Do not let wrong-domain data dominate.

Good target mix for Beverage Iter2/Iter3:

```text
starter: always included
winnerproxy: 2x-4x train oversample
allminers: 1x-3x train oversample
Manako reviewed: 2x-4x if corrected
OpenImages / person-holding: capped
TACO / drinkwaste: capped, mainly for can/hard negatives
```

## Current Local State

Known local scripts:

```bash
scripts/build_phase4_dataset.py
scripts/harvest_all_manako.py
scripts/score_winner_style.py
scripts/score_threshold_sweep.py
scripts/simulate_validator_score.py
scripts/export_yolo_onnx.py
scripts/build_deploy_repo.py
scripts/smoke_deploy_miner.py
scripts/check_repo_size.py
scripts/iter2_rebuild_and_warmstart.sh
```

`build_phase4_dataset.py` supports:

```text
--source tag:path
--source tag:path:cap
--source tag:path::3x
--source tag:path:800:2x
```

The multiplier duplicates train copies only. Validation stays single-copy.

This is important:

```text
oversample Score-distribution sources
do not duplicate val samples
do not let trash/product data dominate
```

Known harvested Beverage proxy result from the other session:

```text
beverage_all_miners_v1:
  images_train: 447
  labels: 447
  total boxes: 5537
  cup: 2509
  bottle: 1174
  can: 1854
```

This is valuable because it fixes the earlier can starvation.

## Interpreting The Current YOLO26s Training

The running command:

```bash
PYTHONPATH=src uv run python scripts/train_baseline.py \
  --config configs/training/beverage_yolo26s_phase4_winner_v1.yaml \
  --epochs 120 --batch 2 --imgsz 1280 --name-suffix local
```

This is an Iter1 high-resolution model. It should be treated as:

```text
general beverage + some proxy data model
not final proof of live win
```

Do not stop it unless it is clearly broken.

After finish:

```bash
find runs/beverage -path "*/weights/best.pt" -printf "%T@ %p\n" | sort -n | tail -10
```

Then probe the exact `best.pt`.

## Iter1 Finish Checklist

### 1. Confirm model exists

```bash
test -f runs/beverage/yolo26s_phase4_winner_v1_local/weights/best.pt
```

### 2. Check ONNX size before trusting YOLO26s

```bash
UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
PYTHONPATH=src uv run python scripts/export_yolo_onnx.py \
  --model runs/beverage/yolo26s_phase4_winner_v1_local/weights/best.pt \
  --output-dir artifacts/deploy_repos/beverage_yolo26s_iter1_probe \
  --imgsz 1280
```

If this fails the 30 MB limit, keep YOLO26s as a teacher/proxy model and deploy a smaller YOLO11n/YOLO26n model.

### 3. Score with winner-style inference

```bash
UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
PYTHONPATH=src uv run python scripts/score_winner_style.py \
  --model runs/beverage/yolo26s_phase4_winner_v1_local/weights/best.pt \
  --data data/yolo_candidates/beverage_winner_proxy_v2/data.yaml \
  --imgsz 1280
```

```bash
UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
PYTHONPATH=src uv run python scripts/score_winner_style.py \
  --model runs/beverage/yolo26s_phase4_winner_v1_local/weights/best.pt \
  --data data/yolo_candidates/beverage_all_miners_v1/data.yaml \
  --imgsz 1280
```

### 4. Score with manifest formula on starter

```bash
UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
PYTHONPATH=src uv run python scripts/score_threshold_sweep.py \
  --model runs/beverage/yolo26s_phase4_winner_v1_local/weights/best.pt \
  --data data/yolo/beverage_starter/data.yaml \
  --name beverage_yolo26s_iter1_on_starter \
  --base-conf 0.001 \
  --per-class
```

### 5. Score with manifest formula on external/proxy val

```bash
UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
PYTHONPATH=src uv run python scripts/score_threshold_sweep.py \
  --model runs/beverage/yolo26s_phase4_winner_v1_local/weights/best.pt \
  --data data/yolo/beverage_phase4_external_v1/data.yaml \
  --name beverage_yolo26s_iter1_on_external_val \
  --base-conf 0.001 \
  --per-class
```

## Decision Thresholds

Use this table for Beverage:

```text
starter/proxy score < 0.50:
  not deployable unless you need a very rough probe

0.50-0.58:
  maybe earns small emissions; deploy only if export is easy and no better model exists

0.58-0.65:
  worth a live probe; unlikely to beat leader but useful for collecting your own traces

0.65-0.72:
  serious top-5/top-3 candidate depending on live distribution

0.72+:
  deploy immediately; possible top-1/top-3 contender
```

Remember:

```text
external-val score can overestimate live if the val split is from the same downloaded data
starter score can underestimate live because starter has only 7 images
winnerproxy/allminers proxy scores are closer to hidden style but labels are not truth
```

The best signal is agreement:

```text
starter good
winnerproxy good
allminers good
external val good
ONNX fits size
latency ok
```

## Iter2 Warm-Start

If Iter1 is not deployable or only barely deployable, run:

```bash
bash scripts/iter2_rebuild_and_warmstart.sh
```

That script:

```text
rebuilds data/yolo/beverage_phase4_iter2_v1
oversamples winnerproxy 3x
oversamples allminers 2x
caps drinkwaste at 800
drops/caps the most wrong-domain data
warm-starts from Iter1 best.pt
trains 40 epochs at 1280
probes winnerproxy and allminers
```

Why this works:

```text
Iter1 learns general cup/bottle/can features
Iter2 shifts the model toward Score-distribution proxy labels
warm-start avoids relearning from scratch
```

## How To Build Iteration Datasets Manually

Score-distribution-heavy:

```bash
UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
PYTHONPATH=src uv run python scripts/build_phase4_dataset.py \
  --element-config configs/elements/beverage.yaml \
  --output-dir data/yolo/beverage_phase4_iter2_manual \
  --source starter:data/yolo/beverage_starter \
  --source winnerproxy:data/yolo_candidates/beverage_winner_proxy_v2::3x \
  --source allminers:data/yolo_candidates/beverage_all_miners_v1::2x \
  --source manako:data/yolo_candidates/beverage_manako_autolabeled \
  --source oiv7:data/yolo_candidates/beverage_oiv7_v1 \
  --source drinkwaste:data/yolo_candidates/beverage_drinkwaste_v1:800 \
  --source smokedrink:data/yolo_candidates/beverage_smokedrink_v1 \
  --val-fraction 0.10
```

More conservative, less pseudo-label risk:

```bash
UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
PYTHONPATH=src uv run python scripts/build_phase4_dataset.py \
  --element-config configs/elements/beverage.yaml \
  --output-dir data/yolo/beverage_phase4_iter2_conservative \
  --source starter:data/yolo/beverage_starter \
  --source winnerproxy:data/yolo_candidates/beverage_winner_proxy_v2::2x \
  --source allminers:data/yolo_candidates/beverage_all_miners_v1 \
  --source oiv7:data/yolo_candidates/beverage_oiv7_v1 \
  --source drinkwaste:data/yolo_candidates/beverage_drinkwaste_v1:500 \
  --source smokedrink:data/yolo_candidates/beverage_smokedrink_v1 \
  --val-fraction 0.10
```

## Training Strategy

For local 3070:

```text
YOLO11n / YOLO26n:
  imgsz 960-1280
  batch 4-8 if memory allows

YOLO26s:
  imgsz 1280
  batch 2
  check export size
```

For A6000:

```text
YOLO11n/YOLO26n:
  imgsz 1280-1536
  batch 16-32

YOLO26s:
  imgsz 1280-1536
  batch 8-16
  maybe teacher model if ONNX too big
```

GPU type does not directly change score. It changes:

```text
max image size
batch stability
experiment speed
teacher-labeling throughput
ability to run many iterations
```

Data quality and thresholding dominate.

## Threshold Strategy

For Beverage, the validator rewards:

```text
high mAP50
low false positives
```

Thresholds are part of the model. Do not deploy raw training defaults.

Always run:

```bash
PYTHONPATH=src uv run python scripts/score_threshold_sweep.py \
  --model <best.pt> \
  --data <data.yaml> \
  --name <run_name> \
  --base-conf 0.001 \
  --per-class
```

Likely Beverage pattern:

```text
cup threshold: medium/high
bottle threshold: medium/high
can threshold: lower
max_det: 20-100 depending on crowd density
```

If SAHI is used:

```bash
PYTHONPATH=src uv run python scripts/score_threshold_sweep.py \
  --model <best.pt> \
  --data <data.yaml> \
  --name <run_name>_sahi \
  --prediction-mode sahi \
  --device cuda:0 \
  --base-conf 0.001 \
  --sahi-slice-height 640 \
  --sahi-slice-width 640 \
  --sahi-overlap 0.25 \
  --per-class
```

Only keep SAHI if it improves manifest score, not just recall.

## Deploy Loop

When a candidate is good enough:

1. Export ONNX.
2. Build deploy repo.
3. Smoke test.
4. Size check.
5. Push to HF.
6. Register / update miner revision.
7. Watch console score and challengeDetails.
8. Harvest your own public traces.
9. Iter3.

Export:

```bash
UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
PYTHONPATH=src uv run python scripts/export_yolo_onnx.py \
  --model <best.pt> \
  --output-dir artifacts/deploy_repos/beverage_candidate \
  --imgsz 1280
```

Build:

```bash
UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
PYTHONPATH=src uv run python scripts/build_deploy_repo.py \
  --weights artifacts/deploy_repos/beverage_candidate/weights.onnx \
  --element-config configs/elements/beverage.yaml \
  --output-dir artifacts/deploy_repos/beverage_candidate_miner \
  --input-size 1280 \
  --conf <thresholds_from_sweep> \
  --max-det <best_max_det>
```

Smoke:

```bash
UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
PYTHONPATH=src uv run python scripts/smoke_deploy_miner.py \
  --repo artifacts/deploy_repos/beverage_candidate_miner \
  --images data/yolo/beverage_starter/images/train \
  --limit 5
```

Size:

```bash
UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
PYTHONPATH=src uv run python scripts/check_repo_size.py \
  artifacts/deploy_repos/beverage_candidate_miner
```

## What To Log Every Iteration

Create a simple table per iteration:

```text
iteration
model family
imgsz
batch
train sources
source caps/multipliers
epochs
best epoch
external mAP50
starter sweep score
winnerproxy score
allminers score
ONNX size
deploy thresholds
live score after deployment
```

Without this, you will not know which source helped.

## Failure Modes

### External val looks great, live bad

Cause:

```text
validation is same distribution as training download
hidden Score images differ
wrong-domain overfit
```

Fix:

```text
raise Score-distribution proxy weight
cap trash/product data
use starter/proxy validation for decisions
```

### High recall, bad false_positive

Cause:

```text
thresholds too low
wrong-domain labels teach lookalikes
SAHI overdetects
```

Fix:

```text
per-class thresholds
hard negatives
reduce noisy pseudo labels
manual review of false positives
```

### Can is dead

Cause:

```text
not enough can labels
teacher misses cans
class confusion with cup/bottle
```

Fix:

```text
allminers can-rich proxy
TACO capped for can
Roboflow beverage containers
lower can threshold
manual can labels on Score-like images
```

### Bottle is weak

Cause:

```text
bottles vary heavily
tiny handheld bottles
confusion with cups/cans/reflections
```

Fix:

```text
person-holding bottle datasets
high-res training
Score-like pseudo labels
manual correction of bottle boxes
```

### Model cannot deploy

Cause:

```text
ONNX > 30 MB
repo too large
latency too slow
schema mismatch
```

Fix:

```text
deploy YOLO11n/YOLO26n
use YOLO26s as teacher
delete caches/__pycache__
check repo size before push
```

## What Other Miners Likely Know

High-level likely recipe:

```text
1. Use public starter as anchor.
2. Pull console leaderboard and top HF repos.
3. Copy/infer thresholding strategy from top miner.py files.
4. Harvest public Manako / challengeDetails / response traces.
5. Build proxy labels from top miner predictions and teacher models.
6. Train a small deployable YOLO model at high image size.
7. Tune per-class thresholds to maximize 0.6*mAP50 + 0.4*FP.
8. Deploy, wait for score, harvest own public traces.
9. Repeat every day.
```

This is why the top score moves. It is not a static model.

## Practical Next Steps From Current State

Do not interrupt the current YOLO26s run.

When it finishes:

```bash
find runs/beverage -path "*/weights/best.pt" -printf "%T@ %p\n" | sort -n | tail -10
```

Run:

```bash
UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
PYTHONPATH=src uv run python scripts/score_winner_style.py \
  --model runs/beverage/yolo26s_phase4_winner_v1_local/weights/best.pt \
  --data data/yolo_candidates/beverage_winner_proxy_v2/data.yaml \
  --imgsz 1280
```

Run:

```bash
UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
PYTHONPATH=src uv run python scripts/score_winner_style.py \
  --model runs/beverage/yolo26s_phase4_winner_v1_local/weights/best.pt \
  --data data/yolo_candidates/beverage_all_miners_v1/data.yaml \
  --imgsz 1280
```

If good enough, export and deploy.

If not:

```bash
bash scripts/iter2_rebuild_and_warmstart.sh
```

## Strategic Answer

The other miners are not doing magic. They are compounding feedback:

```text
more Score-distribution data
better proxy labels
better thresholds
faster redeploy cadence
```

The most valuable asset is not TACO or Open Images. It is the harvested Score-distribution proxy set:

```text
winnerproxy
allminers
your own future live traces
```

The fastest path to catch up is:

```text
finish Iter1
probe
deploy if acceptable
run Iter2 with allminers/winnerproxy oversampling
deploy again
harvest your own traces
Iter3
```

## References Checked

Score / TurboVision:

- Score Technologies GitHub organization: https://github.com/score-technologies
- TurboVision / ScoreVision public code in local repo: `turbovision/scorevision/vlm_pipeline/non_vlm_scoring/objects.py`
- TurboVision manifest scoring code in local repo: `turbovision/scorevision/utils/evaluate.py`
- Current local public manifest: `score_miner_project/current_manifest.yaml`
- ScoreVision console element API pattern: `https://console.scorevision.io/api/v2/elements/<element_id>`
- Public Manako index pattern: `https://turbo.scoredata.me/manako/index.json`

Deployment / Chutes:

- Chutes miner lifecycle docs: https://deepwiki.com/chutesai/chutes-miner/5.2-chute-deployment-lifecycle
- Chutes miner repo: https://github.com/rayonlabs/chutes-miner
- Chutes API repo: https://github.com/rayonlabs/chutes-api

Teacher / labeling tools:

- Ultralytics YOLO-World docs: https://docs.ultralytics.com/models/yolo-world/
- Ultralytics SAM3 docs: https://docs.ultralytics.com/models/sam-3/
- GroundingDINO paper: https://arxiv.org/abs/2303.05499
- Open Images V7 via FiftyOne: https://docs.voxel51.com/dataset_zoo/datasets/open_images_v7.html

Important caveat:

```text
Public artifacts are proxy data, not hidden ground truth.
Use them aggressively, but filter and validate because pseudo-label errors compound.
```
