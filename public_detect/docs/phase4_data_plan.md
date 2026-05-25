# Phase 4 Data Plan

Date checked: 2026-05-26

This phase is the data-improvement loop after starter-only baselines and Phase 3
Score-style threshold sweeps. The goal is not to create a large generic dataset.
The goal is to add the smallest set of data that improves Score-style validation
on Score-like frames.

## What We Have Now

Current training data:

```text
Car-wash starter: 7 images, 63 boxes
Beverage starter: 7 images, 115 boxes
```

Current best starter-only models:

```text
Car-wash: YOLO11n
Beverage: YOLO11n
```

Corrected local Phase 3 scorer mirrors TurboVision public Detect object scoring:

```text
map50: AP at IoU 0.5
false_positive: max(0, 1 - false_positives_per_image / 10)
total: 0.6 * map50 + 0.4 * false_positive
```

TurboVision references:

```text
turbovision/scorevision/vlm_pipeline/non_vlm_scoring/objects.py
turbovision/scorevision/utils/evaluate.py
turbovision/scorevision/validator/central/open_source/runner.py
```

## Current Phase 3 Results

### Car-wash

```text
checkpoint: runs/car_wash/yolo11n_starter/weights/best.pt
score: 0.7464
map50: 0.6154
false_positive: 0.9429
precision: 0.8947
recall: 0.5397
```

Best starter-pack thresholds:

```text
broom: 0.2
drainage gate: 0.1
nozzle: 0.1
track: 0.1
max_det: 20
```

Class bottlenecks:

```text
nozzle: 8 TP / 1 FP / 22 FN
drainage gate: 4 TP / 1 FP / 4 FN
```

Read:

```text
The model is mostly precise but misses many objects.
The first data target is nozzle recall.
The second target is drainage gate recall.
False positives are not the first problem.
```

### Beverage

```text
checkpoint: runs/beverage/yolo11n_starter/weights/best.pt
score: 0.4880
map50: 0.2610
false_positive: 0.8286
precision: 0.7857
recall: 0.3826
```

Best starter-pack thresholds:

```text
cup: 0.4
bottle: 0.1
can: 0.1
max_det: 20
```

Class bottlenecks:

```text
cup: reasonable but low recall
bottle: 0 TP / 0 FP / 20 FN
can: moderate recall, some false positives
```

Read:

```text
Beverage is not deploy-ready from starter-only training.
Unlike Car-wash, outside data is abundant and useful if filtered correctly.
Bottle examples are the first Beverage data target.
```

## Data Source Priority

Use this order for both tasks.

### Level 1: Score Data

Highest value:

```text
Score starterPack images and annotations
Score latestAnnotatedChallenge/proof frames if available
Score challenge frames where we can legally store/review them
```

Why:

```text
This is the target distribution. Every outside dataset is judged by whether it
improves Score-style validation on Score-like frames.
```

Current location:

```text
data/starter_packs/car_wash/
data/starter_packs/beverage/
data/yolo/car_wash_starter/
data/yolo/beverage_starter/
```

### Level 2: Real Outside Data

Use only if visually close to Score starter/proof frames.

Car-wash:

```text
car-wash tunnel videos
self-service car-wash bay videos
car-wash equipment walkthroughs
spray nozzle / wash nozzle scenes inside car-wash bays
wet floor / drain / drainage gate scenes inside wash bays
```

Beverage:

```text
cluttered real-world beverage scenes
trash/litter scenes with cups, bottles, cans
recycling/waste-sorting scenes
indoor/outdoor messy scenes, not catalog photos
```

### Level 3: Hard Negatives

Hard negatives are images or regions that look similar but should not be
detected as target classes.

Car-wash hard negatives:

```text
hoses that are not nozzles
pipes that are not nozzles
rails that are not tracks
floor seams that are not drainage gates
wet reflections
wall stripes
machinery edges
tires and wheel edges
dark shadows around equipment
```

Beverage hard negatives:

```text
jars
glassware
cartons if not mapped to bottle/cup/can
food containers
cylindrical trash bins
labels/posters that look like cans
hands and shelves around drinks
plastic containers that are not beverage bottles
```

### Level 4: Synthetic Data

Use synthetic data only if it matches Score style.

Good synthetic:

```text
small nozzles placed in car-wash backgrounds
drainage gates under water/reflection
partly occluded cans/cups/bottles in cluttered scenes
varied camera angle, object scale, lighting, blur, compression
```

Bad synthetic:

```text
clean product render
white-background object
repeated near-duplicate object paste
too-perfect boxes
unrealistic scale or lighting
```

Synthetic data must be validated by Phase 3. If it improves only synthetic mAP
and not Score-style validation, remove it.

## Searched Outside Sources

### Beverage Sources

Primary candidates:

```text
Roboflow Universe beverage container datasets
TACO Trash Annotations in Context
Roboflow waste/litter datasets with cup/bottle/can classes
```

Why these are useful:

```text
They include real-world clutter and waste/recycling scenes. This is much closer
to Score Beverage than clean product catalog images.
```

Class mapping:

```text
Score cup:
  disposable cup, paper cup, foam cup, cup-handle

Score bottle:
  plastic bottle, glass bottle, bottle-plastic, bottle-glass, gym bottle

Score can:
  drink can, tin can, food can if visually close enough
```

Important filter:

```text
Use cluttered/context images first.
Avoid clean catalog/studio product images as primary data.
```

### Car-wash Sources

Search result:

```text
Direct labeled car-wash datasets for broom/drainage gate/nozzle/track are scarce.
Keyword searches often return road drains, potholes, washing machines, or metal
washers. These are wrong distribution for Score Car-wash.
```

Primary candidates:

```text
car-wash tunnel/walkthrough videos
self-service wash bay videos
facility/equipment photos
custom frame extraction
teacher-assisted pseudo-labels with manual review
```

Class mapping:

```text
broom:
  push broom, brush/broom in car-wash bay context

drainage gate:
  car-wash floor drain, trench drain, grate in wet bay/tunnel

nozzle:
  spray nozzle, wash nozzle, small nozzle rows, equipment jets

track:
  car-wash floor conveyor/guide track, not generic road lane or train track
```

Important filter:

```text
Reject road-drain datasets unless the images visually match car-wash floors.
Reject product-only nozzle images unless used sparingly for object appearance.
Reject washing-machine or bolt-washer datasets entirely.
```

## Tooling Sources Checked

Roboflow Universe:

```text
Use for candidate discovery, but manually inspect distribution and license.
Universe search is semantic, so keyword matches can be misleading.
```

TACO:

```text
Useful for Beverage classes and hard negatives.
Includes object detection/segmentation labels for real trash scenes.
```

GroundingDINO:

```text
Use as an offline open-vocabulary box teacher for candidate outside images.
Do not deploy it live for public Detect.
```

SAM3:

```text
Use as an offline segmentation/label refinement tool where available.
Do not deploy it live because of size and task constraints.
```

Supervision:

```text
Use for converting, visualizing, and reviewing detections/annotations.
```

FiftyOne or CVAT:

```text
Use if manual review volume becomes too high.
Review is still required; blind pseudo-labels are not accepted as final labels.
```

## Phase 4 Execution Plan

Source manifest:

```text
configs/data_sources/phase4_sources.yaml
```

This file records accepted source types, reject patterns, class-query prompts,
and Beverage public candidates. Use it as the control plane before downloading
or labeling outside data.

### Step 1: Failure Review Exporter

Build a script that reads:

```text
reports/score_sweeps/<run>/diagnostics.json
data/yolo/<element>_starter/data.yaml
reports/score_sweeps/<run>/raw_predictions.json
```

It should output review images:

```text
reports/failure_reviews/<run>/
```

Draw:

```text
green: ground-truth boxes
red: predictions
yellow: missed ground truth
purple: false positives
```

Why first:

```text
We need to see what kind of nozzles/gates/bottles are failing before collecting
outside data. The data plan comes from observed failure modes, not guesses.
```

Status:

```text
DONE
```

Implementation:

```text
src/public_detect/review.py
scripts/export_failure_review.py
```

Commands:

```bash
PYTHONPATH=src python3 scripts/export_failure_review.py \
  --data data/yolo/car_wash_starter/data.yaml \
  --diagnostics reports/score_sweeps/car_wash_yolo11n_starter/diagnostics.json \
  --predictions reports/score_sweeps/car_wash_yolo11n_starter/raw_predictions.json \
  --summary reports/score_sweeps/car_wash_yolo11n_starter/summary.json \
  --name car_wash_yolo11n_starter

PYTHONPATH=src python3 scripts/export_failure_review.py \
  --data data/yolo/beverage_starter/data.yaml \
  --diagnostics reports/score_sweeps/beverage_yolo11n_starter/diagnostics.json \
  --predictions reports/score_sweeps/beverage_yolo11n_starter/raw_predictions.json \
  --summary reports/score_sweeps/beverage_yolo11n_starter/summary.json \
  --name beverage_yolo11n_starter
```

Outputs:

```text
reports/failure_reviews/car_wash_yolo11n_starter/
reports/failure_reviews/beverage_yolo11n_starter/
```

Generated files:

```text
full/*.jpg
crops/miss/<class>/*.jpg
crops/false_positive/<class>/*.jpg
review_items.csv
summary.json
```

Observed from review images:

```text
Car-wash nozzles are tiny wall-mounted jets in low-resolution CCTV frames.
The model sees large broom/track structures but misses many small nozzle boxes.
Car-wash outside data must emphasize small repeated wall/floor jets, not product nozzles.

Beverage frames are crowded bar/interior scenes with handheld objects.
The model misses many bottles/cups and creates can false positives.
Beverage outside data must emphasize crowded handheld drinks, not clean product shots.
```

Failure review summaries:

```text
Car-wash:
  misses: 29
  false positives: 4
  by class:
    nozzle: 22 miss / 1 FP
    drainage gate: 4 miss / 1 FP
    broom: 2 miss / 2 FP
    track: 1 miss / 0 FP

Beverage:
  misses: 71
  false positives: 12
  by class:
    cup: 35 miss / 0 FP
    bottle: 20 miss / 0 FP
    can: 16 miss / 12 FP
```

### Step 2: Car-wash Targeted Data

First target:

```text
nozzle recall
```

Build:

```text
200-500 positive nozzle-heavy images, focused on small CCTV-style jets
100-300 drainage-gate-heavy images, focused on wet floor/trench grates
100-300 hard-negative images, focused on hoses/pipes/rails/seams/reflections
```

Best source order:

```text
1. Score-like car-wash video frames
2. Teacher-labeled car-wash frames, manually reviewed
3. Score-style synthetic for missed nozzle/gate patterns
4. Small amount of generic nozzle/broom/drain data only if validation improves
```

Car-wash video-frame ingestion command:

```bash
PYTHONPATH=src python3 scripts/extract_video_frames.py \
  --video /path/to/car_wash_video.mp4 \
  --output-dir data/raw/car_wash_video_frames/<source_slug> \
  --fps 0.5 \
  --max-frames 500 \
  --prefix <source_slug>
```

Notes:

```text
Use local video files as input.
If downloading web videos, use current yt-dlp + ffmpeg outside this script.
Record the source URL/license before labels are created.
Do not train on extracted frames until they are labeled and reviewed.
```

### Step 3: Beverage Targeted Data

First target:

```text
bottle recall
```

Build:

```text
300-800 bottle-heavy real images in crowded scenes
200-500 cup/can real images in crowded scenes
200-500 hard negatives: jars, glassware, cartons, hands, labels, shelves
```

Best source order:

```text
1. TACO / real waste datasets
2. Roboflow beverage-container datasets
3. Roboflow litter/recycling datasets
4. Synthetic only for gaps, such as occluded cups/cans in clutter
```

Beverage COCO/TACO ingestion commands:

```bash
PYTHONPATH=src python3 scripts/ingest_coco_source.py \
  --config configs/data_sources/beverage_taco.yaml \
  --coco-json /path/to/TACO/data/annotations.json \
  --image-root /path/to/TACO/data \
  --output-dir data/yolo_candidates/beverage_taco_v1
```

For a Roboflow COCO export:

```bash
PYTHONPATH=src python3 scripts/ingest_coco_source.py \
  --config configs/data_sources/beverage_roboflow_containers.yaml \
  --coco-json /path/to/roboflow/export/_annotations.coco.json \
  --image-root /path/to/roboflow/export \
  --output-dir data/yolo_candidates/beverage_roboflow_containers_v1
```

The output is a review-gated YOLO candidate set, not automatically trusted:

```text
data.yaml
images/train/
labels/train/
manifest.json
```

The `manifest.json` records source path, mapped labels, hard-negative labels,
review status, and license note.

### Step 4: Label And Review

Use:

```text
manual boxes for small high-value sets
GroundingDINO/SAM3 teacher labels for bulk proposals
Supervision/FiftyOne/CVAT for review
```

Review requirements:

```text
All Score starter/proof labels must remain trusted.
Teacher labels must be spot-checked.
Wrong-domain or noisy labels are removed before training.
```

Current ingestion implementation:

```text
src/public_detect/ingest.py
scripts/ingest_coco_source.py
scripts/extract_video_frames.py
configs/data_sources/beverage_taco.yaml
configs/data_sources/beverage_roboflow_containers.yaml
configs/data_sources/phase4_sources.yaml
```

### Step 5: Merge Dataset

Create new YOLO datasets:

```text
data/yolo/car_wash_phase4_v1/
data/yolo/beverage_phase4_v1/
```

Keep manifests:

```text
data/manifests/car_wash_phase4_v1.json
data/manifests/beverage_phase4_v1.json
```

Each row should record:

```text
source
image path
label path
classes present
review status
license/source URL where applicable
```

### Step 6: Retrain

Retrain:

```text
YOLO11n first
YOLO11s only if final export/repo can still pass 30MB
YOLO26 stays parked until YOLO11n Phase 4 is measured
```

Do not judge by training mAP only.

### Step 7: Re-run Phase 3

After retrain:

```bash
PYTHONPATH=src python3 scripts/score_threshold_sweep.py \
  --model runs/car_wash/<phase4_run>/weights/best.pt \
  --data data/yolo/car_wash_starter/data.yaml \
  --name car_wash_<phase4_run> \
  --per-class
```

Success gates:

Car-wash:

```text
score > 0.80 local
nozzle recall clearly improves
drainage gate recall improves or stays stable
false_positive > 0.90
no new large FP category appears
```

Beverage:

```text
score > 0.65 before considering deploy
bottle AP no longer zero
cup/can precision remains controlled
false_positive > 0.85
```

## Decision

Current best Phase 4 order:

```text
1. Build failure-review exporter.
2. Review Car-wash misses.
3. Build Car-wash nozzle/drainage targeted data.
4. Retrain Car-wash YOLO11n.
5. Re-run Phase 3.
6. In parallel or after Car-wash v1, build Beverage data from TACO/Roboflow.
```

Reason:

```text
Car-wash is already closer to deployable.
Beverage has more outside data upside, but it needs a bigger data rebuild before
it is competitive.
```
