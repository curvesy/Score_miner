# SN44 Score Miner Current Best Plan

Date: 2026-05-24

This replaces the old public-football `PlayerDetect` execution plan for the current live manifest. Keep `plan_for_score.md` as historical football-system research, but do not execute its RF-DETR/player-tracking pipeline for the current public Detect tasks.

## 1. Current Truth

Live manifest:

```text
https://turbo.scoredata.me/manifest/8218320-39f3e736421460b495ea78390191b09ddf98b21d63dbc3643798d589f91597bb.yaml
```

Current active elements:

```text
manak0/Detect-beverage-detect   public   weight 0.02   cap 30MB
manak0/Detect-crime             public   weight 0.03   cap 30MB
manak0/Detect-fire              public   weight 0.02   cap 30MB
manak0/Detect-car-wash          public   weight 0.02   cap 30MB
manako/DetectCricketDelivery    private  weight 0.05   no manifest size cap
manako/DetectFootballEvent      private  weight 0.20   no manifest size cap
```

Public Detect rule:

```text
The full Hugging Face model repository revision must be <= max_model_size_mb.
For current public Detect tasks, that is 30MB in the manifest.
```

This is enforced in `turbovision/scorevision/utils/miner_registry.py`: the validator sums the whole HF model repo tree and skips miners above the cap.

Private football/cricket are different:

```text
No 30MB manifest cap.
Self-hosted/public endpoint path.
Football is soccer_action, not object boxes.
```

## 2. Best Business Decision

Build two tracks, but execute them in this order:

```text
Track A: public Detect garage, fastest proof and lowest cost
Track B: private football, highest upside but harder and blind
```

Start public Detect now because:

```text
1. Chutes-hosted, no 24/7 GPU server.
2. Score provides task-specific starter data.
3. We can locally iterate and deploy fast.
4. Current public boards have weak top-5 cutoffs.
5. Same garage works for the next NEW element.
```

Keep football as second project because:

```text
1. Weight is 0.20.
2. Current leader score is only around 29%.
3. Earnings are real.
4. But it needs a private endpoint, uptime, action-spotting model, and no public GT.
```

## 3. First Targets

Start with:

```text
1. Car-wash
2. Beverage
```

Reason:

```text
Car-wash:
  Fewest participants.
  Current leader around low/mid 70s.
  Large target gap.
  Best chance to rank quickly.

Beverage:
  Easier labels/classes.
  Excellent for proving the whole pipeline.
  Classes are common and easy to augment.
```

Do not start with:

```text
Fire:
  Already around/over target in recent API reads.
  Harder false-positive control.

Crime:
  Higher weight but harder semantics.
  Active notes matter: hoodies count only when hood up, balaclavas under hoodie ignored, floor gloves/hoodies ignored.
  Good second wave, not first baseline.
```

## 4. Data Strategy

Use Score data first. Public internet data is filler, not the anchor.

Score exposes task-specific current data through the console V2 API:

```text
https://console.scorevision.io/api/v2/elements/<element_id>?lookback_days=4
```

The element detail includes:

```text
starterPack:
  starterAssets:
    imageUrl
    annotationUrl
    annotations
    objects

latestAnnotatedChallenge:
  frame image URL
  top miner predictions
  response_url

challengeDetails:
  per-challenge miner scores
  top competitor repos/revisions/slugs
```

Important correction:

```text
The currently exposed starter pack is small, around 7 labeled images per public element.
It is not a full training set.
```

Use it as:

```text
1. Validation anchor.
2. Class-order truth.
3. Visual style reference.
4. Synthetic generation target.
5. Sanity check for boxes and thresholds.
```

Critical rule:

```text
Never trust synthetic validation alone.
Synthetic images are for training diversity, not for final model selection.
Final checkpoint/threshold selection must be anchored on Score starterPack images, proof images, and latestAnnotatedChallenge frames.
```

The public task starterPack is tiny, so the winning move is not "more synthetic images" by itself. The winning move is:

```text
diverse synthetic images
+ real Score-style proof/starter validation
+ hard negatives
+ false-positive weighted threshold sweep
```

### Beverage Data

Classes:

```text
0 cup
1 bottle
2 can
```

Data sources:

```text
Primary:
  Score starterPack images and annotations.
  Score proof/challenge images for style.

Secondary:
  COCO/Open Images cup/bottle/can.
  Public bottle/cup/can datasets.
  Generated synthetic Score-like scenes.

Hard negatives:
  jars, boxes, glassware, labels, trash, hands, plates, shelves, reflections.
```

### Car-wash Data

Classes:

```text
0 broom
1 drainage gate
2 nozzle
3 track
```

Data sources:

```text
Primary:
  Score starterPack images and annotations.
  Score proof/challenge images for style.

Secondary:
  Web/public data for brooms, spray/nozzles, drains/grates, car-wash floors/rails.
  Synthetic copy-paste onto car-wash-like backgrounds.
  Manual labeling of the highest-value 300-1000 images if needed.

Hard negatives:
  hose, pipe, floor seam, tire, shadow, water reflection, wall stripe, floor drain not matching target, rail-like lines.
```

## 5. Model Plan

The model rule is not "must use YOLO." Any model is allowed if:

```text
1. Full HF model repo revision <= 30MB.
2. Output schema is correct.
3. Class IDs match manifest object order.
4. Latency/health pass.
5. Score is better.
```

Model bake-off order:

```text
A. YOLO11n
B. YOLO26n
C. YOLO26s FP16/ONNX if full repo <= 30MB
D. YOLOv8n as mature control
E. RT-DETR tiny only if exported repo <= 30MB and deployment is reliable
```

Why this order:

```text
YOLO11n:
  First-class baseline, not just a backup.
  2026 comparisons suggest YOLO11 can remain very strong at nano/small scales.
  Our 30MB cap forces us into exactly that nano/small regime.

YOLO26n:
  Safest under 30MB.
  Modern NMS-free detector.
  STAL/ProgLoss may help small targets and false-positive control.

YOLO26s:
  Likely stronger.
  Use if export and full HF revision fit <=30MB.

YOLOv8n:
  Mature control model.
  Useful for detecting pipeline bugs because it is stable and well understood.

RT-DETR tiny:
  Worth testing after baseline.
  Do not block first deploy on it.
  More fragile export/training path and size uncertainty.
```

Winning criterion:

```text
validator_sim_score + live score > model name.
```

Expectation:

```text
Do not assume YOLO26n beats YOLO11n just because it is newer.
Do not assume YOLO11n beats YOLO26n just because one external study says it can at nano/small scale.
Run both on the exact Score task data and pick the measured winner.
```

Do not pay for giant public models:

```text
RF-DETR / large DETR public path is not the first move because the 30MB full-repo cap is real.
```

## 6. Training Plan

Train both Car-wash and Beverage during one GPU rental.

Recommended rental:

```text
24GB GPU preferred: RTX 4090, A5000, L4/A10 if cheaper.
16GB works for nano/small.
No need for H100/A100 first.
```

Resolution:

```text
Use Score image geometry as reference.
Live challenge examples are around 1408x768.
Train/eval at the manifest-style resize behavior, and test 768/960/1280 long side.
```

Augmentations:

```text
Moderate mosaic/copy-paste.
Photometric jitter.
Perspective/scale/crop.
Motion blur and compression.
Domain randomization for synthetic images.
Heavy hard negatives.
```

Synthetic generation priority:

```text
1. Diversity of background.
2. Diversity of camera angle and object scale.
3. Clutter and occlusion.
4. Lighting/reflection/compression shifts.
5. Realistic object placement.
6. Only then increase image count.
```

This matters because synthetic validation can look excellent while live score stays mediocre. We use synthetic data to widen coverage, then validate against real Score frames.

Avoid:

```text
Overfitting the 7 starter images.
Training only on generic COCO-style images.
Low threshold that creates many false positives.
Wrong class order.
Selecting checkpoint by synthetic mAP only.
```

## 7. Scoring And Thresholds

Public scoring includes false-positive pressure. The exact implementation can shift, but the current strategic rule is stable:

```text
Do not maximize recall blindly.
Tune confidence threshold for the final composite score.
```

This is not optional for a winning attempt:

```text
normal YOLO mAP is a training signal
Score-style validation is the decision signal
```

The local scorer does not need to reproduce the entire validator stack on day one. It must be good enough to rank candidate checkpoints/configs by:

```text
map50 approximation
false-positive pressure
per-class error profile
```

Per element, sweep:

```text
confidence: 0.20 to 0.95
input size: 768 / 960 / 1280
max detections per image
optional class-specific thresholds
optional SAHI/TTA
```

SAHI/TTA rule:

```text
Use only if it improves Score-style validation and live score.
The latency budget is generous, but the public service also advertises low latency targets, so do not add complexity blindly.
SAHI is most interesting for nozzle/can/small distant objects.
TTA is useful only if it does not add duplicate false positives.
```

Class-specific threshold examples:

```text
Beverage:
  bottle/can/cup can often tolerate moderate threshold.
  suppress glass/jar false positives.

Car-wash:
  track/drainage gate are large structural objects: higher threshold.
  nozzle is smaller: separate threshold may help.
  broom can be confused with rail/hose: hard negatives matter.
```

## 8. Build Components

Build in this order:

```text
1. Score starterPack downloader
2. Dataset converter to YOLO format
3. Manifest class-order guard
4. Real proof/challenge frame collector
5. Training configs for YOLO11n/YOLO26n/YOLO26s
6. Export + full HF repo size gate
7. Local scorer/validator_sim approximation
8. False-positive weighted threshold sweep
9. Chutes deploy wrapper
10. Live score watcher
11. Radar cron
12. RT-DETR tiny bake-off
```

### 8.1 Advanced Components Worth Adding

These are the useful ideas from the old architecture, translated to public Detect:

```text
Replay store:
  Save every starter/proof/live challenge image we can access, predictions, threshold config, score, and failure tags.

Competitor mining:
  Pull leaderboard/result-shard predictions where public. Compare top miners' box counts, thresholds, class mix, and failure profile.

Optuna optimizer:
  Use only after brute-force sweeps work. Tune confidence, per-class thresholds, imgsz, max_det, SAHI/TTA settings. Do not retrain inside early trials.

Experiment tracking:
  Use one backend, preferably MLflow local-first. Log data manifest, git revision, model weights path, export format, HF repo size, and live score.

Visual review:
  Use Supervision/FiftyOne/CVAT-style workflows to inspect misses and false positives. Manual review is for high-value hard cases, not every image.

Offline teachers:
  SAM3 and GroundingDINO can propose boxes/masks for unlabeled or proof-style images. Treat them as pseudo-label sources that need review, not as live public models.
```

Do not add:

```text
TrackLab / sn-gamestate as dependencies.
They are football/video references. For public Car-wash/Beverage, they add dependency weight and solve the wrong problem.
```

The old `plan_for_score.md` concepts to keep:

```text
closed score loop
hard failure mining
threshold sweeps
model bake-off by actual score
deployment reliability checks
watcher/radar
```

The old concepts to ignore for public Detect:

```text
VideoState
team colors
tracking
pitch keypoints
RF-DETR-L/DEIM/D-FINE football detector race
ball/player roles
```

Those only matter for a future football/video task, not Car-wash/Beverage.

## 9. Deployment Rules

Public Detect deploy artifact:

```text
HF model repo revision <= 30MB total.
Keep repo minimal.
One model artifact.
Small config/class-map files only.
No data.
No logs.
No duplicate checkpoints.
No optimizer states.
```

Before commit/deploy:

```bash
du -sh path/to/hf_repo
```

Then verify with the same logic the validator uses if possible:

```text
HfApi.list_repo_tree(... recursive=True, expand=True)
sum node.size
```

## 10. Monitoring

Already available:

```bash
python3 score_miner_project/scripts/competition_radar.py --overview --leaderboards --top 8
python3 score_miner_project/scripts/track_scorevision_earnings.py --top 25 --limit 200
```

Radar tells:

```text
NEW / ENDED / CHANGED elements
active score vs target
participants
winner hotkey
leaderboard scores
```

Earnings tracker tells:

```text
current revenue delta by hotkey/rank over time
```

Run radar hourly and earnings daily.

## 11. Exact Execution

### Day 0

```text
Confirm wallet/hotkeys.
Confirm Chutes/HF access.
Run radar baseline.
Rent one 24GB GPU.
```

### Day 1

```text
Download Score starterPacks for Car-wash + Beverage.
Download/proxy latest proof images.
Convert labels to YOLO.
Render labels for visual inspection.
Train YOLO11n Beverage.
Train YOLO26n Beverage.
Train YOLO11n Car-wash.
Train YOLO26n Car-wash.
Log all runs with data manifest and config.
```

### Day 2

```text
Export YOLO11n and YOLO26n.
Verify full HF repo <=30MB.
Pick first live baseline by Score starter/proof validation, not synthetic mAP.
Deploy Car-wash baseline first.
Deploy Beverage baseline second.
Start live score watcher.
Save deployment release records.
```

### Day 3

```text
Train YOLO26s.
Export FP16/ONNX.
Check full repo size.
Replace live model only if Score-style validation improves and size passes.
```

### Day 4

```text
Generate synthetic data matched to Score starter/proof images.
Add hard negatives.
Add reviewed pseudo-labels from SAM3/GroundingDINO only if they pass visual checks.
Retrain.
False-positive weighted threshold sweep.
Redeploy best.
```

Phase 4 rule:

```text
Do not generate data blindly.
Use Phase 3 failures to decide what data to add.

Beverage:
  attack jars/glassware/hands/labels/boxes/trash false positives.

Car-wash:
  attack hose/pipe/floor-seam/tire/shadow/reflection/rail-like false positives.

Synthetic data:
  prioritize diversity and domain randomization over raw volume.
```

### Day 5

```text
Try YOLOv8n control model.
Try RT-DETR tiny only if export size is plausible.
Try RF-DETR Nano/Small only if minimal exported repo can realistically fit <=30MB.
Run Optuna config-only search after brute-force threshold sweeps.
Keep only if it beats live/local score.
```

### Day 6-7

```text
Stabilize public miners.
Register/prepare next public element if radar shows NEW.
Start private football skeleton only after public miners are live.
```

## 12. Decision Summary

If the goal is "best chance to win soon":

```text
Car-wash first.
Beverage second.
YOLO11n and YOLO26n baseline immediately.
YOLO26s upgrade if size passes.
RT-DETR tiny bake-off later.
Score starterPack is the anchor data.
Diverse synthetic data + hard negatives + real proof validation are the edge.
```

If the goal is "highest ceiling":

```text
Public Detect pays for learning and position.
Football private is the next high-upside project.
```

Do not wait for a perfect model. Deploy a correct baseline, measure, then iterate.
