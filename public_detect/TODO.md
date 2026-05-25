# Public Detect TODOs

Each implementation task starts with a research checkpoint. Do not add code for a task until the checkpoint is done and the source is recorded in `docs/research_sources.md`.

## Phase 0 - Baseline Setup

- [x] Research latest ScoreVision API fields for `starterPack`, `latestAnnotatedChallenge`, and `challengeDetails`.
- [x] Run radar and save current Car-wash/Beverage leaderboard snapshot.
- [x] Define Phase 0 acceptance gate: starter images render with boxes, class IDs match manifest, and no label is outside image bounds.
- [x] Implement Score starterPack downloader.
- [x] Implement image + annotation archive format.
- [x] Implement manifest class-order guard.
- [x] Implement YOLO dataset converter.
- [x] Implement visual sanity renderer for labels and predictions.
- [x] Add tests for bbox conversion and class-order mapping.

## Phase 1 - Training Baselines

- [x] Research current Ultralytics YOLO11/YOLO26 training/export APIs.
- [x] Verify locally that `YOLO("yolo26n.pt")` exists in the installed Ultralytics version before depending on YOLO26.
- [x] Add modern uv GPU setup for rented machines.
- [x] Add training config for YOLO11n.
- [x] Add training config for YOLO26n.
- [x] Add training config for YOLO26s.
- [x] Add YOLOv8n control config for pipeline debugging.
- [x] Add baseline training runner and one-epoch CUDA smoke test.
- [x] Run Car-wash YOLO11n/YOLO26n baseline.
- [x] Run Beverage YOLO11n/YOLO26n baseline.
- [x] Log copied run artifacts under `runs/`.

## Phase 2 - Export And Size Gate

- [ ] Research current Ultralytics ONNX export options and Chutes compatibility.
- [ ] Research current TurboVision `sv deploy-os-miner` path and Chutes template expectations.
- [ ] Implement local export script.
- [ ] Implement full-repo size gate using the same semantics as the validator.
- [ ] Package minimal HF repo layout.
- [ ] Verify each model repo is <= 30MB.
- [ ] Add dry-run deployment checklist: env vars present, HF revision recorded, no extra files in model repo.

## Phase 3 - Score-Style Validation

- [ ] Research current public scoring fields in Score result shards.
- [ ] Implement proof/latest challenge frame collector.
- [ ] Implement competitor prediction collector from console/result shards where accessible.
- [x] Implement local approximation of map50 + false-positive composite. Do not skip this; it is the validator-style instrument for picking a winner.
- [x] Implement confidence threshold sweep.
- [x] Implement per-class threshold sweep.
- [ ] Implement max-det and image-size sweep.
- [ ] Add Optuna config-only search for thresholds/imgsz/max-det/SAHI settings after brute-force sweep works.
- [x] Generate score reports for Car-wash and Beverage.
- [x] Generate diagnostics for false positives and missed ground-truth boxes.
- [ ] Define no-overfit gate: best config must improve starter/proof validation and not only synthetic validation.

Current Phase 3 result:

```text
Car-wash YOLO11n is the lead candidate:
  score 0.7464, fp_score 0.9429, precision 0.8947, recall 0.5397
  problem class: nozzle, 8 TP / 1 FP / 22 FN

Beverage YOLO11n is not ready:
  score 0.4880, fp_score 0.8286, precision 0.7857, recall 0.3826
  problem class: bottle, 0 TP / 0 FP / 20 FN
```

Next implementation should start Phase 4 from these diagnostics, not random data volume.

## Phase 4 - Data Advantage

- [ ] Research modern synthetic-to-real object detection practices for small datasets.
- [ ] Research current SAM3/GroundingDINO/Supervision pseudo-label path for offline labeling only.
- [ ] Research current FiftyOne/CVAT workflow for manual review and hard-negative correction.
- [x] Write Phase 4 source-priority and data-plan doc for Car-wash and Beverage.
- [ ] Use Phase 3 failures to define the data plan before generating data.
- [x] Build failure-review exporter from `diagnostics.json`.
- [x] Export Car-wash and Beverage failure review artifacts.
- [ ] Review Car-wash missed nozzles/drainage gates and false positives.
- [ ] Review Beverage missed bottles/cups and can false positives.
- [x] Build Phase 4 source manifest for approved outside data candidates.
- [x] Implement/choose ingestion path for approved sources only.
- [x] Add Beverage COCO/TACO ingest config and script.
- [x] Add Car-wash local video frame extraction script.
- [ ] Download or mount approved Beverage source data, then run COCO ingestion.
- [ ] Collect approved Car-wash videos, then run frame extraction.
- [ ] Build diverse synthetic generation recipe for Beverage.
- [ ] Build diverse synthetic generation recipe for Car-wash.
- [ ] Build teacher-assisted pseudo-label queue for unlabeled Score-style/proof/public images.
- [ ] Manually review a small high-value set before trusting pseudo-labels.
- [ ] Build hard-negative sets.
- [ ] Retrain with diversity-first data.
- [ ] Compare against starter/proof validation, not synthetic-only validation.
- [ ] Mine failures by category: missed object, wrong class, duplicate box, background false positive, class-order issue.
- [ ] Re-run Phase 3 threshold sweeps after each retrain.

## Phase 5 - Deployment

- [ ] Research current Chutes deployment path for Score public miners.
- [ ] Build deploy package for Car-wash.
- [ ] Build deploy package for Beverage.
- [ ] Register hotkeys.
- [ ] Deploy Car-wash first.
- [ ] Deploy Beverage second.
- [ ] Add health and live-score watchdog.
- [ ] Save each live deployment as a reproducible release record: element, hotkey, HF repo, revision, chute slug, threshold config, size report.

## Phase 6 - Model Bake-Off Extension

- [ ] Research RT-DETR tiny current export/deploy path and expected model size.
- [ ] Research RF-DETR Nano/Small current size; test only if exported minimal repo can realistically pass 30MB.
- [ ] Try YOLOv8n control.
- [ ] Try RT-DETR tiny only if size/export path is clean.
- [ ] Try SAHI/TTA only if proof validation improves without false-positive damage.
- [ ] Try teacher-assisted distillation only after baseline deployment is live.

## Phase 7 - New Element Land Grab

- [ ] Keep radar cron running.
- [ ] When `NEW:` appears, create element config within 1 hour.
- [ ] Download starterPack immediately.
- [ ] Train YOLO11n/YOLO26n baseline same day.
- [ ] Deploy first acceptable model before leaderboard fills.
- [ ] Reuse generic detect profile; do not create element-specific custom code unless the evidence says it helps.
