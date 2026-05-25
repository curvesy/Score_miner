# SN44 #1 Playbook — Game-Changer Moves (May 21 2026, fact-checked)

> Goal: not "a good miner." Top-1 on Subnet 44, and **stay #1 across every element SCORE ships**.
> Premise: the score is earned by aligning to SAM3-PGT, exploiting public audit data, and being modular enough to absorb every new vertical SCORE launches.
> Posture: aggressive, fact-checked, no hype.

---

## 0. The Reframe That Changes Everything

Three facts that change the strategy:

1. **The "ground truth" is SAM3 output.** The validator runs SAM3 with `element.objects` as prompts on selected frames, drops frames where consecutive PGT mask IoU < 0.7, and scores you against what's left. (Verified in `validator/audit/open_source/spotcheck.py:288`.)
2. **`baseline_theta = 0.78` is a vertical cliff.** Score 0.77 → earn `delta_floor = 0.01`. Score 0.79 → earn `0.01 × β`. No partial credit. (Verified in `utils/manifest.py:357-371`.)
3. **SCORE is pivoting from soccer to general-purpose computer vision.** Cricket already shipped (`miner/private_track/CRICKET_MINER_SPEC.md`). Public roadmap announcements include **petrol forecourts, retail, fruit grading, car washes** — confirmed by Two-a-Day fruit-packing partnership and "make every camera intelligent" mission statement. The soccer-only miner becomes obsolete the day a new vertical ships.

Stop building "a better soccer miner." Build **a modular SAM3-aligned vision engine that absorbs any new element SCORE drops**.

---

## 1. Game Changer #1 — Become the SAM3 Student ✅

**Status: real, verified.** Meta released SAM3 weights March 2026 (`facebookresearch/sam3`). SAM 3.1 (March 27 2026) is faster. Free for non-commercial use under custom SAM License.

### The play
1. **Self-host SAM3** locally (open weights from HuggingFace, request access). No Chutes credits required.
2. **Run SAM3 on every scoredata.me clip you can find.** Use `element.objects` as the text prompts. Apply the same `filter_low_quality_pseudo_gt_annotations(min_iou=0.7)` filter the validator does.
3. **Distill SAM3 into a small student**: YOLO26-s or RF-DETR-S, train objective: mAP against SAM3 PGT, not against human SoccerNet labels.
4. **Re-distill weekly** as `pgt_recipe_hash` versions drift.

### Why this is unfair
The miner that produces SAM3-shaped boxes scores higher than the miner that produces "correct" boxes, because the validator's PGT *is* SAM3-shaped.

### Implementation cost
- ~2-4 weeks for the distillation pipeline
- ~$0-50 if you self-host SAM3 on your own GPU (RTX 3090+)
- 1 GPU-day fine-tune per student

### Expected delta
+0.10 to +0.20 on weighted score. Almost certainly enough to clear `baseline_theta=0.78`.

---

## 2. Game Changer #2 — Mine the Public Audit Shards ✅

**Status: real, verified in code.** The central validator commits its public results URL **on-chain** (`get_validator_indexes_from_chain(netuid=44)` in `validator/audit/open_source/spotcheck.py:107`). Every shard exposes: `miner_hotkey`, `central_score`, `composite_score`, `responses_key` (pointer to predictions JSON), `scored_frame_numbers`. **Anyone with a Bittensor wallet can read this.**

### The play
1. Query the chain for the central validator's commitment under netuid 44.
2. Fetch the index URL. List shards.
3. Each shard contains another miner's score and a pointer to their predictions.
4. Group by `miner_hotkey` and `element_id`. Rank by score.
5. For the top 5 miners, **download their actual JSON predictions** and reverse-engineer:
   - Do they emit ball/referee/goalkeeper classes?
   - Average `boxes_per_frame`?
   - `cluster_id` distribution?
   - Score breakdown per pillar?
6. Copy what works. Beat it.

### Why this is unfair
You're staring at the winning answers. Most miners ship and forget to look at competitor outputs.

### Implementation cost
- 2-3 days to build the chain-fetch + R2 listing + analysis script
- Zero ongoing cost (public data)

### Expected delta
Strategic. You stop guessing what works and start copying what works.

---

## 3. Game Changer #3 — Replicate the Validator Locally ✅

**Status: real.** Code is open source. SAM3 weights are open. `pgt_recipe_hash` is in the manifest.

### The play
1. Pull live manifest via `sv manifest current`. Cache `pgt_recipe_hash`, `salt`, `baseline_theta`, all pillar weights.
2. Stand up local SAM3. Now your local PGT generator and the validator's PGT generator are bit-identical.
3. Replay-mine `scoredata.me` clips. For each, run the full validator path: `prepare_challenge_payload → SAM3 PGT → filter_low_quality → score_predictions`.
4. Your local score function now returns the same number as production within tolerance.

### Why this is unfair
Every other miner is tuning to an approximation. You're tuning to the literal scoring function.

### Implementation cost
- ~1 week
- Local GPU for SAM3 inference (RTX 3090 / 4090 / L4)

---

## 4. Game Changer #4 — Private Track (GHCR + TEE) ⚠️ Correctly framed

**Status: real but different from what's commonly hyped.** The "TEE bonus pool" is real (`tee.trusted_share_gamma = 0.2` in the manifest), but the miner side is not "you attest your image with Intel TDX." It's actually: **you publish a private Docker image to GHCR, add SCORE's hotkey as a read-only collaborator, and commit the image hash on-chain.** TEE is on the Chutes/Targon infrastructure side, not something you bring.

### The play
1. Follow `scorevision/miner/private_track/MINER.md` exactly.
2. Build your miner as a Docker image: `sv -v deploy-pt-miner --tag v1.0.0`.
3. Push to a private GHCR package. Add `DataAndMike` as read-only collaborator (per the docs).
4. Commit the image hash on-chain.
5. Serve a single element per deploy (e.g., football, then deploy a second one for cricket).

### Why this is undersubscribed
Most open-source miners only do the public track. The private track has fewer competitors and a separate emissions slice.

### Implementation cost
- 1-2 weeks: build Docker + GHCR setup + on-chain commit
- Need GitHub PAT with `write:packages` scope

### Expected delta
Up to +25% effective TAO emissions for the same model, **if** you're a top-3 private-track miner. Note: private track is element-specific — you serve one element per deploy.

---

## 5. Game Changer #5 — Multi-Vertical Modular Architecture 🆕

**Status: this is the move most miners miss completely.** SCORE has stated publicly that the platform will serve **petrol forecourts, retail, fruit grading, car washes** and the broader "make every camera intelligent" mission. Cricket already shipped. The `keypoint_template` enum already supports `football, cricket, basketball`.

### Why soccer-only miners will get crushed
The day SCORE ships, say, a "ShelfDetect" element for retail or a "VehicleQueue" element for petrol forecourts, soccer-only miners need a full rewrite. The first miner to submit on a new element captures rewards uncontested for weeks.

### The play — modular MinerRuntime
Refactor your pipeline so the `element_id` dispatches a **profile** instead of running soccer-specific logic. Sketch:

```python
ELEMENT_PROFILES = {
  "PlayerDetect_v1@1.0":   {"detector": "rfdetr_soccer_distilled", "tracker": "deep_eiou",  "team_color": True,  "keypoints": False},
  "BallDetect_v1@1.0":     {"detector": "yolo26_ball_specialist",   "tracker": "deep_eiou",  "team_color": False, "keypoints": False},
  "Cricket_v1@1.0":        {"detector": "rfdetr_cricket_distilled", "tracker": "deep_eiou",  "team_color": False, "keypoints": False},
  "PitchCalib_v1@1.0":     {"detector": None,                       "tracker": None,         "team_color": False, "keypoints": "tvcalib_segformer"},
  # Future-ready stubs:
  "FruitGrade_v1@1.0":     {"detector": "yolo26_fruit_distilled",   "tracker": None,         "team_color": False, "keypoints": False},
  "VehicleCount_v1@1.0":   {"detector": "rfdetr_vehicle_distilled", "tracker": "deep_eiou",  "team_color": False, "keypoints": False},
  "SignDetect_v1@1.0":     {"detector": "rfdetr_sign_distilled",    "tracker": None,         "team_color": False, "keypoints": False},
}
```

### What this enables
- **Day-of release deployment**: when SCORE ships a new element, you create a new HF revision with the appropriate detector and ship in hours.
- **Per-element threshold profiles**: ball-detect uses different conf thresholds than player-detect.
- **No wasted code**: pitch keypoints don't run on fruit-grading elements.

### Implementation cost
- 1 week to refactor `MinerRuntime` for element dispatch
- 2-3 days to write a generic "BoundingBoxOnly" profile for any future detection-only element

### Expected delta
Strategic, not numeric. **This is what separates a #1 miner today from a #1 miner across all elements forever.**

---

## 6. Game Changer #6 — Adaptive Compute (corrected) ⚠️

**Status: I overstated this. Corrected.** The salt mechanism (`offsets, strides`) is VRF-derived per-validator. You **cannot** predict which exact frames will be sampled. What you CAN do is process all frames cheaply and intensify compute on high-uncertainty regions.

### The play
1. **Light path** for every frame: ByteTrack or Deep-EIoU using last-known boxes + minimal detector refresh.
2. **Heavy path** triggered by uncertainty signals:
   - Scene cut detected (large frame-diff)
   - Tracking confidence drops below threshold
   - Player count changes abruptly
   - Ball lost
3. Heavy path = full-resolution detector + ReID + jersey-OCR.

### Why this works without salt prediction
You don't need to know which frames the validator samples. By keeping all frames detector-fresh and stable, every sample is high-quality. By intensifying compute only where uncertainty spikes, you preserve latency budget for the cases that matter.

### Implementation cost
- 1 week to architect the scheduler

### Expected delta
Latency cuts that let you afford **ensembles, ReID, jersey OCR, CLIP backbone features** on critical frames without blowing the 200ms RTF gate.

---

## 7. Game Changer #7 — SoccerMaster Foundation Backbone ✅

**Status: real.** `github.com/haolinyang-hlyang/SoccerMaster` — CVPR 2026 Oral, vision foundation model pre-trained on soccer broadcast.

### The play
1. Pull SoccerMaster weights.
2. Use as frozen backbone with a lightweight detection head.
3. Fine-tune the head on your SAM3-distilled labels (from GC#1).

### Why this is unfair
Most miners use COCO-pretrained backbones. They have to undo COCO's coach/fan/spectator priors during fine-tuning. SoccerMaster has soccer priors from day one.

### Implementation cost
- ~3 days
- 1 GPU-day fine-tune

### Expected delta
+0.05 to +0.10 on IoU and Count pillars. Possibly more on Role.

### Caveat
SoccerMaster only helps for **soccer** elements. For cricket/fruit/retail you'd want different foundation models — DINOv3 generalizes, SAM3 already does, or a domain-specific pretrained model. See GC#5 (modular architecture).

---

## ~~Game Changer #5 — CLIP Validator Gate~~ ❌ RETRACTED

**The original Game Changer #5 was wrong.** I assumed "Clip and Homography" in subnet docs meant OpenAI CLIP. After grepping the entire `scorevision/` codebase, every `clip_*` reference is to **video clips (30-second segments)**, not the CLIP model. The validator does **not** run a CLIP semantic check on your boxes. Do not build a CLIP gate. Save the week.

---

## 8. The Architecture (Updated)

```
                     ┌─── element_id dispatch ──────────────────────┐
incoming /predict ─→ │ ELEMENT_PROFILES[element_id] → profile cfg   │
                     └────────────────┬─────────────────────────────┘
                                      │
                ┌─────────────────────┴───────────────────────┐
                │                                              │
                ▼                                              ▼
       SOCCER PROFILE                                   GENERIC DETECT PROFILE
       ┌──────────────────────────┐                  ┌─────────────────────────┐
       │ light path (every frame) │                  │ detector (per profile)  │
       │ Deep-EIoU tracker        │                  │ tracker (optional)      │
       │                           │                  │ output schema           │
       │ heavy path (uncertain)   │                  └─────────────────────────┘
       │ ├─ SAM3-student detector │
       │ ├─ OSNet ReID            │       Same chassis. Different
       │ ├─ Team color k-means    │       detector/tracker/postprocess.
       │ ├─ PARSeq jersey OCR     │       Element-driven, not hardcoded.
       │ └─ TVCalib keypoints     │
       │                           │
       │ VideoState (per-clip)    │
       │ ├─ EMA box smoothing     │
       │ ├─ carry-forward team_id │
       │ ├─ jersey vote per track │
       │ └─ scene-cut history     │
       │                           │
       │ output shaping            │
       │ ├─ class-aware NMS       │
       │ ├─ sideline mask filter  │
       │ └─ frame-count gate      │
       └──────────────────────────┘
```

The chassis is fixed. The detector, tracker, and postprocess swap based on `element_id`.

---

## 9. Operational Edges (Updated)

### 9.1 Long-running chute — cache aggressively
Chutes nodes stay warm. Cache decoded frames within a `/predict` call, TensorRT engine memory across requests, precomputed grass/field masks per scene.

### 9.2 Cold-start engine precompile
Ship a `.engine` file in your HF repo. Load during `@chute.on_startup()`. First `/predict` goes from 60s+ JIT to 200ms.

### 9.3 Determinism shields you against spotcheck ✅
`scores_match(original, spotcheck, threshold)` in `spotcheck/runner/main.py` — divergent re-scores fail spotcheck. Lock:
```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 9.4 Frame count = clip score
Frame count gate is binary. Miss one frame → entire clip scores zero. Add a frame-index integrity check at output time; if decoder fails, emit previous-frame boxes with confidence × 0.95.

### 9.5 Manifest diff subscription
Live manifest is hot-reloadable. New elements appear. `pgt_recipe_hash` rolls. Build a **daily manifest-diff job** that alerts when:
- A new `element.id` appears (← biggest signal — new vertical opportunity)
- Pillar weights change
- `baseline_theta` drops or rises
- `pgt_recipe_hash` rolls (your SAM3 PGT may be invalidated)

### 9.6 On-chain validator commitment is your competitive intelligence
The central validator commits a public-results URL via `get_validator_indexes_from_chain(netuid=44)`. Query the chain daily. Pull new shards. Track top miners.

---

## 10. The 30/60/90 Day Plan (Updated)

### Days 1-7: Foundation fixes (the silent bugs)
- Fix BGR/RGB color channel in `MinerRuntime.predict_batch`
- Fix `ball_cls_id` default
- Fix `min_players_per_frame` null cluster_id leak
- Read live `element.objects` at deploy time and bake correct `player_cls_id`
- Deploy to Chutes, get a real measured `latency_ms`

### Days 8-21: SAM3 + competitive intelligence
- Self-host SAM3
- Build the SAM3 PGT pipeline (regen + filter)
- Build the on-chain audit shard fetcher
- Profile top 5 miners on PlayerDetect and BallDetect
- Distill SAM3 into RF-DETR-S or YOLO26-s, soccer-specific

### Days 22-35: Modular architecture + Deep-EIoU + ReID
- Refactor `MinerRuntime` to element-profile dispatch (GC#5)
- Replace ByteTrack with Deep-EIoU
- Add OSNet small for ReID
- Carry-forward cluster_id when uncertain (fix smoothness leak)
- Sideline mask filter

### Days 36-50: TensorRT + Determinism + Private Track
- Export student detector to TensorRT FP16
- Lock determinism for spotcheck consistency
- Build private-track Docker image, push to GHCR, commit on-chain
- Now earning from both open and private tracks

### Days 51-75: Manifest watcher + Cricket spike
- Daily manifest-diff job
- Build a generic detection-only element profile (for any future non-soccer element)
- Train a cricket student model. Submit on the Cricket element. **First miner advantage.**
- Stage stub profiles for fruit grading / retail / vehicle counting

### Days 76-90: Optuna + Foundation backbone benchmarking
- Optuna over per-class thresholds, EMA alphas, scene-cut reset thresholds
- Benchmark SoccerMaster vs RF-DETR-M backbone on soccer
- Benchmark DINOv3 vs SAM3-distilled student for general elements
- PARSeq jersey OCR for top-conf soccer tracks

### Day 90+: Ready for every new element SCORE ships

---

## 11. Stack Decision (Concrete, Updated)

| Layer | Choice | Why |
|---|---|---|
| Detector (soccer) | RF-DETR-S **fine-tuned on SAM3-distilled clips** | SAM3-aligned; fast; ICLR 2026 |
| Detector (cricket / future) | RF-DETR-S retrained per element via modular framework | Same chassis, element-specific weights |
| Ball specialist | YOLO26-n, ball-only, high-res tile pass | NMS-free; ProgLoss for small objects |
| Tracker | **Deep-EIoU** (port from GTATrack) | 2025 SoccerTrack winner |
| ReID | OSNet-x0_25 | 5 MB, ~5ms/frame, Broadcast2Pitch standard |
| Team affiliation | OSNet cosine + Lab fallback, carry-forward on uncertainty | Avoid null cluster_id smoothness leak |
| Jersey OCR | PARSeq on high-conf tracks (Koshkina pipeline) | Stabilizes id across occlusions |
| Calibration | TVCalib + SegFormer **only for keypoint-weighted elements** | Constructor Tech winning stack |
| Backbone | SoccerMaster for soccer, DINOv3 for general | Domain-tuned features |
| Inference | TensorRT FP16 + CUDA Graphs | 200ms p95 budget |
| Determinism | Locked CUDA + locked seeds | Spotcheck survival |
| Architecture | **Element-profile dispatch** (GC#5) | Ready for any new vertical |
| Deploy | Open track + private (GHCR) track in parallel | 80% + ~20% emissions pools |

---

## 12. What I Got Wrong Last Round (Honest Log)

- ❌ "CLIP gate to match validator CLIP check" → **wrong**, "clip" in subnet docs = video clip, not OpenAI CLIP. Retracted.
- ⚠️ "TEE attestation by you the miner" → **overstated**, private track is private Docker on GHCR; TEE is on Chutes infra side.
- ⚠️ "Predict which frames validator samples via salt" → **overstated**, VRF makes salts per-validator unpredictable. Use uncertainty-driven adaptive compute instead.
- ✅ SAM3 distillation, audit shard mining, local validator replication, SoccerMaster, determinism, cold-start engine precompile — all confirmed.
- ✅ **Multi-vertical modular architecture** added as new game changer — this is what most miners miss.

---

## 13. TL;DR for the Impatient

- Distill SAM3 into a fast student. That's your detector for soccer.
- Mine the on-chain-published audit shards. That's your competitive intelligence.
- Replicate the validator function locally with real SAM3. That's your scoring oracle.
- Refactor for element-profile dispatch. That's your insurance against the soccer-to-everywhere pivot.
- Deploy on both open track and private (GHCR + on-chain) track. That's how you double your emissions.
- Lock determinism. That's how you survive spotchecks.
- Deep-EIoU tracker, OSNet ReID, PARSeq jersey OCR, TensorRT FP16, SoccerMaster backbone, adaptive compute on uncertainty — that's the stack.
- 30 days for foundation. 60 days for stack. 90 days for #1 plausibility and ready for every new element SCORE ships.
