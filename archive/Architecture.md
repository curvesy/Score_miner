# Score / TurboVision Miner — Pro Architecture & How-To Build Guide

**Version:** 2026-05-18
**Owner:** Sina (solo, AI-assisted execution)
**Target:** Top-3 on Bittensor SN44 (Score) via TurboVision
**Operating mode:** No personal GPU. All inference runs on Chutes. Fine-tuning rents A100 on Vast.ai (~$5–20 per session).

This document is the only source of truth. Every code change references a section here. Do not deviate without updating this file first.

---

## 0. North Star (read this every time you open the repo)

You are not building "a YOLO miner." You are building a **persistent football reconstruction system** that:

1. Maintains world state across frames (VideoState).
2. Routes compute adaptively to where uncertainty lives.
3. Optimizes against a **local replica of TurboVision's scorer** (validator_sim), not against COCO mAP.
4. Iterates via a closed loop: deploy → score → replay → mutate config with Optuna → redeploy.

**The moat is not the detector. It is:**
- `validator_sim` (locally replicating TurboVision scoring)
- `replay` (every failure preserved with full context)
- `optimizer_core` (Optuna mutating config against your local scorer)
- `VideoState` (persistent identity, palette, homography across frames)

Other miners have a better detector. You have a feedback loop. That wins.

---

## 1. Build Phases (definitive order)

Each phase is 2–3 days with AI-assisted coding. Do NOT skip phases. Do NOT merge phases. Each one compounds.

| Phase | Deliverable | Days | Gate to advance |
|-------|-------------|------|-----------------|
| **0** | Reconnaissance: manifest + scoring code read | 1 | `notes/scoring_spec.md` exists and is correct |
| **1** | Repo skeleton + class mapping guard + memory budget | 2 | `score_miner_core` wheel builds and installs |
| **2** | Local benchmark harness on test video | 1 | `benchmark/run_local.py` reports p50/p95/p99/mem |
| **3** | RF-DETR-M smoke deploy on Chutes (dry-run only) | 2 | `/health` clean, `/predict` returns valid JSON, p95 < 500ms |
| **4** | `validator_sim` wrapping TurboVision's own scorer | 2 | Score breakdown matches what `sv run-once` reports on same clip |
| **5** | Detector head-to-head (RF-DETR-L vs DEIMv2-L vs D-FINE-L) | 3 | Winner chosen by `validator_sim` score, not COCO AP |
| **6** | Replay mining infrastructure | 1 | Every dry-run saves: input, predictions, score, latency, failure tags |
| **7** | Optuna optimizer (config-only, no retraining) | 2 | 200 trials run on held-out match split; best config beats baseline by >5% |
| **8** | sn-gamestate / TrackLab reference run (offline) | 2 | Document which TrackLab modules to adapt, with memory/latency cost |
| **9** | Team/role logic with HSV/Lab palette + temporal memory | 2 | Palette pillar +20% vs baseline |
| **10** | VideoState + scene analyzer | 3 | Smoothness pillar +30% |
| **11** | Adaptive scheduler | 2 | p95 drops 30% without score loss |
| **12** | Ball specialist (high-res crops on uncertainty) | 2 | Ball pillar +25% without false-positive blowup |
| **13** | Homography upgrade (SegFormer/TVCalib/AuxFlow-inspired) | 3 | Keypoints pillar to top-5 |
| **14** | Score-aware calibration | 2 | Total weighted score +10% with no other changes |
| **15** | ReID + jersey number evidence (selective live) | 3 | Role pillar to top-5; palette stable across occlusions |
| **16** | Optuna v2 with replay-mined hard splits | 2 | Top-10 leaderboard position |
| **17** | Replay-driven fine-tuning of winning detector | 3 | Iou pillar to top-5 |
| **18** | TensorRT FP16 export | 2 | p95 < 150ms for batch 64 |
| **19** | SAM3 / Qwen3-VL offline pseudo-labeling pipeline | 2 | New training data validated |
| **20** | Final Optuna sweep + commit on-chain | 2 | Top-3 |

**Total realistic time:** 6–8 weeks solo with AI-assisted coding. Top-3 by week 8. Top-1 is contested with teams.

---

## 2. Repository Structure

```
validator_improve/
  turbovision/                  # upstream repo, read-only reference
  sn-gamestate/                 # cloned reference for offline benchmarks
  tracklab/                     # cloned reference for module patterns
  score_miner_dev/              # our development workspace
    pyproject.toml              # package definition, builds the wheel
    score_miner_core/           # the actual Python package
      __init__.py
      runtime/
        miner_runtime.py        # MinerRuntime — the entrypoint logic
        orchestrator.py         # routes batches through pipeline
        scheduler.py            # adaptive compute scheduler
        video_state.py          # VideoState world model
        scene_analyzer.py       # scene cuts, camera type, motion
        memory_budget.py        # tracks GPU/CPU memory under 5GB cap
        class_mapping.py        # detector classes -> manifest cls_id
        determinism.py          # seed locking, deterministic ops
      detector/
        base.py                 # DetectorBase ABC — same interface for all
        rfdetr_runner.py        # RF-DETR-M/L via rfdetr package
        deim_runner.py          # DEIMv2-L wrapper
        dfine_runner.py         # D-FINE-L wrapper
        soccerdetr_runner.py    # SoccerDETR only if weights ship
        detector_router.py      # selects active detector by config
        detector_benchmark.py   # head-to-head harness
        confidence_calibrator.py
      chunk/
        frame_sampler.py
        temporal_cache.py       # per-video state across batches
        video_reader.py         # robust BGR frame loader
      tracking/
        association.py          # IoU + appearance matching
        motion_model.py         # Kalman + flow-assisted prediction
        occlusion_memory.py
        reid_embeddings.py      # CLIP-ReIdent or OSNet wrapper
        tracklab_adapter.py     # bridges to TrackLab modules
      team/
        jersey_cluster.py       # HSV/Lab two-centroid clustering
        temporal_palette_memory.py
        role_cleanup.py         # player/GK/ref hysteresis
        jersey_number_evidence.py  # PARSeq-based selective OCR
      keypoints/
        homography_filter.py    # reprojection validation
        auxflow_propagation.py  # flow-assisted point propagation
        camera_motion.py
        tracklab_calibration_adapter.py
      ball/
        crop_refiner.py         # high-res tiled inference on uncertain frames
        trajectory_filter.py    # physics-plausible ball motion
        crop_scheduler.py
      validator_sim/
        manifest_loader.py
        schema_checker.py
        pillar_metrics.py       # wraps turbovision's scoring functions
        rtf_gate.py             # latency gate calculation
        report.py
      replay/
        failure_store.py
        clip_sampler.py
        hard_case_index.py
        regression_suite.py
      optimizer_core/
        search_space.py
        objective.py
        optuna_runner.py
        score_graph.py
        config_registry.py
      active_learning/
        fiftyone_export.py      # lightweight, optional
        uncertainty_sampler.py
        hard_negative_builder.py
      telemetry/
        mlflow_logger.py        # primary tracker
        run_manifest.py
      external/
        sn_gamestate_notes.md
        tracklab_adapter_notes.md
      benchmark/
        run_local.py            # the test harness
        score_breakdown.py
        replay_runner.py
      training/
        pseudo_labeling/
        hard_negative_mining/
        distillation/
        finetune_rfdetr.py
        finetune_deimv2.py
      export/
        onnx/
        tensorrt/
      deploy/
        bundle_builder.py       # builds the score_miner/ deploy package
        miner_flatten.py        # fallback flat single-file miner
        healthcheck.py
      datasets/
        raw/                    # SoccerNet GSR original
        curated/                # cleaned splits
        hard_negatives/
        calibration/
        splits.yaml             # which match IDs are held-out — NEVER touched by Optuna
    experiments/
      configs/                  # Hydra-style YAMLs
      results/                  # per-trial JSONL
      leaderboards/             # rolling top-N tracking
    notes/
      scoring_spec.md           # output of Phase 0
      class_mapping.md
      memory_budget.md
      benchmark_results.md
      detector_race_results.md
      optimizer_results.md
      replay_failures.md
      calibration.md
      validator_sim_design.md
  score_miner/                  # deployable TurboVision package
    miner.py                    # thin entrypoint
    chute_config.yml
    README.md
    dist/
      score_miner_core-*.whl
    models/
      rfdetr_l_finetuned.pth
      deimv2_l_finetuned.pth
      dfine_l_finetuned.pth     # only the winner from Phase 5 ships
      ball_specialist.pth
      pitch_keypoints.pth
      reid_clip.pth             # if Phase 15 ships
    notes/
      class_mapping.md
      benchmark_results.md
      calibration.md
```

**Why two folders:**
- `score_miner_dev/` is where you actually work — clean Python package, real engineering.
- `score_miner/` is what the Chute template loads. Its `miner.py` is a 10-line entrypoint that imports `score_miner_core` from `site-packages`.

**Primary deploy strategy:** `score_miner_core` ships as a wheel, installed via `chute_config.yml`. Allowed by the Chute template whitelist for installed packages.

**Fallback deploy strategy:** if Chutes rejects package imports, use `deploy/miner_flatten.py` to bundle everything into one `miner.py` file.


---

## 3. Phase 0 — Reconnaissance (Day 1, BEFORE writing any code)

**Goal:** Know exactly what TurboVision scores, what shape it wants the output in, and what the manifest expects.

### 3.1 Required reading

```bash
cd turbovision
uv sync

# Manifest — what tasks exist, what classes, what weights
cat tests/test_data/manifests/example_manifest.yml

# Class definitions, pillar weights, eval window
sv elements list
sv manifest current

# Scoring code — read every line, take notes
cat scorevision/vlm_pipeline/non_vlm_scoring/objects.py
cat scorevision/vlm_pipeline/non_vlm_scoring/keypoints.py
cat scorevision/vlm_pipeline/non_vlm_scoring/smoothness.py
cat scorevision/utils/evaluate.py

# Example miner — your starting template
cat scorevision/miner/open_source/example_miner/miner.py
cat scorevision/miner/open_source/example_miner/chute_config.yml

# Chute template — the rules of the game
cat scorevision/miner/open_source/chute_template/turbovision_chute.py.j2
```

### 3.2 What `notes/scoring_spec.md` must contain

Write it down. If you don't write it down, you will get class IDs wrong and lose 50% of your score for a week before you notice.

```markdown
# Scoring Spec — Element <ID>

## Element metadata
- element_id:
- service_rate_fps:
- latency_p95_ms:
- resize_long:
- n_keypoints:
- batch_size:
- eval_window_seconds:
- ground_truth: true | false (false means SAM3 pseudo-GT)

## Class order (THIS IS THE BUG THAT KILLS MINERS)
- cls_id=0: <name from manifest>
- cls_id=1: <name>
- cls_id=2: <name>
- cls_id=3: <name>
(map your detector's class indices to these — get it wrong, score zero)

## Pillar weights (from manifest, NOT assumptions)
- iou:
- count:
- palette:
- role:
- smoothness:
- keypoints:
- map50:
- precision:
- recall:
- false_positive:
- latency penalty curve:

## RTF gate
- target p95: <ms>
- zero-score p95: <ms>
- formula: RTF = (p95_latency_ms / 10000) * (service_rate_fps / 5)

## Output contract
- frame_id type: int, starts at 0 or 1?
- bbox format: x1y1x2y2 in pixels, integer or float?
- conf range: [0, 1]
- team_id type: int | str | null
- cluster_id type: int | str | null
- keypoints count: exactly n_keypoints, (0,0) for missing
- frames length: must equal meta.min_frames_required

## What scoring code actually rewards (from reading objects.py/keypoints.py/smoothness.py)
- iou: Hungarian matching at IoU 0.3 and 0.5
- count: label-agnostic F1 at IoU 0.3
- palette: players-only team assignment, tests both TEAM1/TEAM2 orientations
- role: average of object label score and team score
- smoothness: groups by bbox.label + cluster_id; flips destroy this
- keypoints: projects pitch lines into frame; checks overlap with field-line edges
- bowtie/spread projections, < 4 valid points, unrealistic masks => zero
```

### 3.3 Gate to Phase 1

You cannot start Phase 1 until `notes/scoring_spec.md` is complete and reviewed against the actual scoring source files. Read each scoring file twice.

---

## 4. Phase 1 — Repo Skeleton + Class Mapping Guard + Memory Budget (Days 2-3)

**Goal:** Buildable Python package with a thin Chute-compatible entrypoint. No detector yet. Just bones.

### 4.1 Initialize the dev workspace

```bash
mkdir -p validator_improve
cd validator_improve

# Clone references
git clone https://github.com/score-technologies/turbovision
git clone https://github.com/SoccerNet/sn-gamestate
git clone https://github.com/TrackingLaboratory/tracklab

# Create our work area
mkdir -p score_miner_dev score_miner
cd score_miner_dev

# Use uv (TurboVision uses it, stay consistent)
uv init --package score_miner_core
```

### 4.2 `pyproject.toml`

```toml
[project]
name = "score_miner_core"
version = "0.1.0"
description = "Score/TurboVision miner runtime"
requires-python = ">=3.10,<3.13"
dependencies = [
    "numpy>=1.26",
    "opencv-python-headless>=4.8",
    "torch>=2.4",
    "torchvision>=0.19",
    "pillow>=10",
    "pyyaml>=6",
    "pydantic>=2",
]

[project.optional-dependencies]
rfdetr = ["rfdetr>=1.4.1"]
deim   = []  # install from source per upstream README
dfine  = []  # install from source per upstream README
dev = [
    "pytest>=8",
    "optuna>=3.6",
    "mlflow>=2.14",
    "fiftyone>=0.24",
    "tqdm>=4.66",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
package = true
```

**Why `opencv-python-headless`:** TurboVision runs inside a containerized Chute. No display. Headless avoids X11 deps.

**Why pin Python 3.10–3.12:** RF-DETR requires Python ≥3.10. PyTorch 2.4 supports up to 3.12 reliably.

### 4.3 `score_miner_core/runtime/class_mapping.py`

This file alone has saved miners from week-long zero scores.

```python
"""Class mapping guard. The manifest defines element.objects in a specific order.
Our detector's class indices must be translated to manifest cls_id values.
Get this wrong = score zero. Verify on every load."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
from pathlib import Path

@dataclass(frozen=True)
class ManifestClassMap:
    """Maps detector class names to manifest cls_id integers."""
    name_to_cls_id: Dict[str, int]
    cls_id_to_name: Dict[int, str]
    n_classes: int

    @classmethod
    def from_manifest(cls, manifest_path: Path, element_id: str) -> "ManifestClassMap":
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        element = next(
            (e for e in manifest["elements"] if e["id"] == element_id),
            None,
        )
        if element is None:
            raise ValueError(f"Element {element_id} not in manifest {manifest_path}")

        objects = element["objects"]  # list of {name, cls_id, ...} or just names
        name_to_id: Dict[str, int] = {}
        for i, obj in enumerate(objects):
            if isinstance(obj, str):
                name_to_id[obj.lower()] = i
            else:
                name_to_id[obj["name"].lower()] = obj.get("cls_id", i)

        id_to_name = {v: k for k, v in name_to_id.items()}
        return cls(name_to_cls_id=name_to_id, cls_id_to_name=id_to_name, n_classes=len(name_to_id))

    def detector_to_manifest(self, detector_label: str) -> Optional[int]:
        """Return manifest cls_id or None if detector class is not in manifest."""
        return self.name_to_cls_id.get(detector_label.lower())

    def assert_compatible(self, required_classes: List[str]) -> None:
        """Fail fast if our miner doesn't cover required classes."""
        missing = [c for c in required_classes if c.lower() not in self.name_to_cls_id]
        if missing:
            raise RuntimeError(
                f"Detector cannot produce manifest classes: {missing}. "
                f"Manifest classes: {list(self.name_to_cls_id.keys())}"
            )
```

### 4.4 `score_miner_core/runtime/memory_budget.py`

The Chute template fails if loaded miner memory > 5.0 GB. Treat 4.5 GB as warning.

```python
"""Memory budget tracker. Chute template enforces 5.0 GB cap on loaded miner.
Use this on every model load. Fail fast in dev, not in deploy."""

from __future__ import annotations
import gc
import psutil
import torch
from contextlib import contextmanager
from dataclasses import dataclass

WARN_GB = 4.5
HARD_GB = 5.0

@dataclass
class MemorySnapshot:
    rss_gb: float
    gpu_alloc_gb: float
    gpu_reserved_gb: float

    def total_gb(self) -> float:
        return max(self.rss_gb, self.gpu_alloc_gb)

def snapshot() -> MemorySnapshot:
    process = psutil.Process()
    rss_gb = process.memory_info().rss / (1024**3)
    gpu_alloc_gb = 0.0
    gpu_reserved_gb = 0.0
    if torch.cuda.is_available():
        gpu_alloc_gb = torch.cuda.memory_allocated() / (1024**3)
        gpu_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
    return MemorySnapshot(rss_gb, gpu_alloc_gb, gpu_reserved_gb)

@contextmanager
def budget_guard(label: str, hard_gb: float = HARD_GB, warn_gb: float = WARN_GB):
    before = snapshot()
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        after = snapshot()
        delta = after.total_gb() - before.total_gb()
        if after.total_gb() > hard_gb:
            raise MemoryError(
                f"[{label}] memory {after.total_gb():.2f}GB > {hard_gb}GB cap. "
                f"Delta from this op: {delta:.2f}GB"
            )
        if after.total_gb() > warn_gb:
            print(
                f"[{label}] WARN memory {after.total_gb():.2f}GB > {warn_gb}GB. "
                f"Delta: {delta:.2f}GB"
            )
```

**Usage everywhere:**

```python
from score_miner_core.runtime.memory_budget import budget_guard

with budget_guard("rfdetr-l-load"):
    self.detector = RFDETRLarge(pretrain_weights=path)
```

### 4.5 `score_miner_core/runtime/determinism.py`

Nondeterminism kills replay reproduction and Optuna.

```python
"""Determinism helpers. Use at the top of every miner load."""

import os
import random
import numpy as np
import torch

def lock_determinism(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
```

### 4.6 The thin `score_miner/miner.py` entrypoint

```python
"""TurboVision Chute entrypoint. KEEP THIN. All logic lives in score_miner_core."""

from pathlib import Path
from score_miner_core.runtime.miner_runtime import MinerRuntime

class Miner:
    def __init__(self, path_hf_repo: Path | str):
        self.runtime = MinerRuntime(Path(path_hf_repo))

    def predict_batch(self, batch_images, offset: int, n_keypoints: int):
        return self.runtime.predict_batch(
            batch_images=batch_images,
            offset=offset,
            n_keypoints=n_keypoints,
        )
```

### 4.7 `chute_config.yml`

```yaml
# score_miner/chute_config.yml
name: score_miner_sina_v1
description: Score TurboVision miner — solo build by Sina
hf_repo: <your-hf-username>/score-miner-sina
python_version: "3.11"
pip:
  # whitelisted installed packages — Chute allows imports from these
  - "score_miner_core @ file://./dist/score_miner_core-0.1.0-py3-none-any.whl"
  - "rfdetr==1.4.1"
  - "torch>=2.4,<2.6"
  - "torchvision>=0.19"
  - "opencv-python-headless>=4.8"
  - "numpy>=1.26"
  - "pillow>=10"
  - "pyyaml>=6"
gpu:
  required: true
  min_vram_gb: 16
  preferred: "a10g"
```

### 4.8 Gate to Phase 2

```bash
cd score_miner_dev
uv build         # produces dist/score_miner_core-*.whl

# Verify
python -c "from score_miner_core.runtime.memory_budget import snapshot; print(snapshot())"
python -c "from score_miner_core.runtime.class_mapping import ManifestClassMap; print('ok')"
```

If both prints succeed, Phase 1 is done. If the wheel doesn't build, do not advance.

---

## 5. Phase 2 — Local Benchmark Harness (Day 4)

**Goal:** Have a deterministic local test that scores any miner config on the example video.

### 5.1 The minimum viable benchmark

`score_miner_core/benchmark/run_local.py`:

```python
"""Local benchmark: load test video, run miner, measure latency + schema validity.
No scoring yet — that comes in Phase 4 with validator_sim."""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np

from score_miner_core.runtime.memory_budget import snapshot

TEST_VIDEO = Path("../turbovision/tests/test_data/videos/example_football.mp4")
TEST_MANIFEST = Path("../turbovision/tests/test_data/manifests/example_manifest.yml")

def load_video_frames(path: Path, max_frames: int | None = None) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        frames.append(frame)  # BGR ndarray, what TurboVision passes
    cap.release()
    return frames

def percentile(values: List[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0

def schema_valid(result: dict, expected_frame_count: int, n_keypoints: int) -> tuple[bool, str]:
    if "frames" not in result:
        return False, "missing 'frames' key"
    if len(result["frames"]) != expected_frame_count:
        return False, f"frame count {len(result['frames'])} != expected {expected_frame_count}"
    for i, fr in enumerate(result["frames"]):
        if "frame_id" not in fr or "boxes" not in fr or "keypoints" not in fr:
            return False, f"frame {i} missing required keys"
        if len(fr["keypoints"]) != n_keypoints:
            return False, f"frame {i} keypoints {len(fr['keypoints'])} != {n_keypoints}"
        for j, b in enumerate(fr["boxes"]):
            for k in ("x1", "y1", "x2", "y2", "cls_id", "conf"):
                if k not in b:
                    return False, f"frame {i} box {j} missing {k}"
    return True, "ok"

def run_benchmark(MinerClass, path_hf_repo: Path,
                  batch_size: int = 64, n_keypoints: int = 32,
                  warmup: int = 1, runs: int = 3) -> Dict:
    mem_before = snapshot()
    miner = MinerClass(path_hf_repo)
    mem_after_load = snapshot()

    frames = load_video_frames(TEST_VIDEO, max_frames=batch_size * 2)
    print(f"loaded {len(frames)} frames")

    # warmup
    for _ in range(warmup):
        _ = miner.predict_batch(frames[:batch_size], offset=0, n_keypoints=n_keypoints)

    # timed runs
    latencies = []
    last_result = None
    for r in range(runs):
        t0 = time.perf_counter()
        last_result = miner.predict_batch(frames[:batch_size], offset=0, n_keypoints=n_keypoints)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)
        print(f"run {r}: {dt_ms:.1f}ms")

    mem_after_predict = snapshot()
    ok, msg = schema_valid(last_result, expected_frame_count=batch_size, n_keypoints=n_keypoints)

    return {
        "schema_ok": ok,
        "schema_msg": msg,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "mem_before_gb": mem_before.total_gb(),
        "mem_after_load_gb": mem_after_load.total_gb(),
        "mem_after_predict_gb": mem_after_predict.total_gb(),
        "model_load_delta_gb": mem_after_load.total_gb() - mem_before.total_gb(),
    }

if __name__ == "__main__":
    # placeholder Miner that returns valid empty output for Phase 2
    class StubMiner:
        def __init__(self, path_hf_repo): pass
        def predict_batch(self, batch_images, offset, n_keypoints):
            return {"frames": [
                {"frame_id": offset + i, "boxes": [], "keypoints": [[0, 0]] * n_keypoints}
                for i in range(len(batch_images))
            ]}

    result = run_benchmark(StubMiner, Path("./fake_hf_repo"))
    print(json.dumps(result, indent=2))
```

### 5.2 Gate to Phase 3

`run_local.py` runs and prints valid JSON. Schema validation passes with the stub miner. Memory snapshot works. p50/p95/p99 are real numbers.


---

## 6. Phase 3 — RF-DETR-M Smoke Deploy (Days 5-6)

**Goal:** Get a real model loaded, deployed to Chutes as `--no-commit` dry-run, returning valid JSON.

### 6.1 Why RF-DETR-M for smoke, not RF-DETR-L

Smoke tests answer **deployment** questions, not **score** questions:
- Does the wheel install in the Chute container?
- Does `score_miner.miner.Miner` load without import errors?
- Does the model fit under 5 GB?
- Does `/health` return 200?
- Does `/predict` produce valid JSON?
- Is p95 < 500ms on Chute's GPU type?

RF-DETR-M is ~30M params, fits comfortably, loads fast, runs at 4.4 ms on T4 FP16 TensorRT according to its own benchmark numbers. Perfect smoke target.

You will replace it with the winner of Phase 5's head-to-head. Don't fall in love with it.

### 6.2 `score_miner_core/detector/base.py`

```python
"""Detector abstraction. Every concrete detector (RF-DETR, DEIM, D-FINE)
implements this interface so we can swap them in Phase 5."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np

@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str       # detector-side label name, e.g. "player"
    score: float

class DetectorBase(ABC):
    @abstractmethod
    def __init__(self, weights_path: Path, device: str = "cuda"): ...

    @abstractmethod
    def predict_batch(
        self,
        images: List[np.ndarray],   # list of BGR ndarrays
        conf_thresh: float = 0.35,
    ) -> List[List[Detection]]: ...

    @property
    @abstractmethod
    def class_names(self) -> List[str]: ...
```

### 6.3 `score_miner_core/detector/rfdetr_runner.py`

Verified against rfdetr 1.4.1 docs (March 2026).

```python
"""RF-DETR runner. Uses official rfdetr package.
Verified API: rfdetr==1.4.1, March 2026."""

from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image

from .base import DetectorBase, Detection
from ..runtime.memory_budget import budget_guard

class RFDETRRunner(DetectorBase):
    """Wraps RFDETRMedium/Large via the official rfdetr Python package.

    Usage:
        det = RFDETRRunner(Path("./models/rfdetr_l.pth"), variant="large")
        results = det.predict_batch([bgr_frame_1, bgr_frame_2], conf_thresh=0.4)
    """

    def __init__(self, weights_path: Path, variant: str = "medium",
                 device: str = "cuda", resolution: int = 560):
        with budget_guard(f"rfdetr-{variant}-load"):
            if variant == "medium":
                from rfdetr import RFDETRMedium
                self.model = RFDETRMedium(pretrain_weights=str(weights_path),
                                          resolution=resolution)
            elif variant == "large":
                from rfdetr import RFDETRLarge
                self.model = RFDETRLarge(pretrain_weights=str(weights_path),
                                         resolution=resolution)
            elif variant == "small":
                from rfdetr import RFDETRSmall
                self.model = RFDETRSmall(pretrain_weights=str(weights_path),
                                         resolution=resolution)
            else:
                raise ValueError(f"unknown variant: {variant}")
            self.model.optimize_for_inference()

        self._class_names = list(self.model.class_names)  # provided by rfdetr 1.4.1
        self.device = device
        self.resolution = resolution

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def predict_batch(self, images: List[np.ndarray],
                      conf_thresh: float = 0.35) -> List[List[Detection]]:
        # rfdetr takes PIL or ndarray RGB. Our pipeline carries BGR — convert.
        pil_images = [Image.fromarray(im[..., ::-1]) for im in images]

        detections_per_image = self.model.predict(pil_images, threshold=conf_thresh)

        results: List[List[Detection]] = []
        for det_set in detections_per_image:
            # det_set has .xyxy (N,4), .confidence (N,), .class_id (N,)
            frame_dets = []
            for i in range(len(det_set.confidence)):
                x1, y1, x2, y2 = det_set.xyxy[i]
                frame_dets.append(Detection(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    label=self._class_names[int(det_set.class_id[i])],
                    score=float(det_set.confidence[i]),
                ))
            results.append(frame_dets)
        return results
```

**API note (verified May 2026):** `rfdetr 1.4.1` exposes `RFDETRMedium`, `RFDETRLarge`, etc. as classes. `model.predict()` returns a `supervision.Detections` object per image. Always pass `pretrain_weights=` for fine-tuned checkpoints, not the deprecated `pretrain_weights` positional arg.

### 6.4 `score_miner_core/runtime/miner_runtime.py` (Phase 3 version)

Phase 3 has no VideoState yet — just detector + naive pitch keypoints.

```python
"""MinerRuntime — the brain. Phase 3 version is intentionally simple:
detector + dummy keypoints. Real pipeline comes in Phase 10+."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from .class_mapping import ManifestClassMap
from .memory_budget import budget_guard
from .determinism import lock_determinism
from ..detector.rfdetr_runner import RFDETRRunner

# manifest constants — load real values from path_hf_repo/manifest_cache.yml in production
PLAYER = "player"
GOALKEEPER = "goalkeeper"
REFEREE = "referee"
BALL = "ball"
REQUIRED = [PLAYER, GOALKEEPER, REFEREE, BALL]

class MinerRuntime:
    def __init__(self, path_hf_repo: Path):
        lock_determinism(seed=42)
        manifest_path = path_hf_repo / "manifest_cache.yml"
        element_id = (path_hf_repo / "element_id.txt").read_text().strip()
        self.class_map = ManifestClassMap.from_manifest(manifest_path, element_id)
        self.class_map.assert_compatible(REQUIRED)

        with budget_guard("miner-runtime-init"):
            self.detector = RFDETRRunner(
                weights_path=path_hf_repo / "rfdetr_m.pth",
                variant="medium",
                resolution=560,
            )

        self.conf_threshold = 0.35

    def predict_batch(self, batch_images: List[np.ndarray],
                      offset: int, n_keypoints: int) -> Dict[str, Any]:
        # Run detector on whole batch
        per_frame_dets = self.detector.predict_batch(
            batch_images, conf_thresh=self.conf_threshold,
        )

        frames_out = []
        for i, dets in enumerate(per_frame_dets):
            frame_id = offset + i
            boxes_out = []
            for d in dets:
                cls_id = self.class_map.detector_to_manifest(d.label)
                if cls_id is None:
                    continue   # silently drop unknown classes
                boxes_out.append({
                    "x1": int(round(d.x1)),
                    "y1": int(round(d.y1)),
                    "x2": int(round(d.x2)),
                    "y2": int(round(d.y2)),
                    "cls_id": cls_id,
                    "conf": float(d.score),
                })
            frames_out.append({
                "frame_id": frame_id,
                "boxes": boxes_out,
                "keypoints": [[0, 0]] * n_keypoints,   # Phase 13 will fix this
            })
        return {"frames": frames_out}
```

### 6.5 Build, upload, dry-run deploy

```bash
# Build wheel and HF repo bundle
cd score_miner_dev
uv build
cp dist/score_miner_core-*.whl ../score_miner/dist/

# Download RF-DETR-M weights to score_miner/models/rfdetr_m.pth
# (Roboflow Hugging Face: rf-detr-medium)

# Cache the manifest snapshot you'll use
cp ../turbovision/tests/test_data/manifests/example_manifest.yml ../score_miner/manifest_cache.yml
echo "<your-element-id>" > ../score_miner/element_id.txt

# Push HF artifact
cd ../score_miner
huggingface-cli upload <your-hf-username>/score-miner-sina . --repo-type=model

# Dry-run deploy
cd ../turbovision
sv -v deploy-os-miner --model-path ../score_miner --element-id <ELEMENT_ID> --no-commit
```

### 6.6 Verify

```bash
curl -X POST https://<CHUTE_SLUG>.chutes.ai/health \
  -H "Authorization: Bearer $CHUTES_API_KEY"

curl -X POST https://<CHUTE_SLUG>.chutes.ai/predict \
  -d '{"url":"https://scoredata.me/2025_03_14/35ae7a/h1_0f2ca0.mp4","meta":{}}' \
  -H "Authorization: Bearer $CHUTES_API_KEY"

sv -vv run-once   # validator simulates a real run against your chute
```

### 6.7 Gate to Phase 4

- `/health` returns 200
- `/predict` returns valid TurboVision JSON
- `sv run-once` produces a score (any score, even low)
- p95 < 500ms
- Memory under 4.5 GB

**DO NOT commit on-chain yet.** Phase 4-7 must complete first.

---

## 7. Phase 4 — validator_sim (Days 7-8)

**Goal:** A local Python function that reproduces TurboVision's scoring on any (video, miner-output) pair. This is the moat. Everything else stands on this.

### 7.1 Critical principle

**Do not re-implement TurboVision's scoring math from scratch.** Import their actual scoring functions and call them directly. If their code changes, your sim changes for free.

### 7.2 `score_miner_core/validator_sim/pillar_metrics.py`

```python
"""Wrap TurboVision's own scoring functions. Do NOT reimplement."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List

TURBOVISION_ROOT = Path(__file__).resolve().parents[4] / "turbovision"
sys.path.insert(0, str(TURBOVISION_ROOT))

from scorevision.vlm_pipeline.non_vlm_scoring.objects import (  # noqa: E402
    score_objects,
)
from scorevision.vlm_pipeline.non_vlm_scoring.keypoints import (  # noqa: E402
    score_keypoints,
)
from scorevision.vlm_pipeline.non_vlm_scoring.smoothness import (  # noqa: E402
    score_smoothness,
)

def score_all_pillars(
    miner_output: Dict[str, Any],
    ground_truth: Dict[str, Any],
    pillar_weights: Dict[str, float],
) -> Dict[str, float]:
    """Returns per-pillar score and weighted total."""
    obj_metrics = score_objects(
        predictions=miner_output["frames"],
        ground_truth=ground_truth["frames"],
    )
    kp_metrics = score_keypoints(
        predictions=miner_output["frames"],
        ground_truth=ground_truth["frames"],
    )
    sm_metrics = score_smoothness(
        predictions=miner_output["frames"],
    )

    pillars = {
        "iou": obj_metrics.get("iou", 0.0),
        "count": obj_metrics.get("count_f1", 0.0),
        "palette": obj_metrics.get("palette", 0.0),
        "role": obj_metrics.get("role", 0.0),
        "smoothness": sm_metrics.get("smoothness", 0.0),
        "keypoints": kp_metrics.get("keypoints_iou", 0.0),
    }

    weighted_total = sum(
        pillars[name] * pillar_weights.get(name, 0.0) for name in pillars
    )
    return {**pillars, "weighted_total": weighted_total}
```

**Adjust the imports to whatever the TurboVision repo actually exposes.** In Phase 0 reading, you noted the actual function signatures. Match them exactly.

### 7.3 `score_miner_core/validator_sim/rtf_gate.py`

```python
"""RTF latency gate. Hard zero at the cap."""

def rtf(p95_latency_ms: float, service_rate_fps: int) -> float:
    return (p95_latency_ms / 10000.0) * (service_rate_fps / 5.0)

def latency_penalty(p95_latency_ms: float,
                    target_ms: float, hard_zero_ms: float) -> float:
    """Returns a multiplier in [0, 1]. 1.0 means no penalty; 0.0 means zero score."""
    if p95_latency_ms <= target_ms:
        return 1.0
    if p95_latency_ms >= hard_zero_ms:
        return 0.0
    # linear taper
    span = hard_zero_ms - target_ms
    over = p95_latency_ms - target_ms
    return max(0.0, 1.0 - (over / span))
```

### 7.4 `score_miner_core/validator_sim/report.py`

```python
"""End-to-end local scoring report."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict
from .pillar_metrics import score_all_pillars
from .rtf_gate import rtf, latency_penalty

@dataclass
class ScoreReport:
    pillars: Dict[str, float]
    weighted_total_raw: float
    weighted_total_final: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    rtf: float
    latency_penalty: float
    memory_gb: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def build_report(miner_output, ground_truth, pillar_weights,
                 latencies_ms, memory_gb,
                 target_ms: float, hard_zero_ms: float,
                 service_rate_fps: int) -> ScoreReport:
    import numpy as np
    pillars = score_all_pillars(miner_output, ground_truth, pillar_weights)
    raw = pillars.pop("weighted_total")

    p50 = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    p99 = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0
    pen = latency_penalty(p95, target_ms, hard_zero_ms)

    return ScoreReport(
        pillars=pillars,
        weighted_total_raw=raw,
        weighted_total_final=raw * pen,
        p50_ms=p50, p95_ms=p95, p99_ms=p99,
        rtf=rtf(p95, service_rate_fps),
        latency_penalty=pen,
        memory_gb=memory_gb,
    )
```

### 7.5 Validating validator_sim against the real validator

Run the same clip through your miner with `sv -vv run-once` (real validator) and via `validator_sim` (local). Compare each pillar score. If they differ by > 5%, your sim is wrong — investigate which function call is misaligned.

This is non-negotiable. Without sim-vs-real validation, every later experiment is on sand.

### 7.6 Gate to Phase 5

- `validator_sim` produces per-pillar scores on the test video
- Each pillar matches `sv run-once` within 5%
- Latency penalty curve matches the manifest spec

---

## 8. Phase 5 — Detector Head-to-Head (Days 9-11)

**Goal:** Decide RF-DETR-L vs DEIMv2-L vs D-FINE-L empirically. Pick the winner by `validator_sim` score, not by COCO AP.

### 8.1 Why this matters

The RF-DETR ICLR 2026 paper notes that **D-FINE was over-tuned to COCO validation set and underperforms on RF100-VL test set**. Conversely, RF-DETR shines specifically on RF100-VL but its larger sizes (L, XL) lose ground to D-FINE on raw COCO numbers. **Your task is neither COCO nor RF100-VL — it's SoccerNet broadcasts scored by a custom Hungarian-matching scorer with team/role/smoothness pillars.** None of the published benchmarks predict the winner.

Run the race.

### 8.2 Adapter pattern

Each detector implements `DetectorBase` from §6.2. Concrete files:

`score_miner_core/detector/deim_runner.py`:
```python
"""DEIMv2 wrapper. Install from source:
    git clone https://github.com/Intellindust-AI-Lab/DEIMv2
    cd DEIMv2 && pip install -e .
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import torch
from PIL import Image

from .base import DetectorBase, Detection
from ..runtime.memory_budget import budget_guard

class DEIMv2Runner(DetectorBase):
    def __init__(self, weights_path: Path, config_path: Path,
                 device: str = "cuda", resolution: int = 640):
        with budget_guard("deimv2-load"):
            # DEIMv2 uses config + checkpoint pattern (similar to RT-DETR)
            from deim.engine.core import YAMLConfig  # check actual import path
            cfg = YAMLConfig(str(config_path))
            cfg.resume = str(weights_path)
            self.model = cfg.model.to(device).eval()
            self.postprocessor = cfg.postprocessor

        self.device = device
        self.resolution = resolution
        self._class_names = self._load_class_names(config_path)

    def _load_class_names(self, config_path: Path) -> List[str]:
        # parse config — adapt to actual DEIMv2 config schema
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("class_names", [])

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @torch.no_grad()
    def predict_batch(self, images, conf_thresh: float = 0.35):
        # Convert BGR ndarrays to model input tensor batch
        batch = []
        sizes = []
        for im in images:
            sizes.append(im.shape[:2])  # (h, w)
            pil = Image.fromarray(im[..., ::-1])
            pil = pil.resize((self.resolution, self.resolution))
            arr = np.array(pil).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            arr = (arr - mean) / std
            batch.append(np.transpose(arr, (2, 0, 1)))
        tensor = torch.from_numpy(np.stack(batch)).float().to(self.device)

        outputs = self.model(tensor)
        orig_sizes = torch.tensor(sizes, device=self.device)
        results = self.postprocessor(outputs, orig_sizes)

        out_per_image: List[List[Detection]] = []
        for r, (h, w) in zip(results, sizes):
            frame_dets = []
            keep = r["scores"] > conf_thresh
            boxes = r["boxes"][keep].cpu().numpy()
            scores = r["scores"][keep].cpu().numpy()
            labels = r["labels"][keep].cpu().numpy()
            for b, s, l in zip(boxes, scores, labels):
                frame_dets.append(Detection(
                    x1=float(b[0]), y1=float(b[1]),
                    x2=float(b[2]), y2=float(b[3]),
                    label=self._class_names[int(l)],
                    score=float(s),
                ))
            out_per_image.append(frame_dets)
        return out_per_image
```

`score_miner_core/detector/dfine_runner.py`: same pattern, adapted to D-FINE's config/loading API (check upstream README).

### 8.3 The race harness

`score_miner_core/detector/detector_benchmark.py`:

```python
"""Detector head-to-head. Same test clips, same pre/postprocess,
same scorer, same memory measurement."""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Type
import numpy as np

from .base import DetectorBase
from ..benchmark.run_local import load_video_frames, percentile
from ..runtime.memory_budget import snapshot, budget_guard
from ..validator_sim.report import build_report

def race_one_detector(name: str,
                      DetectorCls: Type[DetectorBase],
                      weights_path: Path,
                      eval_clips: List[Path],
                      ground_truths: List[dict],
                      pillar_weights: Dict[str, float],
                      target_ms: float,
                      hard_zero_ms: float,
                      service_rate_fps: int,
                      conf_thresh: float = 0.35,
                      runs: int = 3,
                      **det_kwargs) -> Dict:
    mem_before = snapshot()
    with budget_guard(f"{name}-load"):
        det = DetectorCls(weights_path=weights_path, **det_kwargs)
    mem_after_load = snapshot()

    all_latencies = []
    pillar_totals = {}
    n_clips = 0

    for clip_path, gt in zip(eval_clips, ground_truths):
        frames = load_video_frames(clip_path)
        # measure latency over 3 runs per clip
        latencies = []
        last_dets = None
        for _ in range(runs):
            t0 = time.perf_counter()
            last_dets = det.predict_batch(frames, conf_thresh=conf_thresh)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        # Build a minimal miner-style output for scoring (no team/role yet)
        miner_output = _to_miner_schema(last_dets, det.class_names)
        report = build_report(
            miner_output=miner_output,
            ground_truth=gt,
            pillar_weights=pillar_weights,
            latencies_ms=latencies,
            memory_gb=snapshot().total_gb(),
            target_ms=target_ms,
            hard_zero_ms=hard_zero_ms,
            service_rate_fps=service_rate_fps,
        )

        all_latencies.extend(latencies)
        for k, v in report.pillars.items():
            pillar_totals[k] = pillar_totals.get(k, 0.0) + v
        n_clips += 1

    avg_pillars = {k: v / n_clips for k, v in pillar_totals.items()}
    return {
        "detector": name,
        "avg_pillars": avg_pillars,
        "p50_ms": percentile(all_latencies, 50),
        "p95_ms": percentile(all_latencies, 95),
        "p99_ms": percentile(all_latencies, 99),
        "model_load_gb": mem_after_load.total_gb() - mem_before.total_gb(),
        "peak_mem_gb": snapshot().total_gb(),
    }

def _to_miner_schema(per_frame_dets, class_names):
    # convert Detection objects -> TurboVision frames dict (no team/keypoints yet)
    out = {"frames": []}
    for i, dets in enumerate(per_frame_dets):
        boxes = []
        for d in dets:
            boxes.append({
                "x1": int(round(d.x1)), "y1": int(round(d.y1)),
                "x2": int(round(d.x2)), "y2": int(round(d.y2)),
                "cls_id": class_names.index(d.label),  # raw — class mapping is per-element
                "conf": float(d.score),
            })
        out["frames"].append({"frame_id": i, "boxes": boxes, "keypoints": [[0, 0]] * 32})
    return out
```

### 8.4 Held-out match split (read this twice)

**Hard rule: certain SoccerNet match IDs NEVER appear in any optimization or training loop.** They are only used for final evaluation.

Save the split:

```yaml
# score_miner_dev/score_miner_core/datasets/splits.yaml
heldout_match_ids:
  - SNGS-100
  - SNGS-112
  - SNGS-145
train_match_ids:
  - SNGS-001
  - SNGS-002
  # ... rest
val_match_ids:
  - SNGS-080
  - SNGS-081
```

The detector race runs on `val_match_ids`. The final evaluation runs on `heldout_match_ids`. Optuna in Phase 7 sees ONLY `val_match_ids`. If you let Optuna see held-out clips, you will overfit and underperform on real validator queries.

### 8.5 Decision matrix template

After running, fill in `notes/detector_race_results.md`:

```markdown
# Detector Race Results — <date>

## Config
- Eval set: <N> clips from val_match_ids
- Resolution: 560 / 640
- Conf threshold: 0.35
- Memory cap: 5.0 GB

## Results

| Detector | Weighted Total | iou | count | role | palette | smoothness | keypoints | p50 ms | p95 ms | p99 ms | Load GB | Peak GB |
|----------|---------------:|----:|------:|-----:|--------:|-----------:|----------:|-------:|-------:|-------:|--------:|--------:|
| RF-DETR-L|                |     |       |      |         |            |           |        |        |        |         |         |
| DEIMv2-L |                |     |       |      |         |            |           |        |        |        |         |         |
| D-FINE-L |                |     |       |      |         |            |           |        |        |        |         |         |

## Decision
Winner: <X>
Rationale: <why — must reference numbers above, not COCO AP>

## Failure profiles
- RF-DETR-L misses: <e.g. small ball in air>
- DEIMv2-L misses: <e.g. crowded penalty box>
- D-FINE-L misses: <e.g. closeup replay>
```

### 8.6 Gate to Phase 6

- Three detectors raced under identical conditions
- Winner chosen and documented
- Loser detectors documented with their failure profiles (you'll need them in Phase 19 as teachers for pseudo-labels)


---

## 9. Phase 6 — Replay Mining (Day 12)

**Goal:** Every run, every failure, preserved with full context. Becomes the training set for Phase 17.

### 9.1 `score_miner_core/replay/failure_store.py`

```python
"""Replay store. Save every run's failures with enough context to reproduce."""

from __future__ import annotations
import json
import time
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2

@dataclass
class ReplayCase:
    case_id: str             # e.g. "20260518-141522-clip-SNGS-045-batch-0"
    clip_id: str
    batch_offset: int
    score_total: float
    pillar_scores: Dict[str, float]
    latencies_ms: List[float]
    failure_tags: List[str]  # ball_miss, team_flip, ...
    miner_output_path: str
    ground_truth_path: str
    frames_path: str
    notes: str = ""
    config_hash: str = ""
    model_hash: str = ""
    timestamp: float = field(default_factory=time.time)

FAILURE_TAGS = {
    "ball_miss",
    "ball_false_positive",
    "player_count_low",
    "player_count_high",
    "team_flip",
    "role_confusion",
    "homography_invalid",
    "smoothness_drop",
    "scene_cut_oversmooth",
    "latency_spike",
    "frame_count_error",
    "schema_failure",
}

class FailureStore:
    def __init__(self, root: Path):
        self.root = root
        (root / "cases").mkdir(parents=True, exist_ok=True)
        (root / "index.jsonl").touch(exist_ok=True)

    def archive(self, case: ReplayCase,
                miner_output: Dict[str, Any],
                ground_truth: Dict[str, Any],
                frames: Optional[List["np.ndarray"]] = None) -> None:
        case_dir = self.root / "cases" / case.case_id
        case_dir.mkdir(exist_ok=True)

        with open(case_dir / "miner_output.json", "w") as f:
            json.dump(miner_output, f, indent=2)
        with open(case_dir / "ground_truth.json", "w") as f:
            json.dump(ground_truth, f, indent=2)

        if frames:
            frames_dir = case_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            for i, fr in enumerate(frames):
                cv2.imwrite(str(frames_dir / f"{i:06d}.jpg"), fr,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
            case.frames_path = str(frames_dir)

        case.miner_output_path = str(case_dir / "miner_output.json")
        case.ground_truth_path = str(case_dir / "ground_truth.json")

        with open(self.root / "index.jsonl", "a") as f:
            f.write(json.dumps(asdict(case)) + "\n")

    def find_by_tag(self, tag: str) -> List[ReplayCase]:
        results = []
        with open(self.root / "index.jsonl") as f:
            for line in f:
                d = json.loads(line)
                if tag in d.get("failure_tags", []):
                    results.append(ReplayCase(**d))
        return results
```

### 9.2 Failure detection rules

`score_miner_core/replay/clip_sampler.py` should auto-tag a run:

```python
def detect_failure_tags(report, miner_output, ground_truth) -> List[str]:
    tags = []
    if report.p95_ms > 1.5 * 200:   # 50% over target
        tags.append("latency_spike")
    if report.pillars.get("smoothness", 1.0) < 0.4:
        tags.append("smoothness_drop")
    if report.pillars.get("palette", 1.0) < 0.5:
        tags.append("team_flip")
    if report.pillars.get("role", 1.0) < 0.5:
        tags.append("role_confusion")
    if report.pillars.get("keypoints", 1.0) < 0.3:
        tags.append("homography_invalid")

    # ball-specific
    gt_balls = sum(1 for f in ground_truth["frames"]
                   for b in f["boxes"] if b["cls_id"] == 3)
    pred_balls = sum(1 for f in miner_output["frames"]
                     for b in f["boxes"] if b["cls_id"] == 3)
    if gt_balls > 0 and pred_balls < 0.5 * gt_balls:
        tags.append("ball_miss")
    if pred_balls > 2 * max(gt_balls, 1):
        tags.append("ball_false_positive")

    return tags
```

### 9.3 Gate to Phase 7

Run validator_sim on 10 val clips. The replay store has 10 entries. Each entry has miner output, ground truth, latency trace, and failure tags. Open one in FiftyOne (Phase 12 wires it up properly; for now, just open the frames manually) to verify it's reproducible.

---

## 10. Phase 7 — Optuna Optimizer (Days 13-14)

**Goal:** Search the config space for the best score. No retraining yet — just thresholds, scheduler params, smoothing weights.

### 10.1 The search space

`score_miner_core/optimizer_core/search_space.py`:

```python
"""Optuna search space. Tune the things that move TurboVision score."""

import optuna

def suggest_config(trial: optuna.Trial) -> dict:
    return {
        # detector
        "detector": trial.suggest_categorical("detector", ["rfdetr_l", "deimv2_l", "dfine_l"]),
        "conf_player": trial.suggest_float("conf_player", 0.25, 0.65),
        "conf_ball": trial.suggest_float("conf_ball", 0.15, 0.55),
        "conf_referee": trial.suggest_float("conf_referee", 0.30, 0.70),
        "conf_goalkeeper": trial.suggest_float("conf_goalkeeper", 0.30, 0.70),
        "max_boxes_per_frame": trial.suggest_int("max_boxes_per_frame", 22, 35),

        # tracker / smoothing
        "tracker_iou_thresh": trial.suggest_float("tracker_iou_thresh", 0.2, 0.6),
        "tracker_appearance_weight": trial.suggest_float("tracker_appearance_weight", 0.3, 0.9),
        "box_smoothing_alpha": trial.suggest_float("box_smoothing_alpha", 0.0, 0.7),

        # team / role hysteresis
        "team_flip_hysteresis_frames": trial.suggest_int("team_flip_hysteresis_frames", 3, 20),
        "role_hysteresis_frames": trial.suggest_int("role_hysteresis_frames", 3, 15),

        # ball
        "ball_temporal_confirm_window": trial.suggest_int("ball_temporal_confirm_window", 2, 8),
        "ball_crop_scale": trial.suggest_float("ball_crop_scale", 1.2, 2.5),

        # keypoints
        "keypoint_conf_thresh": trial.suggest_float("keypoint_conf_thresh", 0.2, 0.7),
        "homography_confidence_min": trial.suggest_float("homography_confidence_min", 0.4, 0.8),

        # adaptive scheduler
        "detector_refresh_cadence": trial.suggest_int("detector_refresh_cadence", 1, 10),
        "scene_cut_threshold": trial.suggest_float("scene_cut_threshold", 0.2, 0.6),
        "high_res_crop_trigger": trial.suggest_float("high_res_crop_trigger", 0.3, 0.7),
    }
```

### 10.2 Objective

`score_miner_core/optimizer_core/objective.py`:

```python
"""Optuna objective. Maximize weighted TurboVision score with penalties."""

import json
import time
from pathlib import Path
from typing import Callable, Dict
import optuna

def make_objective(
    miner_factory: Callable[[dict], object],
    eval_clips: list,
    ground_truths: list,
    pillar_weights: Dict[str, float],
    target_ms: float,
    hard_zero_ms: float,
    service_rate_fps: int,
    trial_log: Path,
) -> Callable[[optuna.Trial], float]:
    from .search_space import suggest_config
    from ..validator_sim.report import build_report
    from ..benchmark.run_local import load_video_frames

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_config(trial)
        miner = miner_factory(cfg)

        all_pillars = {k: 0.0 for k in [
            "iou", "count", "palette", "role", "smoothness", "keypoints",
        ]}
        all_latencies = []

        for clip, gt in zip(eval_clips, ground_truths):
            frames = load_video_frames(Path(clip))
            t0 = time.perf_counter()
            out = miner.predict_batch(frames, offset=0, n_keypoints=32)
            dt = (time.perf_counter() - t0) * 1000.0
            all_latencies.append(dt)

            report = build_report(
                miner_output=out, ground_truth=gt,
                pillar_weights=pillar_weights,
                latencies_ms=[dt], memory_gb=0.0,
                target_ms=target_ms, hard_zero_ms=hard_zero_ms,
                service_rate_fps=service_rate_fps,
            )
            for k in all_pillars:
                all_pillars[k] += report.pillars.get(k, 0.0) / len(eval_clips)

        avg_p95 = sorted(all_latencies)[int(0.95 * (len(all_latencies)-1))]
        weighted = sum(pillar_weights.get(k, 0.0) * v for k, v in all_pillars.items())

        # Latency penalty
        if avg_p95 > hard_zero_ms:
            score = 0.0
        elif avg_p95 > target_ms:
            score = weighted * (1.0 - (avg_p95 - target_ms) / (hard_zero_ms - target_ms))
        else:
            score = weighted

        # Persist trial
        with open(trial_log, "a") as f:
            f.write(json.dumps({
                "trial": trial.number,
                "params": cfg,
                "pillars": all_pillars,
                "p95_ms": avg_p95,
                "score": score,
            }) + "\n")

        return score

    return objective
```

### 10.3 Runner

`score_miner_core/optimizer_core/optuna_runner.py`:

```python
import optuna
from pathlib import Path

def run_study(objective, n_trials: int = 200, study_name: str = "score-miner") -> optuna.Study:
    storage = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
    )
    study.optimize(objective, n_trials=n_trials)
    print("best params:", study.best_params)
    print("best value:", study.best_value)
    return study
```

### 10.4 Critical: prevent overfitting

- Optuna eval set = `val_match_ids` only.
- `heldout_match_ids` never seen during optimization.
- Final winning config must be re-scored on heldout. If heldout score is >15% lower than Optuna best, you overfit — increase val set size or pull back search space.

### 10.5 Gate to Phase 8

- 200 trials run
- Best config beats baseline by >5%
- Heldout score within 10% of Optuna best (no overfit)

---

## 11. Phase 8 — sn-gamestate / TrackLab Reference Pass (Days 15-16)

**Goal:** Read what the SoccerNet winners actually shipped. Identify which modules to adapt, with cost estimates.

### 11.1 What to read

```bash
cd sn-gamestate
cat README.md
cat sn_gamestate/configs/soccernet.yaml
ls sn_gamestate/                          # see module categories
cat sn_gamestate/jersey_number/*.py       # PARSeq-based jersey OCR
cat sn_gamestate/team/*.py                # team affiliation
cat sn_gamestate/pitch/*.py               # camera calibration

cd ../tracklab
cat tracklab/configs/config.yaml
cat tracklab/configs/dataset/soccernet_mot.yaml
ls tracklab/modules/                      # reid, tracker, detector
cat tracklab/modules/reid/bpbreid.py
cat tracklab/modules/reid/torchreid.py    # CLIP-ReIdent, OSNet wrappers
cat tracklab/modules/tracker/bot_sort.py
cat tracklab/modules/tracker/strong_sort.py
```

### 11.2 What to extract

Fill in `external/sn_gamestate_notes.md`:

```markdown
# sn-gamestate module inventory

## Detector modules
- bbox_detector wrapper (Hydra)
- Supports: YOLO, YOLOX, RT-DETR. Trivial to add RF-DETR/DEIMv2.

## ReID modules (most valuable for us)
- BPBreID
- TorchReID (OSNet)
- PRTReID
- Cost: model is ~30M params, 200ms per batch of 64 — borderline

## Tracker modules
- BoT-SORT (with camera-motion compensation)
- StrongSORT
- DeepOCSORT
- ByteTrack
- Cost: pure CPU, negligible latency

## Pitch calibration
- TVCalib (SegFormer-based)
- 74-point keypoint refinement from Broadcast-to-Minimap 2024 winners
- Cost: SegFormer ~50M params, 80ms per frame — runs every 5-10 frames

## Jersey number recognition (Koshkina/PARSeq)
- PARSeq OCR model
- Cost: ~40M params, fast on torso crops, 30ms per crop
- Selective live use only (only on stable tracklets)

## What to adapt vs reimplement
- ADAPT: BoT-SORT (drop into score_miner_core/tracking/)
- ADAPT: TVCalib SegFormer head (after Phase 13)
- ADAPT: PARSeq jersey OCR (selective, Phase 15)
- BUILD FRESH: Hydra config glue — too heavy for Chute
- BUILD FRESH: Team color clustering — simple HSV, no point importing
```

### 11.3 Important: DO NOT install full sn-gamestate into the Chute

The TrackLab + sn-gamestate dep tree is huge (Hydra, omegaconf, mmcv variants, etc.). The Chute has 5GB memory budget and limited install time. **Use them as code references and copy the specific modules you need.**

### 11.4 Gate to Phase 9

- `external/sn_gamestate_notes.md` complete
- Decision: which 3-5 modules to adapt with their memory/latency estimates

---

## 12. Phase 9 — Team/Role Logic (Days 17-18)

**Goal:** Stop team flips. Stop GK/ref confusion. Win the `palette` and `role` pillars.

### 12.1 Why this matters

TurboVision's `palette` and `role` pillars **test both TEAM1/TEAM2 orientations**, so absolute team labels don't matter — but **stability across frames does**. A team_id that flips on one shadowed frame nukes palette and smoothness simultaneously.

### 12.2 Two-centroid clustering with temporal memory

`score_miner_core/team/jersey_cluster.py`:

```python
"""Players-only two-team clustering on torso color.
Phase 9 baseline before ReID upgrade in Phase 15."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2

@dataclass
class ColorCentroid:
    lab: np.ndarray  # shape (3,)
    count: int = 0

def torso_crop(frame_bgr: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    """Extract upper-half-minus-head crop. Players' jerseys live here.
    Strip the top 15% (head/hair) and bottom 50% (shorts/grass)."""
    h = y2 - y1
    top = int(y1 + 0.15 * h)
    bot = int(y1 + 0.50 * h)
    return frame_bgr[max(0, top):min(frame_bgr.shape[0], bot),
                     max(0, x1):min(frame_bgr.shape[1], x2)]

def grass_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Mask out grass pixels (green) from torso crop."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # green range — adapt for floodlights/night matches
    green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    return cv2.bitwise_not(green)

def torso_lab_mean(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr.size == 0:
        return np.array([0.0, 128.0, 128.0])
    mask = grass_mask(crop_bgr)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    valid = lab[mask > 0]
    if len(valid) < 10:
        return np.array([0.0, 128.0, 128.0])
    return valid.mean(axis=0)

class TwoTeamClusterer:
    """Maintains two team centroids with EMA across frames.
    Initializes via k-means on the first stable batch."""

    def __init__(self, alpha: float = 0.1, flip_hysteresis: int = 10):
        self.team_a: Optional[ColorCentroid] = None
        self.team_b: Optional[ColorCentroid] = None
        self.alpha = alpha
        self.flip_hysteresis = flip_hysteresis
        self._initialized = False

    def initialize(self, lab_vectors: List[np.ndarray]) -> None:
        if len(lab_vectors) < 8:
            return
        arr = np.stack(lab_vectors)
        # k=2 k-means
        _, labels, centers = cv2.kmeans(
            arr.astype(np.float32), 2, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
            5, cv2.KMEANS_PP_CENTERS,
        )
        self.team_a = ColorCentroid(lab=centers[0], count=int((labels == 0).sum()))
        self.team_b = ColorCentroid(lab=centers[1], count=int((labels == 1).sum()))
        self._initialized = True

    def assign(self, lab: np.ndarray) -> Tuple[int, float]:
        """Returns (team_id in {0,1}, confidence in [0,1])."""
        if not self._initialized:
            return 0, 0.0
        d_a = np.linalg.norm(lab - self.team_a.lab)
        d_b = np.linalg.norm(lab - self.team_b.lab)
        if d_a + d_b < 1e-6:
            return 0, 0.0
        if d_a < d_b:
            conf = (d_b - d_a) / (d_a + d_b)
            return 0, float(conf)
        conf = (d_a - d_b) / (d_a + d_b)
        return 1, float(conf)

    def update(self, team_id: int, lab: np.ndarray) -> None:
        c = self.team_a if team_id == 0 else self.team_b
        c.lab = (1 - self.alpha) * c.lab + self.alpha * lab
        c.count += 1
```

### 12.3 Per-track team hysteresis

In VideoState (Phase 10), each track gets:

```python
@dataclass
class TeamVote:
    team_a_votes: int = 0
    team_b_votes: int = 0
    last_committed: Optional[int] = None
    frames_since_commit: int = 0

def commit_team(self, vote: TeamVote, hysteresis_frames: int) -> int:
    """Only commit a team flip if the new team has held majority for N frames."""
    candidate = 0 if vote.team_a_votes > vote.team_b_votes else 1
    if vote.last_committed is None:
        vote.last_committed = candidate
        return candidate
    if candidate != vote.last_committed:
        vote.frames_since_commit += 1
        if vote.frames_since_commit >= hysteresis_frames:
            vote.last_committed = candidate
            vote.frames_since_commit = 0
    else:
        vote.frames_since_commit = 0
    return vote.last_committed
```

### 12.4 Role cleanup

`score_miner_core/team/role_cleanup.py`:

```python
"""GK/referee disambiguation.

Heuristics that work without ReID:
- Referees usually wear distinct color (yellow/black) — separate cluster from team A/B
- Goalkeepers stay in defensive third — use pitch homography (Phase 13) for position prior
- Both have stable role over a match — never flip role within a track once committed
"""

def cleanup_role(track, homography, pitch_size_m=(105, 68)) -> str:
    """Returns 'player' | 'goalkeeper' | 'referee'."""
    if track.role_committed:
        return track.role  # never flip

    # color distance from team centroids
    if track.team_color_distance > 2.0 * track.team_color_distance_to_own_team:
        # color isolated from both teams — likely referee
        track.role_referee_votes += 1

    if homography is not None and track.last_bbox is not None:
        # project center to pitch coords
        cx = (track.last_bbox.x1 + track.last_bbox.x2) / 2
        cy = track.last_bbox.y2  # foot point
        pitch_x, pitch_y = homography.project(cx, cy)
        # GKs hover near goals (x < 5m or x > 100m)
        if pitch_x < 5.0 or pitch_x > pitch_size_m[0] - 5.0:
            track.role_gk_votes += 1

    if track.frames_seen >= 30:  # commit after 30 frames
        votes = {
            "player": track.frames_seen - track.role_referee_votes - track.role_gk_votes,
            "referee": track.role_referee_votes,
            "goalkeeper": track.role_gk_votes,
        }
        track.role = max(votes, key=votes.get)
        track.role_committed = True
    return track.role
```

### 12.5 Gate to Phase 10

- `validator_sim` shows palette pillar ≥ 0.6 on val set
- Team flip rate < 1 per 100 frames on stable wide shots
- Role accuracy ≥ 80% on annotated SoccerNet GSR val clips


---

## 13. Phase 10 — VideoState + Scene Analyzer (Days 19-21)

**Goal:** Persistent world model. Stop being stateless. This is the single biggest score lift after detector.

### 13.1 The VideoState object

`score_miner_core/runtime/video_state.py`:

```python
"""VideoState — persistent world model across frames.

Lives for one chunk/video. Reset on scene cut. Bounded memory."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

@dataclass
class BBoxRecord:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    frame_id: int

@dataclass
class PlayerTrack:
    track_id: int
    bbox_history: Deque[BBoxRecord] = field(default_factory=lambda: deque(maxlen=60))
    velocity: Tuple[float, float] = (0.0, 0.0)
    reid_embedding_ema: Optional[np.ndarray] = None
    team_id: Optional[int] = None
    team_votes: List[int] = field(default_factory=list)  # last N frame team votes
    team_committed: bool = False
    role: str = "player"   # player | goalkeeper | referee
    role_committed: bool = False
    jersey_number_votes: Dict[int, int] = field(default_factory=dict)
    last_seen_frame: int = -1
    active: bool = True

@dataclass
class BallTrack:
    bbox_history: Deque[BBoxRecord] = field(default_factory=lambda: deque(maxlen=30))
    trajectory_x: Deque[float] = field(default_factory=lambda: deque(maxlen=15))
    trajectory_y: Deque[float] = field(default_factory=lambda: deque(maxlen=15))
    confidence: float = 0.0
    last_seen_frame: int = -1

@dataclass
class HomographyState:
    H: Optional[np.ndarray] = None        # 3x3 matrix
    confidence: float = 0.0
    last_valid_frame: int = -1
    keypoints_used: List[Tuple[int, float, float]] = field(default_factory=list)

@dataclass
class SceneState:
    scene_id: int = 0
    camera_type: str = "wide"  # wide | medium | closeup | replay
    camera_motion: Tuple[float, float] = (0.0, 0.0)
    last_scene_cut_frame: int = -1

@dataclass
class TeamPalette:
    centroid_a: Optional[np.ndarray] = None    # Lab color
    centroid_b: Optional[np.ndarray] = None
    confidence: float = 0.0
    initialized: bool = False

@dataclass
class VideoState:
    video_id: str = ""
    frame_count: int = 0
    next_track_id: int = 1
    player_tracks: Dict[int, PlayerTrack] = field(default_factory=dict)
    ball: BallTrack = field(default_factory=BallTrack)
    homography: HomographyState = field(default_factory=HomographyState)
    scene: SceneState = field(default_factory=SceneState)
    palette: TeamPalette = field(default_factory=TeamPalette)
    confidence_history: Deque[float] = field(default_factory=lambda: deque(maxlen=120))

    def memory_budget_check(self, max_tracks: int = 50) -> None:
        if len(self.player_tracks) > max_tracks:
            # drop oldest inactive tracks
            inactive = [
                (tid, t) for tid, t in self.player_tracks.items()
                if not t.active and self.frame_count - t.last_seen_frame > 30
            ]
            inactive.sort(key=lambda kv: kv[1].last_seen_frame)
            drop_n = len(self.player_tracks) - max_tracks
            for tid, _ in inactive[:drop_n]:
                del self.player_tracks[tid]

    def reset_on_scene_cut(self) -> None:
        """Soft reset — keep team palette, drop trackers."""
        self.player_tracks.clear()
        self.ball = BallTrack()
        # keep palette and homography — same match, just camera change
        self.scene.last_scene_cut_frame = self.frame_count
        self.scene.scene_id += 1
```

### 13.2 Scene cut detection

`score_miner_core/runtime/scene_analyzer.py`:

```python
"""Scene cut detection via color histogram difference and optical flow magnitude."""

from __future__ import annotations
import cv2
import numpy as np

def color_hist(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()

class SceneAnalyzer:
    def __init__(self, cut_threshold: float = 0.4):
        self.prev_hist: np.ndarray | None = None
        self.prev_gray: np.ndarray | None = None
        self.cut_threshold = cut_threshold

    def step(self, frame_bgr: np.ndarray) -> dict:
        """Returns {'scene_cut': bool, 'motion_mag': float, 'camera_type': str}."""
        hist = color_hist(frame_bgr)
        cut = False
        if self.prev_hist is not None:
            d = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if d > self.cut_threshold:
                cut = True
        self.prev_hist = hist

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        motion_mag = 0.0
        if self.prev_gray is not None and not cut:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            motion_mag = float(np.linalg.norm(flow, axis=2).mean())
        self.prev_gray = gray

        # very rough camera type heuristic
        h, w = frame_bgr.shape[:2]
        camera_type = "wide" if w >= 1280 else "medium"
        if motion_mag > 15.0:
            camera_type = "closeup"   # often handheld/close

        return {"scene_cut": cut, "motion_mag": motion_mag, "camera_type": camera_type}
```

### 13.3 IoU association tracker (minimal)

`score_miner_core/tracking/association.py`:

```python
"""IoU-based association. Wired in Phase 10. Upgrades to BoT-SORT in Phase 15."""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)

def associate(detections: List[tuple], tracks: List[tuple],
              iou_thresh: float = 0.3) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Hungarian IoU matching. Returns matches, unmatched_detections, unmatched_tracks."""
    if not detections or not tracks:
        return [], list(range(len(detections))), list(range(len(tracks)))

    iou_matrix = np.zeros((len(detections), len(tracks)))
    for i, d in enumerate(detections):
        for j, t in enumerate(tracks):
            iou_matrix[i, j] = iou(d, t)

    cost = -iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_thresh:
            matches.append((r, c))

    matched_d = {r for r, _ in matches}
    matched_t = {c for _, c in matches}
    unmatched_d = [i for i in range(len(detections)) if i not in matched_d]
    unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
    return matches, unmatched_d, unmatched_t
```

### 13.4 Wire VideoState into MinerRuntime

```python
# inside score_miner_core/runtime/miner_runtime.py

def predict_batch(self, batch_images, offset, n_keypoints):
    if self.video_state.video_id == "" or self.video_state.frame_count == 0:
        self.video_state.video_id = f"chunk-{offset}"

    frames_out = []
    for i, frame in enumerate(batch_images):
        frame_id = offset + i
        self.video_state.frame_count = frame_id

        # 1. scene analysis
        scene_info = self.scene_analyzer.step(frame)
        if scene_info["scene_cut"]:
            self.video_state.reset_on_scene_cut()

        # 2. detection (Phase 11 will make this adaptive)
        dets = self.detector.predict_batch([frame], conf_thresh=self.cfg["conf_player"])[0]

        # 3. association
        track_bboxes = [
            (t.bbox_history[-1].x1, t.bbox_history[-1].y1,
             t.bbox_history[-1].x2, t.bbox_history[-1].y2)
            for t in self.video_state.player_tracks.values()
            if t.active
        ]
        det_bboxes = [(d.x1, d.y1, d.x2, d.y2) for d in dets if d.label == "player"]
        matches, unmatched_d, unmatched_t = associate(det_bboxes, track_bboxes,
                                                      iou_thresh=self.cfg["tracker_iou_thresh"])
        # ... update tracks, create new ones for unmatched_d, mark inactive for unmatched_t ...

        # 4. team assignment (uses §12)
        # 5. role cleanup
        # 6. ball update
        # 7. emit frame output
```

### 13.5 Gate to Phase 11

- Smoothness pillar ≥ 0.7 on val set (was ~0.3 with stateless miner)
- Memory under 4.5 GB after 5 minutes of continuous predict_batch calls
- Optuna re-run shows VideoState params (team_flip_hysteresis_frames, etc.) actually move score

---

## 14. Phase 11 — Adaptive Scheduler (Days 22-23)

**Goal:** Don't run heavy inference on every frame. Cut p95 by 30-50% with no score loss.

### 14.1 The router

`score_miner_core/runtime/scheduler.py`:

```python
"""Adaptive compute scheduler. Routes work based on uncertainty + scene."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List

class ComputeAction(Enum):
    FULL_DETECT = "full_detect"               # main detector at full resolution
    TRACK_ONLY = "track_only"                 # Kalman + IoU, no detector
    HIGH_RES_PASS = "high_res_pass"          # crop + zoom for tiny ball
    KEYPOINT_REFRESH = "keypoint_refresh"     # re-run pitch model
    BALL_REFINER = "ball_refiner"             # tiled ball-only inference

@dataclass
class FrameDecision:
    actions: List[ComputeAction]
    reason: str

class AdaptiveScheduler:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.frames_since_detect = 999
        self.frames_since_keypoint = 999

    def decide(self, frame_id: int, scene_info: dict, video_state) -> FrameDecision:
        actions = []
        reasons = []

        # Always full-detect on scene cut
        if scene_info["scene_cut"]:
            actions.append(ComputeAction.FULL_DETECT)
            actions.append(ComputeAction.KEYPOINT_REFRESH)
            self.frames_since_detect = 0
            self.frames_since_keypoint = 0
            return FrameDecision(actions, "scene_cut")

        # Detector refresh cadence
        cadence = self.cfg.get("detector_refresh_cadence", 3)
        if self.frames_since_detect >= cadence:
            actions.append(ComputeAction.FULL_DETECT)
            self.frames_since_detect = 0
            reasons.append(f"cadence({cadence})")
        else:
            actions.append(ComputeAction.TRACK_ONLY)
            self.frames_since_detect += 1
            reasons.append("track_advance")

        # Keypoint refresh on homography drift
        kp_cadence = self.cfg.get("keypoint_refresh_cadence", 10)
        if (self.frames_since_keypoint >= kp_cadence
                or video_state.homography.confidence < self.cfg.get("homography_confidence_min", 0.5)):
            actions.append(ComputeAction.KEYPOINT_REFRESH)
            self.frames_since_keypoint = 0
            reasons.append("keypoint_refresh")
        else:
            self.frames_since_keypoint += 1

        # Ball uncertainty
        if video_state.ball.confidence < 0.5 and video_state.ball.last_seen_frame >= frame_id - 5:
            actions.append(ComputeAction.BALL_REFINER)
            reasons.append("ball_uncertain")

        # Penalty box crowd — heuristic via player density
        active = [t for t in video_state.player_tracks.values() if t.active]
        if len(active) >= 12:
            actions.append(ComputeAction.HIGH_RES_PASS)
            reasons.append("crowd")

        return FrameDecision(actions, "+".join(reasons))
```

### 14.2 Gate to Phase 12

- p95 drops by 30% on val set
- Score does NOT drop (all pillars within 2% of pre-scheduler)
- Optuna shows `detector_refresh_cadence` between 3-5 as optimal

---

## 15. Phase 12 — Ball Specialist (Days 24-25)

**Goal:** Catch tiny balls without flooding false positives. Ball pillar is high-impact and many miners ignore it.

### 15.1 The high-res tile pass

`score_miner_core/ball/crop_refiner.py`:

```python
"""When the main detector is uncertain about the ball, tile the frame
at higher effective resolution and re-detect with a stricter ball threshold."""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from ..detector.base import DetectorBase, Detection

def tile_frame(frame: np.ndarray, tile_size: int = 640, overlap: int = 80
               ) -> List[Tuple[int, int, np.ndarray]]:
    """Returns list of (x_offset, y_offset, tile) for overlapping tiles."""
    h, w = frame.shape[:2]
    tiles = []
    step = tile_size - overlap
    for y in range(0, max(1, h - overlap), step):
        for x in range(0, max(1, w - overlap), step):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            tile = frame[y:y2, x:x2]
            tiles.append((x, y, tile))
    return tiles

class BallCropRefiner:
    def __init__(self, detector: DetectorBase, tile_size: int = 640,
                 conf_thresh: float = 0.5):
        self.detector = detector
        self.tile_size = tile_size
        self.conf_thresh = conf_thresh

    def refine(self, frame: np.ndarray,
               trajectory_hint: Optional[Tuple[float, float]] = None
               ) -> Optional[Detection]:
        """If trajectory_hint given, only run on the tile containing that point."""
        tiles = tile_frame(frame, self.tile_size)
        if trajectory_hint is not None:
            tx, ty = trajectory_hint
            tiles = [(x, y, t) for x, y, t in tiles
                     if x <= tx < x + t.shape[1] and y <= ty < y + t.shape[0]]

        best: Optional[Detection] = None
        for x_off, y_off, tile in tiles:
            dets = self.detector.predict_batch([tile], conf_thresh=self.conf_thresh)[0]
            balls = [d for d in dets if d.label == "ball"]
            for b in balls:
                global_det = Detection(
                    x1=b.x1 + x_off, y1=b.y1 + y_off,
                    x2=b.x2 + x_off, y2=b.y2 + y_off,
                    label="ball", score=b.score,
                )
                if best is None or global_det.score > best.score:
                    best = global_det
        return best
```

### 15.2 Trajectory filter (rejects false positives)

`score_miner_core/ball/trajectory_filter.py`:

```python
"""Reject ball detections that violate physics.
The ball moves smoothly between frames; large position jumps are usually false positives."""

from __future__ import annotations
from typing import Optional, Tuple

def predict_next(prev_x, prev_y, prev_vx, prev_vy, dt: float = 1.0) -> Tuple[float, float]:
    return prev_x + prev_vx * dt, prev_y + prev_vy * dt

def is_plausible(prev_x, prev_y, prev_vx, prev_vy,
                 cand_x, cand_y,
                 max_jump_px: float = 100.0,
                 dt: float = 1.0) -> bool:
    """Plausible if candidate is within max_jump_px of predicted position."""
    pred_x, pred_y = predict_next(prev_x, prev_y, prev_vx, prev_vy, dt)
    dist = ((cand_x - pred_x) ** 2 + (cand_y - pred_y) ** 2) ** 0.5
    return dist <= max_jump_px
```

### 15.3 Gate to Phase 13

- Ball recall improves on annotated SoccerNet val
- Ball false positives stay within 10% of true positives
- Latency overhead < 30ms per refiner call

---

## 16. Phase 13 — Homography Upgrade (Days 26-28)

**Goal:** Dominate the `keypoints` pillar. Constructor.tech won SoccerNet GSR 2024 by GS-HOTA 63.81 vs runner-up 43.15 mainly because of camera calibration quality.

### 16.1 What the validator actually scores

Re-read `keypoints.py` from your Phase 0 notes. The validator:
1. Takes your 32 keypoints as projection of the football pitch template
2. Computes a homography from your points to pitch coords
3. Projects pitch lines back into the frame
4. Compares projected lines to real field-line edges (edge detection on the frame)
5. Scores overlap

This means **bad keypoints produce bad homography produce bad projected lines produce low score**. Emit `(0,0)` for any keypoint you're not sure about — that's better than a wrong keypoint.

### 16.2 Multi-stage pipeline

```
Frame
  → SegFormer pitch line segmentation (every K frames)
  → Keypoint regression head from segmentation features
  → Homography fit via RANSAC on confident points
  → Reprojection validation (drop H if reprojection error too large)
  → Temporal smoothing: if confidence < threshold, hold previous H
  → AuxFlow propagation: between SegFormer frames, propagate keypoints with optical flow
```

### 16.3 `score_miner_core/keypoints/homography_filter.py`

```python
"""RANSAC homography fit + reprojection validation."""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import cv2

# Standard FIFA pitch dimensions, in meters
PITCH_W = 105.0
PITCH_H = 68.0

def fit_homography(image_points: List[Tuple[float, float]],
                   pitch_points: List[Tuple[float, float]],
                   ransac_thresh: float = 5.0
                   ) -> Optional[Tuple[np.ndarray, float]]:
    """Returns (H, confidence) or None."""
    if len(image_points) < 4:
        return None
    src = np.array(image_points, dtype=np.float32)
    dst = np.array(pitch_points, dtype=np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if H is None:
        return None
    inlier_ratio = float(mask.sum()) / len(image_points)
    return H, inlier_ratio

def is_valid_projection(H: np.ndarray, img_w: int, img_h: int) -> bool:
    """Reject bowtie / extreme-scale homographies."""
    # project image corners to pitch
    corners = np.array([
        [0, 0, 1], [img_w, 0, 1], [img_w, img_h, 1], [0, img_h, 1],
    ], dtype=np.float32).T
    proj = (H @ corners).T
    proj /= proj[:, 2:3]
    # check convexity (no bowtie)
    pts = proj[:, :2]
    edges = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    crosses = np.cross(edges, np.roll(edges, -1, axis=0))
    if not (np.all(crosses > 0) or np.all(crosses < 0)):
        return False
    # check scale plausibility (projected area shouldn't be absurd)
    area = 0.5 * abs(np.cross(pts[2] - pts[0], pts[3] - pts[1]))
    if area < 100 or area > PITCH_W * PITCH_H * 10:
        return False
    return True
```

### 16.4 AuxFlow-style propagation

Between full keypoint-model passes, use optical flow to move existing keypoints:

```python
def propagate_keypoints_flow(prev_frame_gray, curr_frame_gray,
                              prev_keypoints, max_err: float = 5.0):
    """Lucas-Kanade tracking of keypoints between frames."""
    pts = np.array(prev_keypoints, dtype=np.float32).reshape(-1, 1, 2)
    new_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_frame_gray, curr_frame_gray, pts, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    out = []
    for i, (p, st, e) in enumerate(zip(new_pts, status, err)):
        if st[0] == 1 and e[0] < max_err:
            out.append((float(p[0][0]), float(p[0][1])))
        else:
            out.append((0.0, 0.0))   # mark missing
    return out
```

### 16.5 The 32-keypoint contract

TurboVision expects exactly 32 keypoints in a fixed order matching the manifest. Read `notes/scoring_spec.md` for the exact list (usually corners, penalty box corners, goal box corners, center circle points, etc.). Always emit 32. Emit `(0, 0)` for missing.

### 16.6 Gate to Phase 14

- Keypoints pillar in top-5 on val set
- Homography validity rate > 80% on stable wide shots
- Falls back gracefully on close-ups (emit (0,0) instead of garbage)

---

## 17. Phase 14 — Score-Aware Calibration (Days 29-30)

**Goal:** Stop optimizing recall. Start optimizing the actual scoring function.

### 17.1 Per-class, per-scene confidence calibration

Build a calibration table from replay data:

```python
# score_miner_core/runtime/confidence_calibrator.py

class ConfidenceCalibrator:
    """Per-class, per-scene-type threshold table learned from replay."""

    def __init__(self):
        # (class, scene_type) -> conf_threshold
        self.thresholds = {
            ("player", "wide"): 0.45,
            ("player", "closeup"): 0.55,
            ("ball", "wide"): 0.35,
            ("ball", "closeup"): 0.50,
            ("referee", "wide"): 0.55,
            ("referee", "closeup"): 0.65,
            ("goalkeeper", "wide"): 0.50,
            ("goalkeeper", "closeup"): 0.60,
        }

    def threshold(self, cls: str, scene_type: str) -> float:
        return self.thresholds.get((cls, scene_type), 0.5)

    def update_from_replay(self, replay_cases) -> None:
        """For each (class, scene_type), find the threshold that maximizes
        the pillar most affected by this class."""
        # ... offline analysis from replay store ...
```

### 17.2 Uncertainty-driven action routing

```python
def output_action(conf: float, scene_type: str, cls: str,
                  uncertainty_bucket: str) -> str:
    """Returns one of: 'emit', 'suppress', 'track_only', 'refine'."""
    if uncertainty_bucket == "very_high":
        return "suppress"   # don't emit a guess
    if uncertainty_bucket == "high" and cls == "ball":
        return "refine"     # run BallCropRefiner
    if uncertainty_bucket == "high":
        return "track_only" # rely on tracker
    return "emit"
```

### 17.3 Gate to Phase 15

- Total weighted score +10% with no other changes
- Calibration table is data-driven, not hand-tuned
- Replay shows fewer false positives in close-ups

---

## 18. Phase 15 — ReID + Jersey OCR (Days 31-33)

**Goal:** Role pillar to top-5. Palette stable across occlusions and shadows. This is the move that separates top-10 from top-3.

### 18.1 CLIP-ReIdent integration

CLIP-ReIdent fine-tuned on SoccerNet ReID achieves 98.44% mAP — better than OSNet for sports re-id.

```python
# score_miner_core/tracking/reid_embeddings.py

import torch
import torch.nn.functional as F
from PIL import Image
from typing import List
import numpy as np

class CLIPReIDEmbedder:
    """Wraps a fine-tuned CLIP ViT-L/14 for player re-identification.

    Use selectively — only on stable tracklets, not every frame.
    Latency: ~30ms per batch of 8 crops on A10."""

    def __init__(self, weights_path: str, device: str = "cuda"):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=weights_path
        )
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def embed(self, crops: List[np.ndarray]) -> np.ndarray:
        """crops: list of BGR ndarrays. Returns (N, 768) L2-normalized embeddings."""
        pil_crops = [Image.fromarray(c[..., ::-1]) for c in crops]
        tensors = torch.stack([self.preprocess(p) for p in pil_crops]).to(self.device)
        feats = self.model.encode_image(tensors)
        feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy()
```

### 18.2 Selective use — only on uncertainty

Don't embed every frame. Embed only when:
- A new track is created (initial fingerprint)
- A track has been lost > N frames and we're re-acquiring
- Team assignment confidence < threshold
- A potential team flip is voted

This keeps latency manageable.

### 18.3 Jersey number recognition (Koshkina pipeline)

```python
# score_miner_core/team/jersey_number_evidence.py

"""PARSeq OCR on torso back crops.
Use only when ReID confidence is high and crop is clear."""

import numpy as np

class JerseyNumberRecognizer:
    def __init__(self, weights_path: str, device: str = "cuda"):
        # Koshkina pipeline: pretrained PARSeq + light fine-tune on SoccerNet jersey set
        from parseq import PARSeq
        self.model = PARSeq.load(weights_path).eval().to(device)
        self.device = device

    def recognize(self, torso_crop_bgr: np.ndarray) -> Optional[int]:
        """Returns jersey number 0-99 or None if not confident."""
        # Preprocess, run, extract digits, validate range
        ...
```

### 18.4 Vote aggregation in VideoState

Each track accumulates jersey number votes across frames. Commit when one number has > 5 votes more than runner-up. Use jersey number to:
- Disambiguate team if same color matches two players
- Resolve role: number 1 → likely goalkeeper
- Stabilize track ID through occlusion

### 18.5 Gate to Phase 16

- Role pillar in top-5
- Palette pillar > 0.75 on val set
- ReID + jersey OCR overhead < 50ms per batch of 64


---

## 19. Phase 16 — Optuna v2 with Hard Splits (Days 34-35)

**Goal:** Re-run Optuna with replay-mined hard splits. Force the optimizer to prove gains on hard cases, not easy averages.

### 19.1 Split definition

- **Easy split:** stable wide shots, no scene cuts, well-lit
- **Hard split:** scene cuts, close-ups, replays, shadows, crowded penalty boxes
- **Mixed split:** held-out match clips

### 19.2 Multi-objective Optuna

Use NSGA-II for multi-objective search:

```python
study = optuna.create_study(
    directions=["maximize", "maximize", "minimize"],
    sampler=optuna.samplers.NSGAIISampler(seed=42),
)

def multi_objective(trial):
    cfg = suggest_config(trial)
    miner = miner_factory(cfg)
    easy_score = eval_on_split(miner, easy_split)
    hard_score = eval_on_split(miner, hard_split)
    p95 = measure_p95(miner, mixed_split)
    return easy_score, hard_score, p95
```

This finds Pareto-optimal configs. Pick from the Pareto front based on:
- Top score on hard split
- Within 5% of best on easy split
- p95 < target

### 19.3 Gate to Phase 17

- Pareto front identified
- Selected config scored on held-out match split
- Top-10 leaderboard position on real validator (not just local sim)

---

## 20. Phase 17 — Replay-Driven Fine-Tuning (Days 36-38)

**Goal:** Fine-tune the winning detector on the actual failure cases your replay store identified. NOT generic SoccerNet retraining.

### 20.1 Why this comes so late

If you fine-tune before knowing what fails, you train against the wrong distribution. After Phase 16, you have:
- Replay cases tagged by failure mode (ball_miss, role_confusion, etc.)
- Heatmap of which scene types lose score
- Hard-negative examples mined by Optuna trials

NOW fine-tuning is targeted.

### 20.2 Build the training split

```python
# score_miner_core/training/hard_negative_mining/build_split.py

def build_hard_training_set(replay_store, out_dir):
    """For each failure tag, gather N examples with image + GT annotations.
    Output: COCO-format JSON for RF-DETR fine-tuning."""

    failure_cases = {}
    for tag in ["ball_miss", "role_confusion", "team_flip", "smoothness_drop"]:
        failure_cases[tag] = replay_store.find_by_tag(tag)

    # Oversample hard cases 5x, easy cases 1x
    coco_json = {"images": [], "annotations": [], "categories": []}
    for tag, cases in failure_cases.items():
        for case in cases[:200]:  # cap per tag
            for repeat in range(5):
                _add_case_to_coco(coco_json, case)
    return coco_json
```

### 20.3 RF-DETR fine-tune

Verified against rfdetr 1.4.1 docs:

```python
from rfdetr import RFDETRLarge

model = RFDETRLarge(pretrain_weights="./models/rfdetr_l_base.pth")
model.train(
    dataset_dir="./datasets/curated/hard_negatives_coco",
    output_dir="./output/rfdetr_l_finetuned_v1",
    epochs=15,
    batch_size=8,           # for A100 80GB
    lr=1e-4,
    weight_decay=1e-4,
    resolution=560,
    use_ema=True,
    grad_clip=1.0,
)
```

Rent an A100 on Vast.ai for 4 hours (~$5). Fine-tune. Download checkpoint. Plug into `score_miner/models/rfdetr_l_finetuned.pth`.

### 20.4 Validation

After fine-tune:
- Re-run validator_sim on val_split
- Compare each pillar before/after
- If iou pillar +5% but smoothness -3%, the fine-tune may have made the detector more flickery — back off

### 20.5 Gate to Phase 18

- Iou pillar in top-5
- No pillar regressed by more than 2%
- Fine-tuned model fits in memory budget

---

## 21. Phase 18 — TensorRT Export (Days 39-40)

**Goal:** p95 < 150ms on batch of 64 frames. Free score boost via latency penalty curve.

### 21.1 Export RF-DETR to ONNX → TensorRT

Verified rfdetr 1.4.1 export API:

```python
from rfdetr import RFDETRLarge

# 1. Load fine-tuned checkpoint
model = RFDETRLarge(pretrain_weights="./output/rfdetr_l_finetuned_v1/checkpoint.pth")

# 2. Export to ONNX
model.export(
    output_dir="./export/onnx/rfdetr_l_v1",
    dynamic_batch=True,           # supports variable batch at runtime
    opset=17,                      # 17 avoids the FP16 degradation noted in RF-DETR paper
)

# 3. Compile TensorRT engine on TARGET GPU (must match Chute's GPU family)
from argparse import Namespace
from rfdetr.export.tensorrt import trtexec

args = Namespace(verbose=True, profile=False, dry_run=False)
trtexec("./export/onnx/rfdetr_l_v1/inference_model.onnx", args)
# Produces inference_model.engine
```

### 21.2 Critical FP16 caveat

The RF-DETR ICLR 2026 paper notes that **naive FP16 quantization can collapse D-FINE to 0.5 AP** unless you fix opset to 17. RF-DETR's own export code handles this correctly. Always verify with a calibration run after export:

```python
# Re-score on val set with the .engine — should be within 0.5% of PyTorch
```

### 21.3 TensorRT runner inside miner

```python
# score_miner_core/detector/rfdetr_tensorrt_runner.py

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

class RFDETRTRTRunner:
    """Drop-in replacement for RFDETRRunner using a compiled .engine."""

    def __init__(self, engine_path: str, class_names: list):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self._class_names = class_names
        # allocate I/O buffers...

    def predict_batch(self, images, conf_thresh=0.35):
        # preprocess to NCHW float32
        # H2D copy, execute, D2H copy
        # postprocess to Detection list
        ...
```

### 21.4 Gate to Phase 19

- p95 < 150ms for batch of 64 frames
- Total weighted score within 0.5% of PyTorch baseline
- Memory budget maintained

---

## 22. Phase 19 — SAM3 Offline Pseudo-Labeling (Days 41-42)

**Goal:** Use the same SAM3 model that the validator uses for pseudo-GT to **generate matched training labels** for your detector.

### 22.1 Why this is the move

The validator scores against SAM3 pseudo-GT when `ground_truth=false`. If you train your detector to **match SAM3's bounding-box behavior**, you score higher even when SAM3 is technically wrong about a pixel. You're optimizing the actual scoring function, not the abstract "correct" answer.

### 22.2 SAM3 pseudo-label pipeline

```python
# score_miner_core/training/pseudo_labeling/sam3_labels.py

"""Generate pseudo-GT using SAM3 (concept-prompted, DETR-based segmentation).
This matches what the validator uses when no human GT exists for an Element."""

from sam3 import SAM3Model    # install from facebookresearch/sam3

def label_video_with_sam3(video_path, output_dir, concepts):
    sam = SAM3Model.from_pretrained("facebook/sam3-base")
    sam.set_concepts(concepts)   # ["soccer player", "soccer ball", "referee"]

    for frame_idx, frame in enumerate(load_frames(video_path)):
        masks, scores, labels = sam.predict(frame)
        # convert masks to bounding boxes
        boxes = [mask_to_bbox(m) for m in masks]
        save_coco_annotation(output_dir, frame_idx, boxes, labels, scores)
```

### 22.3 Use these labels alongside SoccerNet GT

- SoccerNet GT for clips with human annotation (best signal)
- SAM3 pseudo-GT for unlabeled clips you mine from broadcasts (volume signal)
- Train detector on union, with class-weighted loss

### 22.4 Qwen3-VL / Molmo2 for hard-crop arbitration

When SAM3 and your detector disagree:
- Crop the disputed region
- Ask Qwen3-VL: "Is this a soccer player, referee, goalkeeper, or other?"
- Use VLM answer as tie-break label

This is slow (a VLM call takes ~500ms), so only run on the ~5% of disputed crops. Build a training set from VLM-arbitrated cases.

### 22.5 Gate to Phase 20

- 5,000+ additional pseudo-labeled frames mined
- Re-fine-tuned detector beats Phase 17 model on val set
- VLM-arbitrated hard cases improve role pillar specifically

---

## 23. Phase 20 — Final Optuna Sweep + Mainnet Commit (Days 43-45)

**Goal:** Last optimization pass. Then commit on-chain and start earning.

### 23.1 Pre-commit checklist

Run through this list. Every line must be true before `sv deploy-os-miner` (no --no-commit):

```
[ ] validator_sim score on heldout match split > 0.7 weighted total
[ ] p95 latency < 200ms on Chutes A10 with batch 64
[ ] Memory under 4.5 GB after sustained load
[ ] No schema failures in 100 consecutive predict_batch calls
[ ] Class mapping correct (matches manifest exactly)
[ ] Frame count matches meta.min_frames_required in 100 cases
[ ] Determinism: same input → same output across 5 runs
[ ] /health returns 200 within 3 seconds
[ ] /predict completes within latency_p95_ms target
[ ] Output JSON validates against TurboVision schema
[ ] Replay store has 500+ cases with no critical regression patterns
[ ] Optuna best config out-performs every earlier config on heldout
[ ] TensorRT engine score matches PyTorch score within 0.5%
```

### 23.2 Final commit

```bash
cd turbovision
sv -v deploy-os-miner --model-path ../score_miner --element-id <ELEMENT_ID>
# (no --no-commit this time)
# Confirms the on-chain commitment of your hotkey to this Chute slug
```

### 23.3 Post-deploy monitoring (week 1)

For the first 7 days, every 6 hours:

```bash
sv -vv stats --hotkey <YOUR_HOTKEY>
# Check: rank, incentive, latest score, dropped requests

btcli s metagraph --netuid 44 | grep <YOUR_HOTKEY>
# Check: trust, incentive, emission

# Pull recent replay artifacts from Chutes logs
# Identify regressions, file in replay store, queue for next Optuna run
```

### 23.4 The continuous loop (forever after)

Top miners don't stop. They run:

```
weekly:
  - Optuna sweep (50 trials on fresh replay)
  - Fine-tune detector if hard-case distribution shifted
  - Calibration table update

monthly:
  - Re-benchmark detector candidates (new model releases happen)
  - Profile latency, look for new optimizations

quarterly:
  - Major architecture review
  - Element expansion if Score adds new Elements
```

### 23.5 Gate to "done"

You're top-3. You earn meaningfully. The loop runs without you.

---

## 24. The Three Things That Beat 90% of Miners

If you only do three things, do these:

1. **`validator_sim`** — Most miners optimize against COCO mAP or vendor benchmarks. You optimize against TurboVision's exact scorer. This alone beats everyone tuning blindly.

2. **`VideoState`** — Most miners are stateless. They lose smoothness, palette, role pillars by default. Your persistent state wins all three.

3. **`replay → Optuna → fine-tune`** — Most miners ship once and pray. You iterate weekly with evidence.

The detector matters. The model architecture matters. But these three systems are what compound.

---

## 25. What NEVER To Do

- Don't pick a detector by COCO AP. Pick by `validator_sim` score on val_split.
- Don't run SAM3 or Qwen3-VL live in the miner. p95 dies.
- Don't patch the Chute template. Integrity check will reject the deploy.
- Don't commit on-chain before dry-run is clean for 100 consecutive predicts.
- Don't let Optuna see held-out match clips. Overfit = leaderboard collapse.
- Don't fine-tune before replay tells you what fails.
- Don't run both MLflow AND W&B. Pick one. Time wasted on dual setup compounds.
- Don't install full TrackLab into the Chute. Memory budget dies. Copy specific modules.
- Don't return < `meta.min_frames_required` frames. Score zero.
- Don't return wrong cls_id. Score zero.
- Don't emit `(NaN, NaN)` for missing keypoints. Emit `(0, 0)`.
- Don't smooth across scene cuts. Reset state.
- Don't trust this doc blindly. Verify each tool version, each API call, each constant against the current upstream repo. 2026 moves fast.

---

## 26. References (verified May 2026)

### TurboVision
- Repo: https://github.com/score-technologies/turbovision
- Score subnet docs: https://scoredata.me

### Detectors
- RF-DETR (ICLR 2026, v1.4.1 March 2026): https://github.com/roboflow/rf-detr
- RF-DETR docs: https://rfdetr.roboflow.com/
- DEIMv2 (DINOv3 backbone): https://github.com/Intellindust-AI-Lab/DEIMv2
- D-FINE: https://github.com/Peterande/D-FINE
- SoccerDETR (MDPI 2026): https://www.mdpi.com/2227-7080/14/3/142

### SoccerNet GSR baseline
- sn-gamestate (updated May 2, 2026): https://github.com/SoccerNet/sn-gamestate
- TrackLab framework: https://github.com/TrackingLaboratory/tracklab
- TrackLab docs: https://trackinglaboratory.github.io/tracklab/
- Constructor.tech (2024 winner, GS-HOTA 63.81): https://arxiv.org/abs/2504.06357
- Broadcast2Pitch (WACV 2026): https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf
- AuxFlow (2026): https://www.sciencedirect.com/science/article/pii/S1077314226000299

### ReID & Jersey
- CLIP-ReIdent (98.44% mAP): https://arxiv.org/pdf/2303.11855
- Koshkina Jersey Number Pipeline: https://github.com/mkoshkina/jersey-number-pipeline
- PARSeq: https://github.com/baudm/parseq

### Pseudo-labeling
- SAM3 (Nov 2025): https://github.com/facebookresearch/sam3
- Qwen3-VL: https://huggingface.co/Qwen/Qwen3-VL
- Molmo2: https://huggingface.co/allenai/Molmo2-7B
- C-RADIOv4-H: https://huggingface.co/nvidia/C-RADIOv4-H

### Tools
- Optuna: https://optuna.org/
- MLflow: https://www.mlflow.org/
- FiftyOne: https://docs.voxel51.com/
- TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/index.html

### Datasets
- SoccerNet GSR: https://www.soccer-net.org/tasks/game-state-reconstruction
- SoccerNet Tracking: https://www.soccer-net.org/tasks/tracking
- SoccerTrack v2: https://atomscott.github.io/SoccerTrack-v2/
- Roboflow Sports Universe: https://universe.roboflow.com/roboflow-jvuqo/soccer-players-5fuqs

### Infrastructure
- Chutes platform: https://chutes.ai
- Bittensor docs: https://docs.learnbittensor.org/
- Vast.ai GPU rentals: https://vast.ai

---

## 27. Build Order Cheat Sheet (single page)

```
Day 1:   Phase 0 — read manifest + scoring source. Fill notes/scoring_spec.md
Day 2-3: Phase 1 — repo skeleton + class_mapping + memory_budget. Build wheel.
Day 4:   Phase 2 — benchmark/run_local.py. Stub miner passes schema.
Day 5-6: Phase 3 — RF-DETR-M smoke deploy. /health 200, /predict valid.
Day 7-8: Phase 4 — validator_sim wrapping turbovision scoring. Sim matches real <5%.
Day 9-11:Phase 5 — RF-DETR-L vs DEIMv2-L vs D-FINE-L. Winner chosen by sim.
Day 12:  Phase 6 — failure_store. Every run archived with tags.
Day 13-14:Phase 7 — Optuna 200 trials on val_split. >5% over baseline.
Day 15-16:Phase 8 — sn-gamestate / TrackLab read-through. Module shortlist.
Day 17-18:Phase 9 — team palette + role hysteresis. Palette ≥ 0.6.
Day 19-21:Phase 10 — VideoState + scene analyzer. Smoothness ≥ 0.7.
Day 22-23:Phase 11 — adaptive scheduler. p95 -30%, score flat.
Day 24-25:Phase 12 — ball specialist. Ball recall up, FP contained.
Day 26-28:Phase 13 — SegFormer + AuxFlow homography. Keypoints top-5.
Day 29-30:Phase 14 — score-aware calibration. +10% total.
Day 31-33:Phase 15 — CLIP-ReIdent + jersey OCR. Role top-5.
Day 34-35:Phase 16 — Optuna v2 multi-objective with hard splits. Top-10.
Day 36-38:Phase 17 — RF-DETR-L fine-tune on replay-mined hard cases. Iou top-5.
Day 39-40:Phase 18 — TensorRT FP16 export. p95 < 150ms.
Day 41-42:Phase 19 — SAM3 + VLM pseudo-labels. Detector v2.
Day 43-45:Phase 20 — final Optuna + mainnet commit. Top-3.
```

**Total: 45 days solo with AI-assisted coding. Top-3 by day 45. Top-1 contested with teams running 24/7.**

This is the document. Open it every morning. Update it whenever a phase deviates. The doc is the build.