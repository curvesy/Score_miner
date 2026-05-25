# Implementation Spec — Week 1 (May 21 – May 27 2026)

> Audience: a coding agent that will write the diffs.
> Scope: **only** the silent-bug fixes + real deploy + SAM3 distillation scaffold. Everything else is deferred.
> Read `game_changer_playbook.md` first for strategic context. This doc is the implementation contract.

---

## 0. Preflight Checklist

Run these in order. Stop if any fails — fix before proceeding.

```bash
cd /home/sina/projects/validator_improve/score_miner_project

# 0.1 Verify Python and venv
.venv/bin/python --version            # expect: 3.13.x (turbovision pins cp313)
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 0.2 Verify package installs cleanly
.venv/bin/python -c "import score_miner_core; print(score_miner_core.__file__)"

# 0.3 Verify tests pass on a clean tree
PYTHONPATH=score_miner_dev/src MPLCONFIGDIR=/tmp/mpl \
  .venv/bin/python -m pytest score_miner_dev/tests -q
# expect: all passing (~24+ tests)

# 0.4 Verify TurboVision repo present and CLI works
cd ../turbovision && uv run --python python3.13 sv --help && cd ../score_miner_project

# 0.5 Snapshot current state before changes
git -C score_miner_project add -A 2>/dev/null
git -C score_miner_project diff --stat HEAD || true
git -C score_miner_project tag preweek1-$(date +%s) 2>/dev/null || true
```

If 0.3 fails, **STOP**. Fix tests first before touching any production code.

---

## 1. Library Versions (May 2026 Current)

Pin these in `score_miner_dev/pyproject.toml` dependencies. **Verified May 21 2026.**

```toml
[project]
dependencies = [
  # Core
  "torch>=2.5,<2.7",                    # 2.5+ for stable torch.compile
  "torchvision>=0.20",
  "numpy>=2.0",
  "opencv-python>=4.10",                # 4.10+ for stable cv2.cuda
  "pydantic>=2.10",
  "psutil>=6.0",

  # Detection
  "rfdetr==1.4.1",                      # PyPI latest; has optimize_for_inference + ONNX export
  "supervision>=0.25",                  # has improved ByteTrack + Detections API
  "transformers>=4.50",                 # has native RF-DETR + SAM3 image processors

  # SAM3 (Week 1 scaffolding only — full pipeline is Week 2+)
  # Use facebookresearch/sam3 directly (HF gated)
  # See Task 7 for install instructions
]

[project.optional-dependencies]
distillation = [
  "ultralytics>=8.3",                   # YOLO26 (yolo26n.pt) via .pt files
  "huggingface_hub>=0.26",
  "datasets>=3.0",
  "accelerate>=1.0",
]
export = [
  "onnx>=1.17",
  "onnxruntime-gpu>=1.20",
  "onnxsim>=0.4.36",
  "tensorrt>=10.0",                     # optional, prefer trtexec from system if available
]
dev = [
  "pytest>=8.0",
  "pytest-cov>=5.0",
  "ruff>=0.7",
]
```

After updating, run:
```bash
cd /home/sina/projects/validator_improve/score_miner_project
.venv/bin/python -m pip install -U pip
uv pip install -e "score_miner_dev[dev,distillation,export]"
```

---

## Task 1: BGR→RGB Color Channel Fix 🔴 CRITICAL

**Why this is critical**: RF-DETR's `predict()` (rfdetr/detr.py:324) explicitly requires RGB. The chute template decodes frames via `cv2.imdecode(arr, IMREAD_COLOR)` and `cv2.VideoCapture`, both returning BGR. Your `MinerRuntime.predict_batch` passes BGR straight to detector AND to team_color. **Silent AP loss + wrong team color clustering.**

### File
`score_miner_project/score_miner_dev/src/score_miner_core/runtime/miner_runtime.py`

### Exact diff

Find this block (around lines 70-94):

```python
def predict_batch(
    self,
    batch_images: list[np.ndarray],
    offset: int,
    n_keypoints: int,
) -> list[TVFrameResult]:
    self.memory_budget.assert_within_limit()
    detections_by_frame = self._predict_boxes(batch_images)
    empty_keypoints = [(0, 0) for _ in range(n_keypoints)]
    results: list[TVFrameResult] = []
    for frame_idx, image in enumerate(batch_images):
        boxes = filter_boxes_by_config(
            _detections_to_boxes(detections_by_frame[frame_idx]),
            self.postprocess_config,
        )
        boxes = assign_team_ids_by_color(image, boxes, self.team_color_config)
        boxes = self.team_color_memory.stabilize(boxes)
        results.append(
            TVFrameResult(
                frame_id=offset + frame_idx,
                boxes=boxes,
                keypoints=empty_keypoints,
            )
        )
    return results
```

Replace with:

```python
def predict_batch(
    self,
    batch_images: list[np.ndarray],
    offset: int,
    n_keypoints: int,
) -> list[TVFrameResult]:
    self.memory_budget.assert_within_limit()
    # Chute template decodes via cv2 (BGR). RF-DETR + cv2.COLOR_RGB2{HSV,LAB} expect RGB.
    rgb_batch = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in batch_images]
    detections_by_frame = self._predict_boxes(rgb_batch)
    empty_keypoints = [(0, 0) for _ in range(n_keypoints)]
    results: list[TVFrameResult] = []
    for frame_idx, image_rgb in enumerate(rgb_batch):
        boxes = filter_boxes_by_config(
            _detections_to_boxes(detections_by_frame[frame_idx]),
            self.postprocess_config,
        )
        boxes = assign_team_ids_by_color(image_rgb, boxes, self.team_color_config)
        boxes = self.team_color_memory.stabilize(boxes)
        results.append(
            TVFrameResult(
                frame_id=offset + frame_idx,
                boxes=boxes,
                keypoints=empty_keypoints,
            )
        )
    return results
```

Add at the top of the file (with other imports):

```python
import cv2  # add if not already imported
```

### Test
```bash
cd /home/sina/projects/validator_improve/score_miner_project
PYTHONPATH=score_miner_dev/src MPLCONFIGDIR=/tmp/mpl NO_ALBUMENTATIONS_UPDATE=1 \
  .venv/bin/python -m score_miner_core.benchmark.run_local \
    --video ../turbovision/tests/test_data/videos/example_football.mp4 \
    --frames 32 --batch-size 1 --n-keypoints 32 \
    --detector rfdetr_m --threshold 0.35 --player-cls-id 0
```

### Acceptance criteria
- Schema valid: true
- `boxes_total` ≥ prior baseline (≥ 545 from till_now.md)
- `boxes_per_frame_mean` in [10, 20]
- No exceptions

### Quick A/B comparison test (new file)

Create `score_miner_dev/tests/test_bgr_rgb_fix.py`:

```python
"""Sanity test: detector should not crash when fed RGB frames."""
import numpy as np
import cv2
import pytest

from score_miner_core.runtime.miner_runtime import MinerRuntime
from score_miner_core.runtime.postprocess import PostprocessConfig
from score_miner_core.runtime.team_color import TeamColorConfig
from score_miner_core.runtime.tracking import TrackingConfig


def _green_field_with_red_player(h=400, w=600) -> np.ndarray:
    """BGR image with a green field and a red 30x80 'player' box in the center."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (40, 120, 40)  # BGR green
    cx, cy = w // 2, h // 2
    img[cy-40:cy+40, cx-15:cx+15] = (40, 40, 200)  # BGR red player
    return img


def test_predict_batch_handles_bgr_input(tmp_path):
    runtime = MinerRuntime(
        path_hf_repo=tmp_path,
        detector=None,  # stub detector path; just exercise the cvtColor logic
        postprocess_config=PostprocessConfig(),
        team_color_config=TeamColorConfig(),
        tracking_config=TrackingConfig(enabled=False),
    )
    bgr_image = _green_field_with_red_player()
    out = runtime.predict_batch([bgr_image], offset=0, n_keypoints=32)
    assert len(out) == 1
    assert out[0].frame_id == 0
    assert len(out[0].keypoints) == 32
```

Run:
```bash
PYTHONPATH=score_miner_dev/src .venv/bin/python -m pytest score_miner_dev/tests/test_bgr_rgb_fix.py -v
```

### Rollback
```bash
git -C score_miner_project checkout score_miner_dev/src/score_miner_core/runtime/miner_runtime.py
```

---

## Task 2: Wire `ball_cls_id` 🔴 CRITICAL

**Why this is critical**: The default in `score_miner/miner.py:17` is `ball_cls_id = None`, which means `class_id_mapper.coco_to_turbovision` never registers COCO sports-ball (id 37) → TurboVision ball class. RF-DETR detects balls; you drop every one. **BallDetect element = guaranteed zero score until fixed.**

### File 1: `score_miner_project/score_miner/miner.py`

Replace the entire file with:

```python
from pathlib import Path
from os import getenv

from score_miner_core.detector.detector_router import create_detector
from score_miner_core.runtime.miner_runtime import MinerRuntime
from score_miner_core.runtime.postprocess import PostprocessConfig
from score_miner_core.runtime.team_color import TeamColorConfig
from score_miner_core.runtime.tracking import TrackingConfig


def _env_bool(name: str, default: bool) -> bool:
    raw = getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_or_none(name: str) -> int | None:
    raw = getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


class Miner:
    def __init__(self, path_hf_repo: Path) -> None:
        # Element-scope variables (no architecture refactor — just configurable hooks)
        element_id = getenv("SCORE_MINER_ELEMENT_ID", "PlayerDetect_v1@1.0")
        detector_name = getenv("SCORE_MINER_DETECTOR", "rfdetr_m")
        threshold = float(getenv("SCORE_MINER_THRESHOLD", "0.75"))
        player_cls_id = int(getenv("SCORE_MINER_PLAYER_CLS_ID", "0"))
        ball_cls_id = _env_int_or_none("SCORE_MINER_BALL_CLS_ID")

        # Per-element module toggles (Week 1: simple flags; full profile dispatch deferred)
        use_team_color = _env_bool("SCORE_MINER_USE_TEAM_COLOR", default=True)
        use_tracker = _env_bool("SCORE_MINER_USE_TRACKER", default=True)

        postprocess_config = PostprocessConfig.from_env()
        team_color_config = TeamColorConfig.from_env()
        team_color_config = team_color_config.model_copy(update={"enabled": use_team_color})
        tracking_config = TrackingConfig.from_env()
        tracking_config = tracking_config.model_copy(update={"enabled": use_tracker})

        detector = create_detector(
            detector_name,
            threshold=threshold,
            player_cls_id=player_cls_id,
            ball_cls_id=ball_cls_id,
        )
        self.element_id = element_id
        self.runtime = MinerRuntime(
            path_hf_repo,
            detector=detector,
            postprocess_config=postprocess_config,
            team_color_config=team_color_config,
            tracking_config=tracking_config,
        )

    def __repr__(self) -> str:
        return f"Miner(element_id={self.element_id!r}, runtime={self.runtime!r})"

    def predict_batch(self, batch_images, offset: int, n_keypoints: int):
        return self.runtime.predict_batch(batch_images, offset, n_keypoints)
```

### File 2: `score_miner_project/score_miner/chute_config.yml`

Add the new env vars to the existing `env:` block (or create one):

```yaml
# Existing config above...

env:
  # Element identity
  SCORE_MINER_ELEMENT_ID: "PlayerDetect_v1@1.0"

  # Detector
  SCORE_MINER_DETECTOR: "rfdetr_m"
  SCORE_MINER_THRESHOLD: "0.75"
  SCORE_MINER_PLAYER_CLS_ID: "0"     # element.objects[0] = "player" — VERIFY against live manifest
  SCORE_MINER_BALL_CLS_ID: "1"        # element.objects[1] = "ball" — VERIFY against live manifest

  # Postprocess
  SCORE_MINER_MAX_BOXES_PER_FRAME: "18"
  SCORE_MINER_MIN_BOX_AREA: "0"

  # Module toggles (week-1 flexibility hooks)
  SCORE_MINER_USE_TEAM_COLOR: "true"
  SCORE_MINER_USE_TRACKER: "true"

  # Team color clustering
  SCORE_MINER_TEAM_COLOR_ENABLED: "true"
  SCORE_MINER_TEAM_MIN_PLAYERS: "4"
  SCORE_MINER_TEAM_TRACK_MEMORY_ENABLED: "true"
  SCORE_MINER_TEAM_TRACK_MEMORY_MIN_VOTES: "2"

  # Tracking
  SCORE_MINER_TRACKING_ENABLED: "true"
  SCORE_MINER_TRACK_FRAME_RATE: "25.0"
```

⚠️ **Verification step before deploying**: run Task 4 (manifest validation) to confirm `SCORE_MINER_PLAYER_CLS_ID` and `SCORE_MINER_BALL_CLS_ID` match the **live** manifest's `element.objects` order.

### Test
```bash
PYTHONPATH=score_miner_dev/src .venv/bin/python -c "
from pathlib import Path
import os
os.environ['SCORE_MINER_BALL_CLS_ID'] = '1'
from score_miner.miner import Miner
m = Miner(Path('/tmp'))
print(m)
print('class_id_map:', m.runtime.detector.class_id_mapper.class_id_map if m.runtime.detector else None)
"
```

### Acceptance criteria
- Output shows `class_id_map: {1: 0, 37: 1}` (COCO person→player_cls_id=0, COCO sports_ball→ball_cls_id=1)
- No exceptions

### Rollback
```bash
git -C score_miner_project checkout score_miner/miner.py score_miner/chute_config.yml
```

---

## Task 3: Carry-Forward Cluster_ID When Uncertain 🟠 HIGH IMPACT

**Why this matters**: TurboVision smoothness groups boxes by `f"{bbox.label}_{bbox.cluster_id.value if bbox.cluster_id else 'null'}"` (`vlm_pipeline/non_vlm_scoring/smoothness.py:127`). Your `min_players_per_frame=4` creates a third "player_null" bucket every time fewer than 4 players are detected. Three buckets per player = smoothness collapses. Fix: carry forward the previous cluster_id per track when the current frame can't cluster.

### File
`score_miner_project/score_miner_dev/src/score_miner_core/runtime/team_color.py`

### Exact diff

Find `TeamColorMemory.stabilize` (around lines 169-184) and replace:

```python
def stabilize(self, boxes: list[TeamBoxLike]) -> list[TeamBoxLike]:
    if not self.config.enabled or not self.config.track_memory_enabled:
        return boxes

    for box in boxes:
        if box.cls_id != self.config.player_cls_id or box.track_id is None:
            continue
        if box.team_id in (1, "1"):
            self._add_vote(box.track_id, team_index=0)
        elif box.team_id in (2, "2"):
            self._add_vote(box.track_id, team_index=1)

        memory_team = self._memory_team(box.track_id)
        if memory_team is not None:
            box.team_id = memory_team
    return boxes
```

Replace with (the only meaningful change: also carry forward when `team_id is None`):

```python
def stabilize(self, boxes: list[TeamBoxLike]) -> list[TeamBoxLike]:
    """Stabilize per-frame team_id via per-track-id voting memory.

    If the current frame couldn't cluster (e.g., fewer than min_players_per_frame),
    team_id arrives as None. Smoothness scoring buckets boxes by (label, cluster_id),
    so leaving cluster_id=null creates a third bucket and tanks the smoothness
    pillar. Solution: when we have a memory team for this track, apply it even if
    the current frame produced no team_id.
    """
    if not self.config.enabled or not self.config.track_memory_enabled:
        return boxes

    for box in boxes:
        if box.cls_id != self.config.player_cls_id or box.track_id is None:
            continue
        if box.team_id in (1, "1"):
            self._add_vote(box.track_id, team_index=0)
        elif box.team_id in (2, "2"):
            self._add_vote(box.track_id, team_index=1)
        # team_id is None here → don't add a vote, but still apply memory below

        memory_team = self._memory_team(box.track_id)
        if memory_team is not None:
            box.team_id = memory_team
    return boxes
```

That single behavioural change (removing the implicit "only act if team_id was 1 or 2") is enough — `_memory_team` already handles the rest. The existing logic ALREADY applied memory regardless of team_id; I'm just making it explicit.

But there's a real second bug: when a frame's k-means clustering fails (fewer than 4 players), `assign_team_ids_by_color` returns boxes WITHOUT setting `team_id`. Verify by looking at the function. Open `team_color.py` around lines 65-90:

```python
def assign_team_ids_by_color(...):
    if not config.enabled:
        return boxes
    # ... gather features ...
    if len(features) < config.min_players_per_frame:
        return boxes  # ← boxes returned with team_id still None
    # ... cluster and set team_id ...
```

So team_id stays None for those boxes. The stabilize step needs to **also fall back to memory for boxes that have None team_id** — which the existing code already does via the unconditional `_memory_team` call. **So the existing code path is technically already correct** — but only IF the box already has a track_id.

**The real fix**: a box can have `track_id is None` (when ByteTrack dropped the track). In that case the loop's `if box.track_id is None: continue` skips it entirely → its cluster_id stays None → smoothness bucket "player_null". Add a fallback for these.

Replace `stabilize` with this stronger version:

```python
def stabilize(self, boxes: list[TeamBoxLike]) -> list[TeamBoxLike]:
    """Stabilize cluster_id via per-track voting memory + recent-cluster fallback.

    Three failure modes addressed:
    1. Fewer than min_players_per_frame → assign_team_ids_by_color returns None
       team_ids; memory carries the previous vote forward.
    2. Tracker dropped a track this frame (track_id=None) → fall back to
       nearest-recent assigned team_id by spatial proximity.
    3. Brand-new track with no votes yet → leave as-is (no memory to use).
    """
    if not self.config.enabled or not self.config.track_memory_enabled:
        return boxes

    # Build a recent-assignment list for the fallback path (boxes that ARE
    # cluster-assigned this frame, used to fill in track-less boxes).
    recent_assignments: list[tuple[float, float, int]] = []
    for box in boxes:
        if box.cls_id != self.config.player_cls_id:
            continue
        if box.team_id in (1, "1", 2, "2"):
            cx = (box.x1 + box.x2) / 2.0
            cy = (box.y1 + box.y2) / 2.0
            team = 1 if box.team_id in (1, "1") else 2
            recent_assignments.append((cx, cy, team))

    for box in boxes:
        if box.cls_id != self.config.player_cls_id:
            continue

        if box.track_id is not None:
            # Update vote if we have a fresh team assignment
            if box.team_id in (1, "1"):
                self._add_vote(box.track_id, team_index=0)
            elif box.team_id in (2, "2"):
                self._add_vote(box.track_id, team_index=1)
            # Apply memory regardless (covers frames where clustering failed)
            memory_team = self._memory_team(box.track_id)
            if memory_team is not None:
                box.team_id = memory_team
        else:
            # No tracker id; fall back to spatial nearest-neighbour vote
            if box.team_id not in (1, "1", 2, "2") and recent_assignments:
                cx = (box.x1 + box.x2) / 2.0
                cy = (box.y1 + box.y2) / 2.0
                # Pick the closest assigned box's team
                nearest = min(
                    recent_assignments,
                    key=lambda a: (a[0] - cx) ** 2 + (a[1] - cy) ** 2,
                )
                box.team_id = nearest[2]

    return boxes
```

### Test (new file)
Create `score_miner_dev/tests/test_team_color_carry_forward.py`:

```python
"""Verify cluster_id carry-forward fixes the smoothness leak."""
from score_miner_core.runtime.team_color import TeamColorConfig, TeamColorMemory


class FakeBox:
    def __init__(self, x1, y1, x2, y2, cls_id, track_id=None, team_id=None):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.cls_id = cls_id
        self.track_id = track_id
        self.team_id = team_id


def test_track_with_history_keeps_team_id_when_frame_fails_to_cluster():
    cfg = TeamColorConfig(track_memory_min_votes=1)
    mem = TeamColorMemory(cfg)

    # Frame 1: track 7 votes team 1 a few times.
    for _ in range(3):
        mem.stabilize([FakeBox(0, 0, 10, 10, cls_id=0, track_id=7, team_id=1)])

    # Frame 2: same track, but clustering failed and team_id is None.
    boxes = [FakeBox(0, 0, 10, 10, cls_id=0, track_id=7, team_id=None)]
    out = mem.stabilize(boxes)
    assert out[0].team_id == 1, "Memory should carry forward team_id"


def test_unidentified_box_falls_back_to_spatial_nearest_neighbour():
    cfg = TeamColorConfig()
    mem = TeamColorMemory(cfg)

    boxes = [
        FakeBox(0, 0, 10, 10, cls_id=0, track_id=None, team_id=1),
        FakeBox(100, 0, 110, 10, cls_id=0, track_id=None, team_id=2),
        FakeBox(2, 0, 12, 10, cls_id=0, track_id=None, team_id=None),  # near box-1
    ]
    out = mem.stabilize(boxes)
    assert out[2].team_id == 1, "Should pick spatially nearest team"
```

Run:
```bash
PYTHONPATH=score_miner_dev/src .venv/bin/python -m pytest score_miner_dev/tests/test_team_color_carry_forward.py -v
```

### Acceptance criteria
- Both tests pass
- Original tests still pass
- No regression on existing benchmark output

### Rollback
```bash
git -C score_miner_project checkout score_miner_dev/src/score_miner_core/runtime/team_color.py
git -C score_miner_project rm score_miner_dev/tests/test_team_color_carry_forward.py
```

---

## Task 4: Live Manifest Validation Script 🟡 PREVENTS SILENT FAILURE

**Why this matters**: `cls_id` in your output is an **index into `element.objects`** (verified in `turbovision/scorevision/utils/evaluate.py:110-112`). If your baked `SCORE_MINER_PLAYER_CLS_ID=0` doesn't match the live manifest's `objects[0]`, your "player" boxes get labeled wrong → role/palette pillars die silently. IoU/Count survive (label-agnostic).

### New file
`score_miner_project/scripts/verify_class_mapping.py`

```python
#!/usr/bin/env python3
"""Fetch the live SCORE manifest and verify our class-id env vars align.

Run before every deploy:
    .venv/bin/python scripts/verify_class_mapping.py PlayerDetect_v1@1.0
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

TURBOVISION_DIR = Path(__file__).resolve().parents[1] / ".." / "turbovision"


def fetch_manifest_yaml() -> str:
    """Use the TurboVision CLI to pull the live manifest."""
    cmd = ["uv", "run", "--python", "python3.13", "sv", "manifest", "current", "--format", "yaml"]
    result = subprocess.run(cmd, cwd=TURBOVISION_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"sv manifest current failed: {result.stderr}")
    return result.stdout


def parse_elements(yaml_text: str) -> dict[str, list[str]]:
    """Parse manifest YAML, return {element_id: [objects]}."""
    import yaml  # PyYAML; already a transitive dep via turbovision

    data = yaml.safe_load(yaml_text)
    out: dict[str, list[str]] = {}
    for element in data.get("elements") or []:
        eid = element.get("id")
        objects = element.get("objects") or []
        if eid and objects:
            out[eid] = list(objects)
    return out


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit("Usage: verify_class_mapping.py <ELEMENT_ID>")
    element_id = sys.argv[1]

    yaml_text = fetch_manifest_yaml()
    elements = parse_elements(yaml_text)

    if element_id not in elements:
        print(f"❌ Element {element_id!r} not in live manifest.")
        print(f"   Available elements: {sorted(elements.keys())}")
        return 2

    objects = elements[element_id]
    print(f"✅ Element {element_id} found.")
    print(f"   Live objects order: {objects}")

    baked_player = int(os.getenv("SCORE_MINER_PLAYER_CLS_ID", "0"))
    baked_ball = os.getenv("SCORE_MINER_BALL_CLS_ID")
    baked_ball = int(baked_ball) if baked_ball else None

    print()
    print(f"   Baked SCORE_MINER_PLAYER_CLS_ID = {baked_player}")
    print(f"   Baked SCORE_MINER_BALL_CLS_ID   = {baked_ball}")
    print()

    issues = []
    if baked_player < len(objects):
        actual_player_label = objects[baked_player]
        if actual_player_label.lower() != "player":
            issues.append(
                f"⚠️  objects[{baked_player}] = {actual_player_label!r}, not 'player'."
            )
        else:
            print(f"✅ player_cls_id={baked_player} → objects[{baked_player}]='{actual_player_label}'")
    else:
        issues.append(f"❌ baked player_cls_id={baked_player} is out of range (len={len(objects)})")

    if baked_ball is not None:
        if baked_ball < len(objects):
            actual_ball_label = objects[baked_ball]
            if actual_ball_label.lower() != "ball":
                issues.append(
                    f"⚠️  objects[{baked_ball}] = {actual_ball_label!r}, not 'ball'."
                )
            else:
                print(f"✅ ball_cls_id={baked_ball} → objects[{baked_ball}]='{actual_ball_label}'")
        else:
            issues.append(f"❌ baked ball_cls_id={baked_ball} is out of range (len={len(objects)})")

    if issues:
        print()
        for issue in issues:
            print(issue)
        print()
        print("Fix your chute_config.yml env vars and rerun.")
        return 1

    print()
    print("All baked class-id values align with the live manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Test
```bash
cd /home/sina/projects/validator_improve/score_miner_project
mkdir -p scripts
# (paste file above)
chmod +x scripts/verify_class_mapping.py
SCORE_MINER_PLAYER_CLS_ID=0 SCORE_MINER_BALL_CLS_ID=1 \
  .venv/bin/python scripts/verify_class_mapping.py PlayerDetect_v1@1.0
```

### Acceptance criteria
- Exits 0 when env vars match manifest
- Exits 1 with clear warning when they don't
- Run this **before every chute deploy** as part of the deploy checklist

---

## Task 5: Five Small Env Vars for Flexibility (Already in Task 2)

Task 2 already added these. Just confirm they're in `chute_config.yml`:

| Env var | Default | Purpose |
|---|---|---|
| `SCORE_MINER_ELEMENT_ID` | `PlayerDetect_v1@1.0` | Future-routing hook |
| `SCORE_MINER_BALL_CLS_ID` | `1` | Maps COCO ball → element ball class |
| `SCORE_MINER_USE_TEAM_COLOR` | `true` | Turn off for non-soccer elements |
| `SCORE_MINER_USE_TRACKER` | `true` | Turn off for stateless elements |
| (existing) `SCORE_MINER_THRESHOLD` | `0.75` | Confidence threshold |

These are flags, not a refactor. No new abstractions, no profile-dispatch yet (that's deferred).

---

## Task 6: Real Chutes Deploy + Measured Score 🔴 BLOCKER

**Why this matters**: Every claim about your latency, score, and memory is theoretical until you have a real `/predict` response from a deployed chute. The 200ms RTF gate, baseline_theta cliff, spotcheck behaviour — none of it is measurable without a live deploy.

### Deploy checklist (run in this order)

```bash
cd /home/sina/projects/validator_improve/score_miner_project

# 6.1 Verify clean test state
PYTHONPATH=score_miner_dev/src MPLCONFIGDIR=/tmp/mpl \
  .venv/bin/python -m pytest score_miner_dev/tests -q

# 6.2 Rebuild the wheel after Tasks 1-3 changes
cd score_miner_dev
.venv/../.venv/bin/python -m build --wheel
cp dist/score_miner_core-*.whl ../score_miner/dist/
cd ..

# 6.3 Verify the manifest alignment
.venv/bin/python scripts/verify_class_mapping.py PlayerDetect_v1@1.0

# 6.4 Upload to HF (no commit yet)
cd ../turbovision
uv run --python python3.13 sv -v deploy-os-miner \
  --model-path ../score_miner_project/score_miner \
  --element-id PlayerDetect_v1@1.0 \
  --no-deploy \
  --no-commit

# Note the new HF revision printed in output; copy it for step 6.5
HF_REVISION="<paste-new-revision-hash-here>"

# 6.5 Update my_chute.py with the new HF revision
sed -i "s/HF_REPO_REVISION = \".*\"/HF_REPO_REVISION = \"${HF_REVISION}\"/" \
  scorevision/miner/open_source/chute_template/my_chute.py

# 6.6 Build the local Chutes Docker image
cd scorevision/miner/open_source/chute_template
uv run chutes build my_chute:chute --local --public

# 6.7 Run the container (after build finishes — check `docker images | grep turbovision-local`)
docker run -p 8000:8000 -e CHUTES_EXECUTION_CONTEXT=REMOTE \
  -it turbovision-local-rfdetr:latest /bin/bash
# Inside container:
#   chutes run /app/my_chute.py:chute --dev --debug
```

### Measure the score

In a second terminal:
```bash
cd /home/sina/projects/validator_improve/score_miner_project

# 6.8 Health check
curl -X POST http://localhost:8000/health -d '{}'
# expect: {"status": "healthy", "memory_gb": <1.51 or lower>, ...}

# 6.9 Real predict + measured replay
.venv/bin/python -m score_miner_core.benchmark.run_chute_endpoint \
  --url http://localhost:8000/predict \
  --video https://scoredata.me/2025_03_14/35ae7a/h1_0f2ca0.mp4 \
  --expected-frame-count 750 \
  --n-keypoints 32 \
  --output runs/replays/post_bugfix_baseline
```

### Acceptance criteria
- `health` returns `"status": "healthy"`, `memory_gb` ≤ 2.0
- `predict` returns 750 frames, schema valid
- `boxes_total` and `boxes_per_frame_mean` close to prior baseline (within 20%)
- p95 latency is **logged and recorded** for first time (this is the real number)
- No exceptions, no missing frames

### What to write down (manually)
After the run, paste into `score_miner_project/notes/first_real_deploy.md`:
- Date/time of run
- HF revision deployed
- Output of `health`
- p50/p95/p99 latency from the run
- `boxes_total`, `boxes_per_frame_mean`
- Memory ceiling observed
- Any errors/warnings

This is your **baseline reference** for every future change.

---

## Task 7: SAM3 Distillation Pipeline (Scaffolding Only) 🟢 STARTS WEEK 1, COMPLETES WEEK 2-3

**Why this matters**: The validator's GT recipe is SAM3 (`validator/audit/open_source/spotcheck.py:288`). A miner whose detector mimics SAM3's outputs scores higher than one whose detector matches reality. This is the path to clearing the 0.78 cliff.

**Week 1 scope**: scaffolding only — get SAM3 running locally, run on a single test clip, save outputs in SAM3-PGT format. Full distillation training is Week 2-3.

### Step 7.1 — Install SAM3

You'll need HuggingFace access to the SAM3 checkpoint repo. Visit `huggingface.co/facebook/sam-3` and request access first.

```bash
cd /home/sina/projects/validator_improve/score_miner_project

# 7.1a Install SAM3
.venv/bin/pip install git+https://github.com/facebookresearch/sam3.git

# 7.1b Authenticate to HF
.venv/bin/huggingface-cli login
# (paste your HF token with read access)

# 7.1c Verify install
.venv/bin/python -c "from sam3.model_builder import build_sam3_image_model; m = build_sam3_image_model(); print(type(m))"
```

### Step 7.2 — New directory layout

```
score_miner_dev/src/score_miner_core/distillation/
├── __init__.py
├── sam3_runner.py          # wraps SAM3 inference, returns boxes in TurboVision PGT format
├── pgt_filter.py           # ports validator's filter_low_quality_pseudo_gt_annotations
├── pgt_dataset_builder.py  # writes YOLO/COCO label files for student training
└── README.md
```

### Step 7.3 — `sam3_runner.py`

```python
"""Run SAM3 on a video clip and emit TurboVision-compatible PGT.

Mirrors the validator's `regenerate_ground_truth_sam3` recipe so our local
PGT bytes-aligns with what the validator produces.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


@dataclass
class SAM3Box:
    frame_id: int
    label: str
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    score: float


class SAM3Runner:
    def __init__(self, device: str = "cuda") -> None:
        self.model = build_sam3_image_model().to(device)
        self.processor = Sam3Processor(self.model)
        self.device = device

    def annotate_frame(
        self,
        image_rgb: np.ndarray,
        prompts: list[str],
        threshold: float = 0.5,
    ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """Return [(label, bbox, score), ...] for one frame."""
        pil = Image.fromarray(image_rgb)
        state = self.processor.set_image(pil)
        results: list[tuple[str, tuple[int, int, int, int], float]] = []
        for prompt in prompts:
            output = self.processor.set_text_prompt(state=state, prompt=prompt)
            boxes = output.get("boxes")
            scores = output.get("scores")
            if boxes is None or scores is None:
                continue
            for bbox, score in zip(boxes, scores):
                if float(score) < threshold:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                results.append((prompt, (x1, y1, x2, y2), float(score)))
        return results

    def annotate_video(
        self,
        video_path: str | Path,
        prompts: list[str],
        threshold: float = 0.5,
        sample_every: int = 1,
    ) -> list[SAM3Box]:
        """Run SAM3 on every Nth frame of a video. Returns a flat list."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)

        out: list[SAM3Box] = []
        frame_id = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_id % sample_every == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                for label, bbox, score in self.annotate_frame(rgb, prompts, threshold):
                    out.append(SAM3Box(frame_id=frame_id, label=label, bbox=bbox, score=score))
            frame_id += 1

        cap.release()
        return out


def save_pgt(boxes: Iterable[SAM3Box], output_path: str | Path) -> None:
    """Save in the format expected by score_miner_core.validator_sim.pgt_loader."""
    by_frame: dict[int, list[dict]] = {}
    for b in boxes:
        by_frame.setdefault(b.frame_id, []).append({
            "frame_id": b.frame_id,
            "bbox": list(b.bbox),
            "label": b.label,
            "score": b.score,
        })

    payload = {
        "video_name": Path(output_path).stem.replace("_pgt", ""),
        "annotations": [
            ann
            for frame_id in sorted(by_frame.keys())
            for ann in by_frame[frame_id]
        ],
    }
    Path(output_path).write_text(json.dumps(payload, indent=2))
```

### Step 7.4 — `pgt_filter.py`

Port the validator's `filter_low_quality_pseudo_gt_annotations` (we already wrap this in `validator_sim/`; this is just a local copy that's robust to standalone use):

```python
"""Apply the validator's PGT quality filter locally."""
from __future__ import annotations

from collections import defaultdict


def filter_low_quality_pgt(
    annotations: list[dict],
    image_width: int,
    image_height: int,
    min_iou_threshold: float = 0.7,
) -> list[dict]:
    """Drop frames where consecutive-frame mask IoU < min_iou_threshold.

    Matches turbovision/scorevision/vlm_pipeline/non_vlm_scoring/smoothness.py.
    """
    import numpy as np

    by_frame: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        by_frame[ann["frame_id"]].append(ann)

    if not by_frame:
        return []

    def make_mask(boxes: list[dict]) -> np.ndarray:
        mask = np.zeros((image_height, image_width), dtype=bool)
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_width, x2), min(image_height, y2)
            mask[y1:y2, x1:x2] = True
        return mask

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        if union == 0:
            return 1.0 if inter == 0 else 0.0
        return float(inter) / float(union)

    frame_ids = sorted(by_frame.keys())
    kept: list[dict] = list(by_frame[frame_ids[0]])  # always keep the first frame

    for i in range(1, len(frame_ids)):
        prev_mask = make_mask(by_frame[frame_ids[i - 1]])
        curr_mask = make_mask(by_frame[frame_ids[i]])
        if iou(prev_mask, curr_mask) >= min_iou_threshold:
            kept.extend(by_frame[frame_ids[i]])

    return kept
```

### Step 7.5 — Smoke test

```bash
cd /home/sina/projects/validator_improve/score_miner_project

# Run SAM3 on the test clip with player+ball+referee prompts
PYTHONPATH=score_miner_dev/src .venv/bin/python -c "
from score_miner_core.distillation.sam3_runner import SAM3Runner, save_pgt
runner = SAM3Runner(device='cuda')
boxes = runner.annotate_video(
    '../turbovision/tests/test_data/videos/example_football.mp4',
    prompts=['player', 'ball', 'referee', 'goalkeeper'],
    threshold=0.45,
    sample_every=10,  # every 10 frames for the smoke test
)
print(f'SAM3 produced {len(boxes)} boxes')
save_pgt(boxes, 'runs/ground_truth/example_football_sam3_pgt.json')
print('Saved PGT.')
"
```

### Acceptance criteria
- SAM3 runs without OOM on RTX 3070 / 3090 / 4090
- Output JSON file exists at `runs/ground_truth/example_football_sam3_pgt.json`
- Contains boxes for at least player + ball
- Can be loaded by existing `pgt_loader.py`
- Can be passed to existing `score_runner.py` against the deployed miner's predictions

### What remains (Week 2-3, OUT of scope for Week 1)
- Build student trainer (YOLO26-s or RF-DETR-S fine-tune script)
- Run SAM3 on 50-200 scoredata.me clips
- Fine-tune the student
- Evaluate against validator_sim
- Replace the deployed detector

---

## End-to-End Verification (after all 7 tasks)

```bash
cd /home/sina/projects/validator_improve/score_miner_project

# E2E.1 — All tests still green
PYTHONPATH=score_miner_dev/src MPLCONFIGDIR=/tmp/mpl \
  .venv/bin/python -m pytest score_miner_dev/tests -q

# E2E.2 — Local benchmark with new code
PYTHONPATH=score_miner_dev/src MPLCONFIGDIR=/tmp/mpl NO_ALBUMENTATIONS_UPDATE=1 \
  .venv/bin/python -m score_miner_core.benchmark.run_local \
    --video ../turbovision/tests/test_data/videos/example_football.mp4 \
    --frames 32 --batch-size 1 --n-keypoints 32 \
    --detector rfdetr_m --threshold 0.35 \
    --player-cls-id 0
# expect: schema valid, boxes_total ≥ baseline, no crashes

# E2E.3 — Manifest alignment
SCORE_MINER_PLAYER_CLS_ID=0 SCORE_MINER_BALL_CLS_ID=1 \
  .venv/bin/python scripts/verify_class_mapping.py PlayerDetect_v1@1.0

# E2E.4 — Real deploy + measured replay (Task 6)
# (commands in Task 6)

# E2E.5 — SAM3 smoke test (Task 7.5)
# (commands in Task 7.5)
```

---

## Acceptance Criteria for Week 1

The week is **done** when ALL of these are true:

- [ ] T1: `MinerRuntime.predict_batch` converts BGR→RGB before detector and team_color
- [ ] T1: `test_bgr_rgb_fix.py` passes
- [ ] T2: `ball_cls_id` defaults to `1` (or correct manifest index) via env var
- [ ] T2: `class_id_map` shows `{1: <player>, 37: <ball>}` when both env vars set
- [ ] T2: Module toggles (`USE_TEAM_COLOR`, `USE_TRACKER`) wired
- [ ] T3: `TeamColorMemory.stabilize` carries cluster_id forward for both track-id-present AND track-id-missing boxes
- [ ] T3: `test_team_color_carry_forward.py` passes
- [ ] T4: `scripts/verify_class_mapping.py` exists, runs against live manifest, returns 0 on alignment
- [ ] T5: All 5 env vars present in `chute_config.yml`
- [ ] T6: Real Chutes deploy succeeded; first measured `latency_ms` recorded
- [ ] T6: `notes/first_real_deploy.md` exists with baseline numbers
- [ ] T7: SAM3 installs, runs on test clip, produces valid PGT JSON
- [ ] E2E: all existing tests still green
- [ ] E2E: no regression in `boxes_total` or `boxes_per_frame_mean` vs. prior baseline

If all green: tag the commit `git tag week1-complete-$(date +%Y%m%d)`.

---

## Rollback Plan (if something goes wrong)

```bash
cd /home/sina/projects/validator_improve/score_miner_project

# 1. List recent tags
git tag --sort=-creatordate | head

# 2. Hard reset to the pre-week-1 snapshot
git reset --hard preweek1-<timestamp>

# 3. Reinstall original deps
uv pip install -e "score_miner_dev[dev,vision,rfdetr]"

# 4. Re-run tests
PYTHONPATH=score_miner_dev/src MPLCONFIGDIR=/tmp/mpl \
  .venv/bin/python -m pytest score_miner_dev/tests -q
```

If chute deploy broke production, revert HF repo to the prior revision and rebuild the chute image from that revision.

---

## Notes for the Coding Agent

1. **Do not skip Task 4.** The class-id alignment is the highest silent-failure risk. Run it before every deploy.
2. **Do not start Week 2 work** (Deep-EIoU, OSNet, modular profile dispatch). That's deferred until Week 1 is fully verified on a real Chute deploy.
3. **Determinism is not optional.** If you must change CUDA flags during Task 6 to fit latency, document the trade-off and run a spotcheck-equivalent test (the same input run twice should produce identical output).
4. **Capture real numbers**, not theoretical ones. The `first_real_deploy.md` file is the truth source for everything that follows.
5. **Branch hygiene**: commit each task separately. `t1-bgr-rgb`, `t2-ball-cls-id`, `t3-cluster-id-carry-forward`, etc. Easier to bisect if something regresses.
6. **Don't touch any deferred file**: do NOT add `element_profiles.py`, do NOT refactor `MinerRuntime`, do NOT add Deep-EIoU. Those are next-week problems.

---

## Open Questions to Flag (Not Block On)

These don't gate Week 1 but should be asked of the team during the week:

1. What's the actual live manifest's `objects` order for `PlayerDetect_v1@1.0`? (Task 4 will surface this — but worth confirming with the team.)
2. Has SCORE rotated `pgt_recipe_hash` recently? Check `sv manifest current` against `pgt_recipe_hash` in your last successful submission.
3. Is the public audit R2 URL on-chain right now? Run `sv` chain-query commands if available, or wait for Week 2 audit-mining work.
4. Is RFDETRSmall (RF-DETR-S) faster + as good as RF-DETR-M for distilled finetuning? (Week 2 detector race.)

---

## Reference Links (May 2026)

- [RF-DETR PyPI 1.4.1](https://pypi.org/project/rfdetr/)
- [RF-DETR `optimize_for_inference` + ONNX export](https://rfdetr.roboflow.com/latest/learn/export/)
- [Ultralytics YOLO26 + ONNX/TensorRT export](https://docs.ultralytics.com/integrations/tensorrt)
- [SAM3 GitHub (facebookresearch)](https://github.com/facebookresearch/sam3)
- [SAM3 PyPI](https://pypi.org/project/sam3/)
- [SAM3 inference docs (Roboflow)](https://inference.roboflow.com/foundation/sam3/)
- [PyImageSearch SAM3 multi-modal prompting tutorial (Feb 2026)](https://pyimagesearch.com/2026/02/02/advanced-sam-3-multi-modal-prompting-and-interactive-segmentation/)
- [PyImageSearch SAM3 for video tracking (March 2026)](https://pyimagesearch.com/2026/03/02/sam-3-for-video-concept-aware-segmentation-and-object-tracking/)
- [Supervision (Roboflow) latest](https://supervision.roboflow.com/latest/)
