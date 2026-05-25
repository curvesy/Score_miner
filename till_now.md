# Till Now

This is the current handoff/state for the Score/TurboVision open-source miner work.

## Goal

Build a Score open-source miner for the `PlayerDetect` element using a 2026-style real-time detection stack:

- RF-DETR-M first as the working smoke detector.
- Keep RF-DETR-L / DEIMv2-L / D-FINE-L for later head-to-head benchmarking.
- Use fast structured outputs, not a heavy VLM runtime.
- Validate schema, memory, latency, and Chutes packaging before any real on-chain commit.

## Current Correction From Week 1 Spec

The phase roadmap is no longer the authority for the next work. The current source of truth is:

```bash
score_miner_project/implementation_spec_week1.md
```

Use:

```bash
score_miner_project/game_changer_playbook.md
```

as strategic context only. Do not execute the deferred modular refactor, modular profile dispatch, full TEE work, or 30/60/90 roadmap yet.

Accepted correction:

- Do not start Phase 13 pitch keypoints for `PlayerDetect_v1@1.0` right now. The active manifest weights `iou`, `count`, `palette`, `smoothness`, and `role`; `keypoints_iou` is not weighted for the current PlayerDetect work.
- Do not start Phase 5 detector head-to-head yet. RF-DETR-L / DEIMv2-L / D-FINE-L benchmarking is still important, but only after the current baseline silent bugs are fixed.
- Do not call bootstrap PGT an honest validator score. It is a review/audit tool until SAM3 PGT scaffolding exists.
- Correct status: Phase 4 replay/schema/local endpoint tooling works; the SAM3 scoring oracle is not done yet.

Current next task order:

```text
Task 1: BGR/RGB image handling fix
Task 2: explicit ball_cls_id and runtime env toggles
Task 3: team carry-forward / null-team smoothness leak fix
Task 4: live manifest validator
Task 6: measured real Chutes deploy/baseline when credentials/funds allow
Task 7: local SAM3 PGT scaffold
```

## Active Project

Main working folder:

```bash
/home/sina/projects/validator_improve/score_miner_project
```

TurboVision repo:

```bash
/home/sina/projects/validator_improve/turbovision
```

Chutes SDK repo:

```bash
/home/sina/projects/validator_improve/chutes
```

The miner code is not copied into TurboVision. TurboVision is only the official CLI/validator repo used to upload, build, deploy, and eventually commit the miner.

## What We Built

Created a clean project package under:

```bash
score_miner_project/
```

Important files:

```bash
score_miner_project/score_miner_dev/pyproject.toml
score_miner_project/score_miner_dev/src/score_miner_core/runtime/class_mapping.py
score_miner_project/score_miner_dev/src/score_miner_core/runtime/memory_budget.py
score_miner_project/score_miner_dev/src/score_miner_core/runtime/miner_runtime.py
score_miner_project/score_miner_dev/src/score_miner_core/detector/rfdetr_runner.py
score_miner_project/score_miner_dev/src/score_miner_core/detector/detector_router.py
score_miner_project/score_miner_dev/src/score_miner_core/detector/class_id_mapper.py
score_miner_project/score_miner_dev/src/score_miner_core/runtime/postprocess.py
score_miner_project/score_miner_dev/src/score_miner_core/runtime/team_color.py
score_miner_project/score_miner_dev/src/score_miner_core/runtime/tracking.py
score_miner_project/score_miner_dev/src/score_miner_core/runtime/role_cleanup.py
score_miner_project/score_miner_dev/src/score_miner_core/benchmark/run_local.py
score_miner_project/score_miner_dev/src/score_miner_core/benchmark/schema_check.py
score_miner_project/score_miner_dev/src/score_miner_core/benchmark/latency.py
score_miner_project/score_miner/miner.py
score_miner_project/score_miner/chute_config.yml
score_miner_project/score_miner/chute_config_cu126.yml
score_miner_project/score_miner/dist/score_miner_core-0.1.2-py3-none-any.whl
```

The runtime is intentionally library-backed:

- `supervision.VideoInfo.from_video_path`
- `supervision.get_video_frames_generator`
- `supervision.Detections`
- `pydantic` v2 schema validation
- official `rfdetr` package adapter
- `psutil` and optional `torch.cuda` memory snapshots

## Local Python Environment

Created local venv in:

```bash
/home/sina/projects/validator_improve/score_miner_project/.venv
```

Installed:

```bash
uv pip install -e "score_miner_dev[dev,vision,rfdetr]"
```

Tests passed:

```bash
PYTHONPATH=score_miner_dev/src MPLCONFIGDIR=/tmp/mpl python -m pytest score_miner_dev/tests
```

Result:

```text
7 passed
```

## Local RF-DETR-M Smoke Test

Command used:

```bash
cd /home/sina/projects/validator_improve/score_miner_project

PYTHONPATH=score_miner_dev/src MPLCONFIGDIR=/tmp/mpl NO_ALBUMENTATIONS_UPDATE=1 python -m score_miner_core.benchmark.run_local \
  --video ../turbovision/tests/test_data/videos/example_football.mp4 \
  --frames 32 \
  --batch-size 1 \
  --n-keypoints 32 \
  --detector rfdetr_m \
  --threshold 0.35 \
  --player-cls-id 0
```

Important result:

```text
schema_check.valid = true
frames_processed = 32
boxes_total = 545
boxes_per_frame_mean = 17.0312
boxes_per_frame_max = 18
memory_after_load ~= 1.51 GB
memory_after_predict ~= 1.51 GB
p95 local latency ~= 289 ms
```

Local latency is not final because the local NVIDIA driver is too old for the installed CUDA/PyTorch stack:

```text
CUDA initialization: The NVIDIA driver on your system is too old
```

So local benchmark is currently useful for:

- schema
- box emission
- memory estimate
- package wiring

It is not reliable for final GPU latency.

## TurboVision CLI Setup

TurboVision needed Python 3.13, not Python 3.14, because its pinned Torch wheel does not support `cp314`.

Working setup:

```bash
cd /home/sina/projects/validator_improve/turbovision
uv venv --python python3.13
uv sync --python python3.13
uv run --python python3.13 sv elements list
```

Element list showed:

```text
PitchCalib
Detect
PlayerDetect
```

We are targeting:

```text
PlayerDetect_v1@1.0
```

## Hugging Face Upload

HF-only upload command:

```bash
cd /home/sina/projects/validator_improve/turbovision

uv run sv -v deploy-os-miner \
  --model-path ../score_miner_project/score_miner \
  --element-id PlayerDetect_v1@1.0 \
  --no-deploy \
  --no-commit
```

Result:

```text
Upload succeeded
10 files found
Detected revision: b9aec8857972c743035c2e3510699fd57d061fd3
Hf revision: b9aec8857972c743035c2e3510699fd57d061fd3
```

Current HF repo values for local Chutes test:

```python
HF_REPO_NAME = "Curvesy/ScoreVision"
HF_REPO_REVISION = "b9aec8857972c743035c2e3510699fd57d061fd3"
CHUTES_USERNAME = "local"
CHUTE_NAME = "turbovision-local-rfdetr"
```

## Why `sv_chutes.py` Was Missing

This command:

```bash
uv run sv -v deploy-os-miner \
  --model-path ../score_miner_project/score_miner \
  --element-id PlayerDetect_v1@1.0 \
  --no-deploy \
  --no-commit
```

only uploads to Hugging Face.

Because `--no-deploy` was used, TurboVision does not render `sv_chutes.py`. That file is temporary and only created during real Chutes deploy.

For no-money local testing, use the template manually:

```bash
cd /home/sina/projects/validator_improve/turbovision/scorevision/miner/open_source/chute_template
cp turbovision_chute.py.j2 my_chute.py
```

Then edit `my_chute.py` and set:

```python
HF_REPO_NAME = "Curvesy/ScoreVision"
HF_REPO_REVISION = "b9aec8857972c743035c2e3510699fd57d061fd3"
CHUTES_USERNAME = "local"
CHUTE_NAME = "turbovision-local-rfdetr"
```

Do not keep `{{ }}` in those values.

## Current Step

The user started local Chutes build:

```bash
cd /home/sina/projects/validator_improve/turbovision/scorevision/miner/open_source/chute_template
uv run chutes build my_chute:chute --local --public
```

The first build failed while installing CUDA PyTorch:

```text
Failed to download nvidia-cublas-cu12==12.8.4.1
Failed to download distribution due to network timeout.
Try increasing UV_HTTP_TIMEOUT (current value: 30s).
```

This was a Docker build network/extraction timeout, not a miner code failure.

Patched locally:

```bash
score_miner_project/score_miner/chute_config.yml
score_miner_project/score_miner/chute_config_cu126.yml
```

The Chutes install commands now prefix installs with:

```bash
UV_HTTP_TIMEOUT=300
```

The next build got past Torch and failed here:

```text
RUN UV_HTTP_TIMEOUT=300 pip install dist/score_miner_core-0.1.1-py3-none-any.whl
error: Distribution not found at: file:///app/dist/score_miner_core-0.1.1-py3-none-any.whl
```

Reason: Chutes local Docker build context is the chute template directory, not the downloaded Hugging Face repo directory. So `/app/dist/...` is not available inside the Docker image at build time.

Patched both Chutes configs again to install the wheel from Hugging Face instead:

```bash
UV_HTTP_TIMEOUT=300 pip install https://huggingface.co/Curvesy/ScoreVision/resolve/main/dist/score_miner_core-0.1.1-py3-none-any.whl
```

Important: the local Chutes build downloads the miner package from Hugging Face, so after this patch we must run another HF-only upload and update `my_chute.py` to the new HF revision before rebuilding locally.

This local build requires Docker. It does not require TAO, Chutes production registration, or on-chain commit.

## What To Do After Build Finishes

If the local build succeeds, start a container from the built image.

First check the actual image name:

```bash
docker images | grep turbovision-local-rfdetr
```

Then run the image. Try:

```bash
docker run -p 8000:8000 -e CHUTES_EXECUTION_CONTEXT=REMOTE -it turbovision-local-rfdetr:latest /bin/bash
```

If Docker says image not found, use the exact image name shown by `docker images`.

Inside the container, run:

```bash
cd /app
chutes run /app/my_chute.py:chute --dev --debug
```

If `/app/my_chute.py` is not present, find it:

```bash
find / -name my_chute.py 2>/dev/null
```

Then run:

```bash
chutes run <found-path>:chute --dev --debug
```

In a second terminal on the host machine, test health:

```bash
curl -X POST http://localhost:8000/health -d '{}'
```

Then test predict:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url":"https://scoredata.me/2025_03_14/35ae7a/h1_0f2ca0.mp4","meta":{}}'
```

## Chutes Production Registration Status

Do not spend real TAO yet.

Production Chutes registration failed because Chutes requires at least `0.25 TAO` on the coldkey:

```text
You must have at least 0.25 tao on your coldkey to register an account.
```

The coldkey checked was:

```text
5H8deNTX8atqyMvxufb24CoGLY7nCYBC16x5hFdJzLPQPAP2
```

The hotkey selected was:

```text
miner_hotkey
```

That hotkey selection was correct. Do not select `pub.txt` files or validator hotkey for the miner deploy.

Production Chutes is needed later only when we want Score validators to call the miner for real.

Current no-money path:

```text
local benchmark
-> HF-only upload
-> local Chutes build/run
-> local /health and /predict
-> then decide whether production deploy is worth funding
```

## Next Engineering Steps After Local Chutes Works

1. Confirm local `/health` works.
2. Confirm local `/predict` returns valid JSON.
3. The first local `/predict` worked, but loaded `detector: None`, so it returned valid empty fallback boxes.
4. Patched `score_miner/miner.py` to load `rfdetr_m` by default through `create_detector`.
5. Patched `chute_config.yml` and `chute_config_cu126.yml` to install RF-DETR's non-Torch dependencies after installing `rfdetr --no-deps`.
6. Local container import check then showed RF-DETR was installed but incompatible with the unpinned package version:

```text
ImportError: cannot import name 'BackboneConfigMixin' from 'transformers'
```

Local working environment uses:

```text
rfdetr==1.5.2
transformers==4.57.6
```

Patched Chutes configs to pin:

```bash
pip install rfdetr==1.5.2 --no-deps
pip install transformers==4.57.6 ...
```

7. Upload the patched deploy package to Hugging Face again, update `my_chute.py` to the new revision, rebuild local Chutes image, and test `/predict` again.
8. Added a local validator-style endpoint replay runner:

```bash
score_miner_dev/src/score_miner_core/benchmark/run_chute_endpoint.py
score_miner_dev/src/score_miner_core/benchmark/prediction_summary.py
score_miner_dev/tests/test_prediction_summary.py
```

It calls the actual Chutes `/predict` endpoint, saves replay artifacts, validates the returned JSON, and creates a rich summary.

Run it while the local container is serving on port 8000:

```bash
cd /home/sina/projects/validator_improve/score_miner_project

.venv/bin/python -m score_miner_core.benchmark.run_chute_endpoint \
  --url http://localhost:8000/predict \
  --video https://scoredata.me/2025_03_14/35ae7a/h1_0f2ca0.mp4 \
  --expected-frame-count 750 \
  --n-keypoints 32 \
  --output runs/replays/rfdetr_m_local_chute_smoke
```

Artifacts written:

```text
runs/replays/<run>/
  request.json
  response.json
  summary.json
  report.json
  report.md
```

Summary includes:

```text
success/error
schema_check
frames_returned
frame_id continuity
empty/nonempty frames
boxes_total
boxes/frame min/max/mean/p50/p95/p99
confidence min/max/mean/p50/p95/p99
class_counts
team_id_counts
cluster_id_counts
keypoint counts
valid keypoint counts
```

Tests:

```bash
cd /home/sina/projects/validator_improve/score_miner_project
.venv/bin/python -m pytest score_miner_dev/tests
```

Result:

```text
9 passed
```

9. Next: wrap TurboVision's actual scorer into `validator_sim` so these saved predictions can produce pillar scores:
   - Done as a strict wrapper that requires a PGT/GT JSON file. TurboVision test data has a manifest and video but no local GT annotation file, so the scorer cannot honestly produce a score from replay alone.
   - Added:

```text
score_miner_dev/src/score_miner_core/validator_sim/__init__.py
score_miner_dev/src/score_miner_core/validator_sim/replay_loader.py
score_miner_dev/src/score_miner_core/validator_sim/pgt_loader.py
score_miner_dev/src/score_miner_core/validator_sim/score_runner.py
score_miner_dev/tests/test_validator_sim_replay_loader.py
```

PGT JSON format:

```json
{
  "video_name": "example_football",
  "annotations": [
    {
      "frame_id": 0,
      "bbox": [100, 100, 150, 240],
      "label": "player",
      "score": 1.0
    }
  ]
}
```

Run when a replay and PGT file exist:

```bash
cd /home/sina/projects/validator_improve/score_miner_project

.venv/bin/python -m score_miner_core.validator_sim.score_runner \
  --replay-dir runs/replays/rfdetr_m_local_chute_smoke \
  --pgt runs/ground_truth/example_football_pgt.json \
  --manifest ../turbovision/tests/test_data/manifests/example_manifest.yml \
  --turbovision-path ../turbovision \
  --element-id PlayerDetect_v1@1.0
```

Outputs:

```text
runs/replays/<run>/score_report.json
runs/replays/<run>/score_report.md
```

Verified:

```text
13 passed
compileall passed
```

10. Added review-required PGT bootstrap and frame export tools:

```text
score_miner_dev/src/score_miner_core/validator_sim/pgt_bootstrap.py
score_miner_dev/src/score_miner_core/validator_sim/export_review_frames.py
score_miner_dev/tests/test_pgt_bootstrap.py
```

These follow the 2026 model-assisted annotation pattern: use model predictions as prelabels, export review images, and require manual correction before treating labels as ground truth.

Generated from the local replay:

```text
runs/ground_truth/example_football_pgt_bootstrap.json
runs/review_frames/rfdetr_m_local_chute_smoke/
```

Bootstrap result:

```text
16 selected frames
182 predicted boxes
review_required = true
missing_requested_frames = []
```

Review frames exported:

```text
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000000.jpg
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000050.jpg
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000100.jpg
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000150.jpg
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000200.jpg
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000250.jpg
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000300.jpg
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000350.jpg
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000400.jpg
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000450.jpg
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000500.jpg
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000550.jpg
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000600.jpg
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000650.jpg
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000700.jpg
runs/review_frames/rfdetr_m_local_chute_smoke/frame_000749.jpg
```

Important: do not use `example_football_pgt_bootstrap.json` as final score truth without review. It is model output copied into a PGT format. It is useful for tooling smoke tests and manual correction, not an honest benchmark by itself.

After manual correction, save as:

```text
runs/ground_truth/example_football_pgt_reviewed.json
```

Then run:

```bash
cd /home/sina/projects/validator_improve/turbovision

PYTHONPATH=/home/sina/projects/validator_improve/score_miner_project/score_miner_dev/src \
uv run python -m score_miner_core.validator_sim.score_runner \
  --replay-dir /home/sina/projects/validator_improve/score_miner_project/runs/replays/rfdetr_m_local_chute_smoke \
  --pgt /home/sina/projects/validator_improve/score_miner_project/runs/ground_truth/example_football_pgt_reviewed.json \
  --manifest /home/sina/projects/validator_improve/turbovision/tests/test_data/manifests/example_manifest.yml \
  --turbovision-path /home/sina/projects/validator_improve/turbovision \
  --element-id PlayerDetect_v1@1.0
```
11. Added PGT quality/audit gate and Label Studio export:

```text
score_miner_dev/src/score_miner_core/validator_sim/pgt_audit.py
score_miner_dev/src/score_miner_core/validator_sim/export_labelstudio.py
score_miner_dev/tests/test_pgt_audit.py
```

Generated:

```text
runs/ground_truth/example_football_pgt_bootstrap_audit.json
runs/labelstudio/rfdetr_m_local_chute_smoke_tasks.json
```

Audit result:

```text
annotations = 182
frames = 16
invalid_annotations = []
score_ready = false
review_required = true
review_status_counts.needs_manual_review = 182
warnings:
  - PGT still contains review-required or missing review_status labels.
  - PGT only contains player labels; role/team/ball scoring remains incomplete.
```

This is intentional. The code now prevents accidentally treating RF-DETR bootstrap labels as honest ground truth.

Run audit after manual correction:

```bash
cd /home/sina/projects/validator_improve/score_miner_project

.venv/bin/python -m score_miner_core.validator_sim.pgt_audit \
  --pgt runs/ground_truth/example_football_pgt_reviewed.json \
  --output runs/ground_truth/example_football_pgt_reviewed_audit.json
```

Only use reviewed PGT for real score decisions when `score_ready=true`.

Tests:

```text
14 passed
compileall passed
```
12. Added confidence threshold sweep for cleaner review prelabels:

```text
score_miner_dev/src/score_miner_core/validator_sim/threshold_sweep.py
score_miner_dev/tests/test_threshold_sweep.py
```

Run:

```bash
cd /home/sina/projects/validator_improve/score_miner_project

.venv/bin/python -m score_miner_core.validator_sim.threshold_sweep \
  --replay-dir runs/replays/rfdetr_m_local_chute_smoke \
  --expected-frame-count 750 \
  --n-keypoints 32 \
  --target-boxes-per-frame-min 8 \
  --target-boxes-per-frame-max 12 \
  --output runs/replays/rfdetr_m_local_chute_smoke/threshold_sweep.json
```

Recommended review threshold:

```text
threshold = 0.75
boxes_total = 6231
boxes/frame mean = 8.308
boxes/frame p95 = 12.0
empty_frames = 0
```

Generated cleaner review set:

```text
runs/ground_truth/example_football_pgt_bootstrap_t075.json
runs/ground_truth/example_football_pgt_bootstrap_t075_audit.json
runs/review_frames/rfdetr_m_local_chute_smoke_t075/
runs/labelstudio/rfdetr_m_local_chute_smoke_t075_tasks.json
```

Cleaner bootstrap result:

```text
16 selected frames
139 predicted boxes
8.6875 boxes/frame average on selected frames
review_required = true
score_ready = false
```

Use the `_t075` review images instead of the original messier `0.5` threshold images.

Tests:

```text
15 passed
compileall passed
```
13. Run smoke tests on more than one video chunk.
14. Validator_sim was run against the unreviewed `_t075` bootstrap only as a plumbing test:

```bash
cd /home/sina/projects/validator_improve/turbovision

PYTHONPATH=/home/sina/projects/validator_improve/score_miner_project/score_miner_dev/src \
UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
uv run python -m score_miner_core.validator_sim.score_runner \
  --replay-dir /home/sina/projects/validator_improve/score_miner_project/runs/replays/rfdetr_m_local_chute_smoke \
  --pgt /home/sina/projects/validator_improve/score_miner_project/runs/ground_truth/example_football_pgt_bootstrap_t075.json \
  --manifest /home/sina/projects/validator_improve/turbovision/tests/test_data/manifests/example_manifest.yml \
  --turbovision-path /home/sina/projects/validator_improve/turbovision \
  --element-id PlayerDetect_v1@1.0 \
  --output /home/sina/projects/validator_improve/score_miner_project/runs/replays/rfdetr_m_local_chute_smoke/score_report_t075_unreviewed.json
```

Result:

```text
score_available = true
mean_weighted = 0.01
pgt_frames = 16
prediction_frames = 750
```

Important: this is NOT an honest model score because the PGT is still unreviewed bootstrap output and audit says `score_ready=false`. Use it only as proof that validator_sim wiring works.

15. `turbovision/scorevision/spotcheck/orchestrator/ground_truth_client.py` was inspected. It fetches private-track ground truth from:

```text
{api_url}/api/private-track/ground-truth/{challenge_id}
```

with bearer auth. This can help if we have `api_url`, `challenge_id`, and `auth_token`, but it does not provide local open-source GT by itself.

16. Remaining Phase 4 work:

```text
- Get honest GT/PGT, either by private GT API access, reviewed manual labels, or SAM3 pseudo-GT.
- Score replay against that GT.
- Add threshold/max-box calibration runner against validator_sim score.
- Run at least 2 more replay clips so the score loop is not fitted to one video.
```

17. Only after the container works and score loop exists, consider real Chutes production registration/deploy.

18. Added private ground-truth fetch integration:

```text
score_miner_dev/src/score_miner_core/validator_sim/ground_truth_client.py
score_miner_dev/tests/test_ground_truth_client.py
```

It mirrors the repo's spotcheck idea from:

```text
turbovision/scorevision/spotcheck/orchestrator/ground_truth_client.py
```

but saves a local validator_sim-compatible PGT JSON and optional raw API response.

Use only if we have:

```text
SCORE_GT_API_URL
SCORE_GT_CHALLENGE_ID
SCORE_GT_AUTH_TOKEN
```

Run:

```bash
cd /home/sina/projects/validator_improve/score_miner_project

SCORE_GT_API_URL="https://..." \
SCORE_GT_CHALLENGE_ID="..." \
SCORE_GT_AUTH_TOKEN="..." \
.venv/bin/python -m score_miner_core.validator_sim.ground_truth_client \
  --output runs/ground_truth/example_football_pgt_api.json \
  --raw-output runs/ground_truth/example_football_pgt_api_raw.json \
  --video-name example_football
```

Then audit:

```bash
.venv/bin/python -m score_miner_core.validator_sim.pgt_audit \
  --pgt runs/ground_truth/example_football_pgt_api.json \
  --output runs/ground_truth/example_football_pgt_api_audit.json
```

Tests:

```text
16 passed
compileall passed
```

19. Returned focus to actual miner runtime, not validator tooling. Added runtime postprocess config:

```text
score_miner_dev/src/score_miner_core/runtime/postprocess.py
score_miner_dev/src/score_miner_core/runtime/miner_runtime.py
score_miner/miner.py
score_miner_dev/tests/test_postprocess.py
score_miner_dev/tests/test_miner_runtime.py
```

Deploy/runtime defaults:

```text
SCORE_MINER_THRESHOLD = 0.75
SCORE_MINER_MAX_BOXES_PER_FRAME = 18
SCORE_MINER_MIN_BOX_AREA = 0
```

Behavior:

```text
- detector still uses SCORE_MINER_THRESHOLD
- runtime applies final confidence filtering
- runtime keeps top-N boxes per frame by confidence
- runtime can drop tiny boxes with SCORE_MINER_MIN_BOX_AREA
- schema remains unchanged
```

This is real miner work. It changes what `/predict` returns and does not depend on private GT/API access.

Verification:

```text
18 passed
compileall passed
empty-detector local benchmark schema valid
wheel rebuilt cleanly:
score_miner/dist/score_miner_core-0.1.1-py3-none-any.whl
```

Next deploy-package step:

```bash
cd /home/sina/projects/validator_improve/turbovision

uv run sv -v deploy-os-miner \
  --model-path /home/sina/projects/validator_improve/score_miner_project/score_miner \
  --element-id PlayerDetect_v1@1.0 \
  --no-deploy \
  --no-commit
```

Then update `my_chute.py` to the new HF revision, rebuild local Chutes image, and run `/predict` again. Expected summary should be close to the previous threshold-sweep result:

```text
boxes_total about 6231
boxes/frame mean about 8.3
empty_frames 0
schema valid
```

## Important Current Decision

Do not pay TAO just to test.

Local Chutes build/run is the correct next step.

20. Added the first runtime palette component: player team color clustering.

Files:

```text
score_miner_dev/src/score_miner_core/runtime/team_color.py
score_miner_dev/src/score_miner_core/runtime/miner_runtime.py
score_miner/miner.py
score_miner_dev/tests/test_team_color.py
score_miner_dev/tests/test_miner_runtime.py
plan_for_score.md
```

Runtime defaults:

```text
SCORE_MINER_TEAM_COLOR_ENABLED = true
SCORE_MINER_TEAM_MIN_PLAYERS = 4
SCORE_MINER_TEAM_MIN_CROP_PIXELS = 24
SCORE_MINER_TEAM_TORSO_TOP_RATIO = 0.15
SCORE_MINER_TEAM_TORSO_BOTTOM_RATIO = 0.65
SCORE_MINER_TEAM_TORSO_CENTER_WIDTH_RATIO = 0.70
SCORE_MINER_TEAM_EXCLUDE_GRASS = true
SCORE_MINER_TEAM_KMEANS_ATTEMPTS = 3
SCORE_MINER_TEAM_KMEANS_MAX_ITER = 20
SCORE_MINER_TEAM_RANDOM_SEED = 2026
```

Behavior:

```text
- Takes final filtered player boxes.
- Extracts upper-body/torso crops.
- Removes likely grass pixels.
- Converts crop colors to OpenCV Lab.
- Runs OpenCV k-means with k=2 per frame.
- Emits team_id 1 or 2 for player boxes only.
- Leaves cluster_id null until real tracking exists, to avoid fake identity flips hurting smoothness.
```

Why this is the correct next miner step:

```text
- It changes real /predict output.
- It targets TurboVision's palette pillar.
- It does not require private GT or validator API access.
- It is cheap enough for runtime and can be disabled by env if it hurts.
- It creates a baseline that VideoState/ReID can stabilize later.
```

Verification:

```text
20 passed
compileall passed
```

Next deploy-package step remains:

```bash
cd /home/sina/projects/validator_improve/turbovision

uv run sv -v deploy-os-miner \
  --model-path /home/sina/projects/validator_improve/score_miner_project/score_miner \
  --element-id PlayerDetect_v1@1.0 \
  --no-deploy \
  --no-commit
```

Then update `my_chute.py` to the new HF revision, rebuild local Chutes, run `/predict`, and save:

```text
runs/replays/rfdetr_m_postprocess_t075_teamcolor
```

21. Added internal ByteTrack tracking and track-based team memory.

Files:

```text
score_miner_dev/src/score_miner_core/runtime/tracking.py
score_miner_dev/src/score_miner_core/runtime/team_color.py
score_miner_dev/src/score_miner_core/runtime/miner_runtime.py
score_miner/miner.py
score_miner_dev/tests/test_tracking.py
score_miner_dev/tests/test_team_color.py
plan_for_score.md
```

Runtime defaults:

```text
SCORE_MINER_TRACKING_ENABLED = true
SCORE_MINER_TRACK_ACTIVATION_THRESHOLD = 0.25
SCORE_MINER_LOST_TRACK_BUFFER = 30
SCORE_MINER_MINIMUM_MATCHING_THRESHOLD = 0.8
SCORE_MINER_TRACK_ASSIGNMENT_IOU = 0.5
SCORE_MINER_TRACK_FRAME_RATE = 25.0
SCORE_MINER_MINIMUM_CONSECUTIVE_FRAMES = 1
SCORE_MINER_TEAM_TRACK_MEMORY_ENABLED = true
SCORE_MINER_TEAM_TRACK_MEMORY_MIN_VOTES = 2
SCORE_MINER_TEAM_TRACK_MEMORY_MAX_VOTES = 20
```

Important schema decision:

```text
- Do not output ByteTrack IDs as cluster_id.
- TurboVision's parser normalizes cluster_id/team_id into team color.
- Fake identity IDs in cluster_id would hurt palette and role scoring.
- ByteTrack IDs stay internal and stabilize team_id only.
```

Verification:

```text
24 passed
compileall passed
RF-DETR-M 8-frame smoke:
schema valid
boxes_total 42
boxes/frame mean 5.25
memory_after_predict about 1.54 GB RSS
```

22. Added guarded role cleanup baseline.

Files:

```text
score_miner_dev/src/score_miner_core/runtime/role_cleanup.py
score_miner_dev/src/score_miner_core/runtime/miner_runtime.py
score_miner/miner.py
score_miner_dev/tests/test_role_cleanup.py
plan_for_score.md
```

Runtime defaults:

```text
SCORE_MINER_ROLE_CLEANUP_ENABLED = true
SCORE_MINER_REFEREE_CLS_ID = unset
SCORE_MINER_REFEREE_MIN_CONFIDENCE = 0.85
SCORE_MINER_REFEREE_MIN_TEAM_DISTANCE = 35
SCORE_MINER_REFEREE_MARGIN = 8
SCORE_MINER_REFEREE_MAX_PER_FRAME = 2
```

Behavior:

```text
- No-op unless SCORE_MINER_REFEREE_CLS_ID is explicitly set.
- If enabled, extracts torso Lab color features.
- Computes team centroids from team_id 1/2.
- Relabels only high-confidence player boxes whose torso color is far from both teams.
- Clears team_id for referee relabels.
```

Why guarded:

```text
TurboVision parses cls_id through the active manifest object order. A wrong referee class ID can be worse than keeping all boxes as players. Enable only after the live manifest confirms class order.
```

Verification:

```text
26 passed
compileall passed
RF-DETR-M 8-frame smoke:
schema valid
boxes_total 42
boxes/frame mean 5.25
memory_after_predict about 1.54 GB RSS
```

23. Finished Phase 4 for local, non-private development.

Phase 4 status:

```text
Complete for replay/schema/review/local validator_sim wiring.
Honest score oracle is not complete yet; Week 1 Task 7 should add local SAM3 PGT scaffolding.
```

Current phase boundary:

```text
Phase 0 - done
Phase 1 - done
Phase 2 - done
Phase 3 - done
Phase 4 - replay/schema/local endpoint tooling done
SAM3 scoring oracle - not done
Next authority - implementation_spec_week1.md, not old phase numbers
```

Created:

```text
phase_status.md
```

Latest accepted replay:

```text
runs/replays/rfdetr_m_t075_team_tracking_role_guard_v1
```

Expected replay files:

```text
request.json
response.json
summary.json
report.json
report.md
```

Latest successful replay summary:

```text
success true
schema_check.valid true
frames_returned 750
missing_frame_ids []
empty_frames 0
boxes_total 6231
boxes_per_frame.mean 8.308
confidence.min 0.750405
team_id_counts {"1": 3269, "2": 2779, "null": 183}
class_counts {"0": 6231}
valid_keypoints_total 0
```

Stop point before next phase:

```text
Do not enable referee/GK relabel until live manifest object order is known.
Do not claim a validator score from bootstrap PGT.
Do not start Phase 13 keypoints for PlayerDetect while keypoints_iou is absent from the manifest.
Do not start Phase 5 detector head-to-head until silent-bug fixes are complete.
Next work package is implementation_spec_week1.md in task order.
```

24. Started Week 1 silent-bug fixes.

Completed Task 1, with one important production/local distinction:

```text
Chutes template frame source:
  cv2.imdecode(..., IMREAD_COLOR)
  cv2.VideoCapture.read()
  -> BGR

Local benchmark frame source:
  supervision.get_video_frames_generator
  -> handled as RGB for benchmark parity
```

Code changes:

```text
score_miner_dev/src/score_miner_core/runtime/miner_runtime.py
score_miner_dev/src/score_miner_core/benchmark/run_local.py
score_miner/miner.py
score_miner_dev/tests/test_miner_runtime.py
```

Runtime behavior:

```text
MinerRuntime input_color_space default = "bgr"
score_miner/miner.py reads SCORE_MINER_INPUT_COLOR_SPACE, default "bgr"
run_local.py defaults --input-color-space rgb
detector, team_color, and role_cleanup now receive RGB images internally
```

Completed Task 3 carry-forward fix:

```text
score_miner_dev/src/score_miner_core/runtime/team_color.py
score_miner_dev/tests/test_team_color_carry_forward.py
```

Behavior:

```text
track_id with prior team votes carries team_id through weak frames
track-less player boxes can use nearest same-frame assigned team as fallback
no arbitrary cluster_id is emitted
```

Verification:

```text
PYTHONPATH=score_miner_dev/src MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest score_miner_dev/tests -q
32 passed

RF-DETR-M local smoke:
schema_check.valid true
frames_processed 32
boxes_total 211
boxes_per_frame_mean 6.5938
memory_after_predict about 1.54 GB RSS
```

Note:

```text
The old implementation_spec_week1.md acceptance number of >=545 boxes came from a pre-postprocess baseline.
The current runtime keeps the production threshold/postprocess path active, so lower local box count is expected.
The Chutes endpoint replay is still the real check after uploading the rebuilt wheel.
```

25. Added manifest/class verifier because `sv manifest current` requires R2 credentials.

File:

```text
scripts/verify_class_mapping.py
```

Why:

```text
uv run --python python3.13 sv manifest current
fails locally with RuntimeError: R2 credentials not set.
Miner-side verification should use the public manifest index or a local manifest file.
```

Commands:

```bash
.venv/bin/python scripts/verify_class_mapping.py \
  --manifest ../turbovision/tests/test_data/manifests/example_manifest.yml \
  --element-id PlayerDetect_v1@1.0

.venv/bin/python scripts/verify_class_mapping.py \
  --url https://turbo.scoredata.me/manifest/index.json \
  --list
```

Public manifest result on 2026-05-22:

```text
source https://turbo.scoredata.me/manifest/8218320-39f3e736421460b495ea78390191b09ddf98b21d63dbc3643798d589f91597bb.yaml
PlayerDetect_v1@1.0 not present
active public detection elements:
  manak0/Detect-beverage-detect
  manak0/Detect-crime
  manak0/Detect-fire
  manak0/Detect-car-wash
active private elements:
  manako/DetectCricketDelivery
  manako/DetectFootballEvent
```

Implication:

```text
The local PlayerDetect miner is valid as a Chutes/package/runtime exercise, but it is not currently an active public manifest element.
Do not spend more time optimizing PlayerDetect for live score unless the active manifest again contains PlayerDetect_v1@1.0 or a matching public football element.
```
