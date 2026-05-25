# Research Sources

This file records sources checked before implementation. Add dated notes before writing related code.

## 2026-05-25 Baseline Research

### Ultralytics Python Training And Export

Source: https://docs.ultralytics.com/usage/python/

Why it matters:

```text
The current Ultralytics Python API supports loading pretrained models, training custom data, validation, prediction, and ONNX export through the `YOLO` class.
```

### Ultralytics YOLO26

Source: https://docs.ultralytics.com/models/yolo26/

Why it matters:

```text
YOLO26 is current in Ultralytics docs and supports detect/train/val/export flows through the same Ultralytics API.
```

### Ultralytics Detection Task Docs

Source: https://docs.ultralytics.com/tasks/detect/

Why it matters:

```text
Detection data format, train/val/export examples, and YOLO26 detect-task usage should drive the dataset/config layout.
```

### Hugging Face Model Upload

Source: https://huggingface.co/docs/hub/en/models-uploading

Why it matters:

```text
Public Detect validation sums the Hugging Face model repository size, so deployment must use a minimal HF model repo revision.
```

Source: https://huggingface.co/docs/huggingface_hub/main/en/guides/upload

Why it matters:

```text
The Python `huggingface_hub` upload workflow should be used for reproducible revision uploads once packaging is ready.
```

### SAHI / Slicing

Source: https://arxiv.org/abs/2202.06934

Why it matters:

```text
Slicing aided inference can help small objects, but it is not first-line. Use only after baseline if it improves Score-style validation without increasing false positives.
```

### YOLO11 vs YOLO26 At Small Scale

Source: https://www.mdpi.com/2079-9292/15/6/1146

Why it matters:

```text
The plan should not assume YOLO26n beats YOLO11n under a 30MB cap. Run both.
```

### Ultralytics SAHI Guide

Source: https://docs.ultralytics.com/guides/sahi-tiled-inference/

Why it matters:

```text
SAHI is directly documented with current Ultralytics YOLO flows. It is an extension experiment for small objects such as can/nozzle, not the first deploy path.
```

### RF-DETR Current Repo

Source: https://github.com/roboflow/rf-detr

Why it matters:

```text
RF-DETR remains a strong fine-tuning detector family, but public Detect has a 30MB full-repo cap. Test Nano/Small size only after YOLO baselines and do not assume it fits.
```

### SAM3 / Promptable Concept Segmentation

Source: https://docs.ultralytics.com/models/sam-3/

Source: https://arxiv.org/abs/2511.16719

Why it matters:

```text
SAM3 can be used offline for concept-guided pseudo-labeling and review. It should not be used live for public Detect first because model size/latency/30MB deployment constraints are wrong for the task.
```

### Grounding DINO

Source: https://arxiv.org/abs/2303.05499

Why it matters:

```text
Grounding DINO is useful as an offline open-vocabulary pseudo-label teacher for unlabeled public images, especially for rare class examples. It is not a live model candidate under the 30MB cap.
```

### Supervision

Source: https://supervision.roboflow.com/

Why it matters:

```text
Supervision provides a standard Detections object and converters for Ultralytics, Transformers, SAM, Roboflow, and other model outputs. Use it to avoid custom annotation/detection glue where practical.
```

### CVAT Auto Annotation

Source: https://docs.cvat.ai/docs/annotation/auto-annotation/automatic-annotation/

Why it matters:

```text
CVAT supports model-assisted annotation workflows. Use it only if manual review volume becomes a bottleneck; do not make it a Phase 0 dependency.
```

### MLflow Tracking

Source: https://mlflow.org/docs/latest/ml/tracking/

Why it matters:

```text
MLflow records parameters, metrics, artifacts, and dataset metadata for training runs. Use one tracker so model/threshold/live-score decisions are reproducible.
```

### Optuna

Source: https://optuna.org/

Why it matters:

```text
Optuna 4.x supports black-box optimization. Use it after brute-force sweeps to tune thresholds, image size, max detections, and SAHI/TTA options against Score-style validation.
```

## Pending Source Checks

- [ ] Current Chutes Score miner deployment docs/path.
- [ ] Current RT-DETR tiny export size and ONNX compatibility.
- [ ] Current Ultralytics embedded-NMS / end-to-end export behavior for YOLO11 and YOLO26.

## 2026-05-26 Phase 3 Scoring Research

### Ultralytics Predict Results

Source: https://docs.ultralytics.com/modes/predict/

Why it matters:

```text
Ultralytics predict mode returns Results objects with Boxes fields including
xyxy coordinates, confidence scores, and class IDs. Phase 3 should use these
Python objects directly and cache raw detections, instead of parsing CLI logs.
```

### Ultralytics Validation Arguments

Source: https://docs.ultralytics.com/modes/val/

Source: https://docs.ultralytics.com/usage/cfg/

Why it matters:

```text
Ultralytics validation defaults use low confidence, configurable NMS IoU, and
max_det. Our local sweep follows the same principle: run low-confidence
prediction once, then tune confidence/max_det against the Score-style metric.
```

### COCO/VOC-Style AP50

Source: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

Why it matters:

```text
AP is computed from precision-recall behavior after confidence sorting and IoU
matching. For SN44 Phase 3 we only need a transparent AP50 approximation plus
false-positive pressure, not a full COCO benchmark dependency.
```

## 2026-05-26 Phase 4 Data Source Research

### Roboflow Universe Dataset Search

Source: https://roboflow.com/universe

Source: https://docs.roboflow.com/roboflow/roboflow-jp/universe/find-a-dataset-on-universe

Why it matters:

```text
Roboflow Universe is useful for finding candidate object-detection datasets, but
the search is semantic. Keyword matches such as "drain" or "washer" can return
wrong-domain data. Candidate datasets must be visually filtered against Score
starter/proof frames before training.
```

### Beverage Containers Candidate

Source: https://universe.roboflow.com/roboflow-universe-projects/beverage-containers-3atxb

Why it matters:

```text
This is a real beverage-container object-detection source with bottle/cup/can-like
classes. It is a useful Beverage candidate if mapped carefully to Score classes
and filtered for cluttered/context images over clean product shots.
```

### TACO Trash Annotations In Context

Source: https://github.com/pedropro/TACO

Source: https://datasetninja.com/taco

Source: https://arxiv.org/abs/2003.06975

Why it matters:

```text
TACO contains real contextual waste/litter images and classes such as clear
plastic bottle, drink can, disposable plastic cup, glass bottle, paper cup, and
cartons. It is a strong Beverage source and hard-negative source because it is
messy/contextual, not product catalog data.
```

### GroundingDINO / Offline Teacher Labels

Source: https://github.com/IDEA-Research/GroundingDINO

Why it matters:

```text
GroundingDINO can propose boxes from text prompts for outside images and video
frames. Use it offline only; teacher labels need manual review before training.
```

### Supervision

Source: https://supervision.roboflow.com/

Why it matters:

```text
Supervision provides practical converters and visualization tools for object
detection data. Use it for review/export glue rather than hand-rolled one-off
annotation formats where practical.
```

### FiftyOne / CVAT Review

Source: https://docs.voxel51.com/user_guide/evaluation.html

Source: https://docs.cvat.ai/docs/annotation/auto-annotation/automatic-annotation/

Why it matters:

```text
FiftyOne and CVAT are useful when Phase 4 review volume grows. They can help
inspect false positives, false negatives, and teacher labels. They should not
replace manual review of high-value examples.
```

### COCO Detection Format

Source: https://claru.ai/formats/coco-format

Why it matters:

```text
TACO and many Roboflow exports use COCO-style detection annotations: images,
annotations, categories, and bbox in [x, y, width, height]. Phase 4 ingestion
maps those categories into Score's exact class order and writes YOLO labels.
```

### yt-dlp And FFmpeg For Video Sources

Source: https://github.com/yt-dlp/yt-dlp

Source: https://renderio.dev/blogs/ffmpeg-extract-frames

Why it matters:

```text
Car-wash public labeled data is scarce. The practical source is real video
frames from car-wash facilities. The project script uses ffmpeg for local frame
extraction; web downloads should be handled separately with current yt-dlp and
source/license recorded before labels are created.
```

## 2026-05-25 Phase 1 Training Check

### uv Project Setup

Source: https://docs.astral.sh/uv/

Source: https://docs.astral.sh/uv/concepts/projects/dependencies/

Source: https://docs.astral.sh/uv/guides/integration/pytorch/

Why it matters:

```text
uv is the current Python project manager for fast, reproducible environments. The PyTorch integration supports `UV_TORCH_BACKEND=auto`, which is the cleanest setup for rented GPU machines with different CUDA driver stacks.
```

### PyTorch CUDA Install

Source: https://pytorch.org/get-started/

Why it matters:

```text
The official PyTorch selector provides CUDA wheel indexes including CUDA 12.8. The project setup should not hard-code a guessed CUDA package; it should resolve the correct CUDA backend and then verify `torch.cuda.is_available()` before training.
```

### Ultralytics YOLO Python API

Source: https://docs.ultralytics.com/usage/python/

Source: https://docs.ultralytics.com/modes/train/

Source: https://docs.ultralytics.com/tasks/detect/

Why it matters:

```text
Current Ultralytics training uses the `YOLO(model).train(data=..., epochs=..., imgsz=...)` API. The public Detect starter datasets are already in YOLO detection format with `data.yaml`, so the baseline runner should stay thin and call the official API instead of building custom training loops.
```

### YOLO26 Availability

Source: https://docs.ultralytics.com/models/yolo26/

Local check:

```text
ultralytics==8.4.14
yolo11n.pt OK
yolo26n.pt OK
yolo26s.pt OK
yolov8n.pt OK
GPU: NVIDIA GeForce RTX 3070 Laptop GPU, 8GB VRAM
```

Why it matters:

```text
YOLO26 is available in this local Ultralytics install, so it can be included in Phase 1. Because local VRAM is 8GB, first baselines should use nano models with conservative batch sizes. YOLO26s should be tested after nano smoke runs and may need lower batch or rented 16-24GB GPU for larger image sizes.
```

Updated project lock, resolved 2026-05-25:

```text
Python 3.12 target
torch 2.11.x
torchvision 0.26.x
ultralytics 8.4.54
torchaudio omitted because public Detect does not need audio
CUDA 12.8 backend is the default because rented GPU drivers may not support CUDA 13 wheels yet
```

## 2026-05-25 Winning-Loop Research Update

### Synthetic-To-Real Domain Randomization

Source: https://arxiv.org/abs/2509.15045

Why it matters:

```text
The reported winning setup used diverse synthetic data and domain randomization. For public Detect, Phase 4 should generate varied camera angles, lighting, clutter, occlusion, compression, and object scale rather than many near-duplicate images.
```

### YOLO26 Small-Object Features

Source: https://arxiv.org/abs/2601.12882

Source: https://www.ultralytics.com/blog/ultralytics-yolo26-the-new-standard-for-edge-first-vision-ai

Why it matters:

```text
YOLO26's NMS-free design plus STAL/ProgLoss make YOLO26n/YOLO26s worth testing for small objects and false-positive control. It does not remove the need to bake off YOLO11n on the exact Score task.
```

### SAHI For Small Objects

Source: https://docs.ultralytics.com/guides/sahi-tiled-inference/

Why it matters:

```text
SAHI can recover small/distant objects but can also increase duplicate detections. It belongs after the baseline and must be judged by Score-style validation, not recall alone.
```

### Teacher Pseudo-Labeling

Source: https://arxiv.org/abs/2303.05499

Source: https://docs.ultralytics.com/models/sam-3/

Why it matters:

```text
GroundingDINO/SAM3-style teachers are useful offline for pseudo-labels and hard-case discovery. They should be reviewed manually and distilled into the sub-30MB deploy model, not deployed live for public Detect.
```

## 2026-05-25 Phase 0 API Check

### ScoreVision Console V2 Element Detail

Source: `https://console.scorevision.io/api/v2/elements/<urlencoded_element_id>?lookback_days=4`

Checked live for:

```text
manak0/Detect-car-wash
manak0/Detect-beverage-detect
```

Observed top-level fields:

```text
element
starterPack
latestAnnotatedChallenge
challengeDetails
leaderboard
history
```

Observed `starterPack.starterAssets[]` fields:

```text
assetId
imageUrl
annotationUrl
frameIndex
objects
annotations
```

Observed annotation shape:

```text
className: string
bbox: [x1, y1, x2, y2]
```

Why it matters:

```text
Phase 0 downloader should prefer embedded annotations but also save the annotationUrl payload when available. The YOLO converter must map className through the manifest/starter objects order and treat boxes as absolute pixel xyxy.
```
