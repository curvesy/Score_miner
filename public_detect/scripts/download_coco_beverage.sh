#!/usr/bin/env bash
# Download COCO 2017 train + annotations (~19 GB) and ingest cup/bottle subset.
#
# Outputs:
#   data/external/coco2017/{train2017/, annotations/instances_train2017.json}
#   data/yolo_candidates/beverage_coco_v1/{images,labels}/train/
#
# Skips work that's already done (re-runs are cheap).
set -euo pipefail

PROJECT="/home/sina/projects/validator_improve/score_miner_project/public_detect"
COCO_DIR="${PROJECT}/data/external/coco2017"
OUT_DIR="${PROJECT}/data/yolo_candidates/beverage_coco_v1"
CONFIG="${PROJECT}/configs/data_sources/beverage_coco2017.yaml"

mkdir -p "${COCO_DIR}"
cd "${COCO_DIR}"

if [ ! -d "annotations" ]; then
  echo "[coco] downloading annotations (~241 MB)"
  curl -L -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -q annotations_trainval2017.zip
  rm annotations_trainval2017.zip
else
  echo "[coco] annotations already present, skipping"
fi

if [ ! -d "train2017" ]; then
  echo "[coco] downloading train2017 images (~18 GB) — this is the long step"
  echo "[coco] if you want to skip, Ctrl+C and use --skip-images on the ingest call"
  curl -L -O http://images.cocodataset.org/zips/train2017.zip
  unzip -q train2017.zip
  rm train2017.zip
else
  echo "[coco] train2017 images already present, skipping"
fi

cd "${PROJECT}"

echo "[coco] ingesting via ingest_coco_source.py -> ${OUT_DIR}"
UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
PYTHONPATH=src uv run python scripts/ingest_coco_source.py \
  --config "${CONFIG}" \
  --coco-json "${COCO_DIR}/annotations/instances_train2017.json" \
  --image-root "${COCO_DIR}/train2017" \
  --output-dir "${OUT_DIR}"

echo "[coco] done. output: ${OUT_DIR}"
