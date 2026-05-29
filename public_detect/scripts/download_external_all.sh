#!/usr/bin/env bash
# Top-level orchestrator: pull every external beverage dataset and remap to
# YOLO cup/bottle/can in data/yolo_candidates/<source>_v1.
#
# Usage:
#   bash scripts/download_external_all.sh [--skip-coco]
#
# Prereqs:
#   - fiftyone     (uv pip install fiftyone)        for Open Images V7
#   - kaggle CLI   (uv pip install kaggle)          for Kaggle datasets
#   - ~/.kaggle/kaggle.json (chmod 600)             API token
#   - ~30 GB free disk if --skip-coco is NOT set    (COCO train2017 is 18 GB)
set -euo pipefail

PROJECT="/home/sina/projects/validator_improve/score_miner_project/public_detect"
UV_CACHE_DIR="/home/sina/projects/validator_improve/.uv-cache"

SKIP_COCO=0
for arg in "$@"; do
  case "$arg" in
    --skip-coco) SKIP_COCO=1 ;;
    *) echo "unknown arg: $arg"; exit 1 ;;
  esac
done

cd "${PROJECT}"

echo "============================================================"
echo "[1/4] Open Images V7 (Bottle / Cup / Tin can)"
echo "============================================================"
UV_CACHE_DIR="${UV_CACHE_DIR}" \
PYTHONPATH=src uv run python scripts/download_openimages_beverage.py \
  --output-dir data/yolo_candidates/beverage_oiv7_v1 \
  --split validation \
  --max-samples 5000

echo "============================================================"
echo "[2/4] Kaggle drinking-waste (cans + bottles)"
echo "============================================================"
UV_CACHE_DIR="${UV_CACHE_DIR}" \
PYTHONPATH=src uv run python scripts/download_kaggle_drinkwaste.py \
  --output-dir data/yolo_candidates/beverage_drinkwaste_v1 \
  --max-images 2500

echo "============================================================"
echo "[3/4] Kaggle smoking-and-drinking (people holding drinks)"
echo "============================================================"
UV_CACHE_DIR="${UV_CACHE_DIR}" \
PYTHONPATH=src uv run python scripts/download_kaggle_smokedrink.py \
  --output-dir data/yolo_candidates/beverage_smokedrink_v1 \
  --drinking-as cup \
  --max-images 1000

if [ "${SKIP_COCO}" -eq 1 ]; then
  echo "============================================================"
  echo "[4/4] COCO 2017 person+bottle/cup  --  SKIPPED"
  echo "============================================================"
else
  echo "============================================================"
  echo "[4/4] COCO 2017 person+bottle/cup (~18 GB download)"
  echo "============================================================"
  bash scripts/download_coco_beverage.sh
fi

echo
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
for d in beverage_oiv7_v1 beverage_drinkwaste_v1 beverage_smokedrink_v1 beverage_coco_v1; do
  full="${PROJECT}/data/yolo_candidates/${d}"
  if [ -d "${full}/images/train" ]; then
    n_imgs=$(find "${full}/images/train" -type f | wc -l)
    n_lbls=$(find "${full}/labels/train" -type f | wc -l)
    echo "  ${d}: ${n_imgs} images / ${n_lbls} labels"
  else
    echo "  ${d}: not built"
  fi
done
echo
echo "Next: run scripts/build_phase4_dataset.py with these as --source args."
