#!/usr/bin/env bash
# Full Phase 4 external-v1 retrain + scoring pipeline.
#
# Steps:
#   1. Sanity-check the dataset built by build_phase4_dataset.py
#   2. Train YOLO11n on data/yolo/beverage_phase4_external_v1
#   3. Score-sweep on the model's own val split  (mixed distribution)
#   4. Score-sweep on the starter val split      (apples-to-apples vs old runs)
#   5. SAHI score-sweep on the starter val split (tiled inference, small-object boost)
#   6. simulate_validator_score.py on starter    (real manifest: 0.6*map50 + 0.4*fp)
#
# Run from the project root:
#   bash scripts/retrain_phase4_external.sh
#
# Override knobs via env:
#   EPOCHS=120 BATCH=4 IMGSZ=960 NAME_SUFFIX=local bash scripts/retrain_phase4_external.sh
set -euo pipefail

PROJECT="/home/sina/projects/validator_improve/score_miner_project/public_detect"
UV_CACHE_DIR="/home/sina/projects/validator_improve/.uv-cache"

DATASET_DIR="${PROJECT}/data/yolo/beverage_phase4_external_v1"
STARTER_DATA="${PROJECT}/data/yolo/beverage_starter/data.yaml"
TRAIN_CONFIG="configs/training/beverage_yolo11n_phase4_external_v1.yaml"
RUN_NAME="yolo11n_phase4_external_v1"

EPOCHS="${EPOCHS:-120}"
BATCH="${BATCH:-4}"
IMGSZ="${IMGSZ:-960}"
NAME_SUFFIX="${NAME_SUFFIX:-local}"

cd "${PROJECT}"

run() {
  echo
  echo "============================================================"
  echo "$ $*"
  echo "============================================================"
  "$@"
}

# -------------------------------------------------------------------------
echo "[0/6] dataset sanity check"
# -------------------------------------------------------------------------
if [ ! -f "${DATASET_DIR}/data.yaml" ]; then
  echo "missing ${DATASET_DIR}/data.yaml -- run build_phase4_dataset.py first" >&2
  exit 1
fi
n_train_imgs=$(find "${DATASET_DIR}/images/train" -type f | wc -l)
n_train_lbls=$(find "${DATASET_DIR}/labels/train" -type f | wc -l)
n_val_imgs=$(find "${DATASET_DIR}/images/val" -type f | wc -l)
n_val_lbls=$(find "${DATASET_DIR}/labels/val" -type f | wc -l)
echo "  train: ${n_train_imgs} images / ${n_train_lbls} labels"
echo "  val  : ${n_val_imgs} images / ${n_val_lbls} labels"
if [ "${n_train_imgs}" -lt 100 ]; then
  echo "train set too small (<100); aborting" >&2
  exit 1
fi

# -------------------------------------------------------------------------
echo "[1/6] train YOLO11n"
# -------------------------------------------------------------------------
run env UV_CACHE_DIR="${UV_CACHE_DIR}" PYTHONPATH=src uv run python scripts/train_baseline.py \
  --config "${TRAIN_CONFIG}" \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --imgsz "${IMGSZ}" \
  --name-suffix "${NAME_SUFFIX}"

MODEL_DIR="runs/beverage/${RUN_NAME}_${NAME_SUFFIX}"
BEST_PT="${MODEL_DIR}/weights/best.pt"
if [ ! -f "${BEST_PT}" ]; then
  echo "training finished but ${BEST_PT} not found" >&2
  exit 1
fi
echo "best weights: ${BEST_PT}"

# -------------------------------------------------------------------------
echo "[2/6] score sweep on mixed val (own dataset)"
# -------------------------------------------------------------------------
run env UV_CACHE_DIR="${UV_CACHE_DIR}" PYTHONPATH=src uv run python scripts/score_threshold_sweep.py \
  --model "${BEST_PT}" \
  --data "${DATASET_DIR}/data.yaml" \
  --name "${RUN_NAME}_${NAME_SUFFIX}_mixedval_single" \
  --base-conf 0.001 \
  --per-class

# -------------------------------------------------------------------------
echo "[3/6] score sweep on STARTER val (apples-to-apples vs prior runs)"
# -------------------------------------------------------------------------
run env UV_CACHE_DIR="${UV_CACHE_DIR}" PYTHONPATH=src uv run python scripts/score_threshold_sweep.py \
  --model "${BEST_PT}" \
  --data "${STARTER_DATA}" \
  --name "${RUN_NAME}_${NAME_SUFFIX}_starter_single" \
  --base-conf 0.001 \
  --per-class

# -------------------------------------------------------------------------
echo "[4/6] SAHI tiled score sweep on STARTER val"
# -------------------------------------------------------------------------
run env UV_CACHE_DIR="${UV_CACHE_DIR}" PYTHONPATH=src uv run python scripts/score_threshold_sweep.py \
  --model "${BEST_PT}" \
  --data "${STARTER_DATA}" \
  --name "${RUN_NAME}_${NAME_SUFFIX}_starter_sahi" \
  --base-conf 0.001 \
  --per-class \
  --prediction-mode sahi \
  --sahi-slice-height 640 \
  --sahi-slice-width 640 \
  --sahi-overlap 0.25 \
  --sahi-postprocess-iou 0.5

# -------------------------------------------------------------------------
echo "[5/6] multi-pillar diagnostic on STARTER val"
# -------------------------------------------------------------------------
# Default weights inside simulate_validator_score.py are the real Beverage
# manifest: 0.6 * map50 + 0.4 * false_positive
run env UV_CACHE_DIR="${UV_CACHE_DIR}" PYTHONPATH=src uv run python scripts/simulate_validator_score.py \
  --model "${BEST_PT}" \
  --data "${STARTER_DATA}" \
  --conf 0.10 \
  --max-det 100 \
  --imgsz "${IMGSZ}"

# -------------------------------------------------------------------------
echo "[6/6] summary"
# -------------------------------------------------------------------------
echo
echo "============================================================"
echo " RESULTS"
echo "============================================================"
echo "  model       : ${BEST_PT}"
echo "  sweeps dir  : reports/score_sweeps/"
echo
echo "  mixed val sweep  : reports/score_sweeps/${RUN_NAME}_${NAME_SUFFIX}_mixedval_single/"
echo "  starter single   : reports/score_sweeps/${RUN_NAME}_${NAME_SUFFIX}_starter_single/"
echo "  starter SAHI     : reports/score_sweeps/${RUN_NAME}_${NAME_SUFFIX}_starter_sahi/"
echo
echo "Look at the 'best.score' field of each summary.json to compare."
echo "Decision rule:"
echo "  <0.55  -> not deployable, iterate on data"
echo "  0.55-0.65 -> probe live, expect noise"
echo "  0.65-0.70 -> serious candidate"
echo "  >0.70  -> deploy"
