#!/usr/bin/env bash
# Iteration 2: rebuild dataset with all-miners harvest + oversample Score-distribution
# sources, then warm-start fine-tune from the iteration-1 best.pt.
#
# Pre-reqs (all already produced):
#   data/yolo_candidates/beverage_winner_proxy_v2/   (149 frames, navierstocks labels)
#   data/yolo_candidates/beverage_all_miners_v1/     (447 frames, all-miner labels)
#   runs/beverage/yolo26s_phase4_winner_v1_local/weights/best.pt   (after iter1 train finishes)
#
# Outputs:
#   data/yolo/beverage_phase4_iter2_v1/
#   runs/beverage/yolo26s_phase4_iter2_v1_local/weights/best.pt
#
# Time: ~5 min rebuild + ~1.5 hours warm-start train (40 epochs).
set -euo pipefail

PROJECT="/home/sina/projects/validator_improve/score_miner_project/public_detect"
UV_CACHE_DIR="/home/sina/projects/validator_improve/.uv-cache"
cd "${PROJECT}"

ITER1_BEST="${PROJECT}/runs/beverage/yolo26s_phase4_winner_v1_local/weights/best.pt"
if [ ! -f "${ITER1_BEST}" ]; then
  echo "iter1 weights not found at ${ITER1_BEST}" >&2
  echo "  -> run iter1 training first; this script warm-starts from iter1" >&2
  exit 1
fi

echo "============================================================"
echo "[1/4] rebuild Phase 4 dataset (winnerproxy 3x + all_miners 2x)"
echo "============================================================"
UV_CACHE_DIR="${UV_CACHE_DIR}" \
PYTHONPATH=src uv run python scripts/build_phase4_dataset.py \
  --element-config configs/elements/beverage.yaml \
  --output-dir data/yolo/beverage_phase4_iter2_v1 \
  --source starter:data/yolo/beverage_starter \
  --source manako:data/yolo_candidates/beverage_manako_autolabeled \
  --source winnerproxy:data/yolo_candidates/beverage_winner_proxy_v2::3x \
  --source allminers:data/yolo_candidates/beverage_all_miners_v1::2x \
  --source oiv7:data/yolo_candidates/beverage_oiv7_v1 \
  --source drinkwaste:data/yolo_candidates/beverage_drinkwaste_v1:800 \
  --source smokedrink:data/yolo_candidates/beverage_smokedrink_v1 \
  --val-fraction 0.10

echo
echo "============================================================"
echo "[2/4] make iter2 training config (warm-start from iter1)"
echo "============================================================"
cat > configs/training/beverage_yolo26s_phase4_iter2_v1.yaml <<EOF
element_config: configs/elements/beverage.yaml
data: data/yolo/beverage_phase4_iter2_v1/data.yaml
model: ${ITER1_BEST}
project: runs/beverage
name: yolo26s_phase4_iter2_v1
epochs: 40
imgsz: 1280
batch: 2
patience: 15
device: 0
workers: 4
seed: 44
optimizer: auto
cos_lr: true
close_mosaic: 5
cache: false
amp: true
plots: true
val: true
EOF
cat configs/training/beverage_yolo26s_phase4_iter2_v1.yaml

echo
echo "============================================================"
echo "[3/4] warm-start fine-tune (~1.5 hrs)"
echo "============================================================"
UV_CACHE_DIR="${UV_CACHE_DIR}" \
PYTHONPATH=src uv run python scripts/train_baseline.py \
  --config configs/training/beverage_yolo26s_phase4_iter2_v1.yaml \
  --epochs 40 --batch 2 --imgsz 1280 --name-suffix local

BEST_ITER2="${PROJECT}/runs/beverage/yolo26s_phase4_iter2_v1_local/weights/best.pt"
if [ ! -f "${BEST_ITER2}" ]; then
  echo "iter2 training did not produce best.pt" >&2
  exit 1
fi

echo
echo "============================================================"
echo "[4/4] probe new model with winner-style inference"
echo "============================================================"
UV_CACHE_DIR="${UV_CACHE_DIR}" \
PYTHONPATH=src uv run python scripts/score_winner_style.py \
  --model "${BEST_ITER2}" \
  --data data/yolo_candidates/beverage_winner_proxy_v2/data.yaml \
  --imgsz 1280

echo
echo "ALSO scoring on bigger all-miners proxy set:"
UV_CACHE_DIR="${UV_CACHE_DIR}" \
PYTHONPATH=src uv run python scripts/score_winner_style.py \
  --model "${BEST_ITER2}" \
  --data data/yolo_candidates/beverage_all_miners_v1/data.yaml \
  --imgsz 1280

echo
echo "DONE. iter2 weights: ${BEST_ITER2}"
echo "Decision:"
echo "  proxy >= 0.55  -> export ONNX + deploy + register on subnet"
echo "  proxy <  0.55  -> harvest more frames, iterate 3"
