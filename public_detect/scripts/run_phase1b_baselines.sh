#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"

COMMON_ARGS=()
if [[ -n "${TRAIN_IMGSZ:-}" ]]; then
  COMMON_ARGS+=(--imgsz "${TRAIN_IMGSZ}")
fi
if [[ -n "${TRAIN_BATCH:-}" ]]; then
  COMMON_ARGS+=(--batch "${TRAIN_BATCH}")
fi
if [[ -n "${TRAIN_DEVICE:-}" ]]; then
  COMMON_ARGS+=(--device "${TRAIN_DEVICE}")
fi

uv run python scripts/check_gpu_env.py

uv run python scripts/train_baseline.py \
  --config configs/training/car_wash_yolo11n.yaml \
  "${COMMON_ARGS[@]}"

uv run python scripts/train_baseline.py \
  --config configs/training/car_wash_yolo26n.yaml \
  "${COMMON_ARGS[@]}"

uv run python scripts/train_baseline.py \
  --config configs/training/beverage_yolo11n.yaml \
  "${COMMON_ARGS[@]}"

uv run python scripts/train_baseline.py \
  --config configs/training/beverage_yolo26n.yaml \
  "${COMMON_ARGS[@]}"

echo
echo "Phase 1B baseline runs completed. Copy back:"
echo "$(pwd)/runs/"

