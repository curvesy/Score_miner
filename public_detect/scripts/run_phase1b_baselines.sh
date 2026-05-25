#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PATH="$HOME/.local/bin:$PATH"
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Run:"
  echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "source \$HOME/.local/bin/env"
  exit 1
fi

if [[ ! -f "data/yolo/car_wash_starter/data.yaml" || ! -f "data/yolo/beverage_starter/data.yaml" ]]; then
  echo "Starter YOLO data is missing. Downloading and preparing it now..."
  ./scripts/prepare_phase1_data.sh
fi

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
