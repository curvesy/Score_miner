#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Run:"
  echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "source \$HOME/.local/bin/env"
  exit 1
fi

export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu128}"

echo "Preparing Score starter data for Car-wash and Beverage..."

uv run python scripts/download_starter_pack.py \
  --element-config configs/elements/car_wash.yaml \
  --output-root data/starter_packs

uv run python scripts/download_starter_pack.py \
  --element-config configs/elements/beverage.yaml \
  --output-root data/starter_packs

uv run python scripts/build_yolo_dataset.py \
  --element-config configs/elements/car_wash.yaml \
  --starter-dir data/starter_packs/car_wash \
  --output-dir data/yolo/car_wash_starter

uv run python scripts/build_yolo_dataset.py \
  --element-config configs/elements/beverage.yaml \
  --starter-dir data/starter_packs/beverage \
  --output-dir data/yolo/beverage_starter

uv run python scripts/render_starter_labels.py \
  --element-config configs/elements/car_wash.yaml \
  --starter-dir data/starter_packs/car_wash \
  --output-dir reports/label_previews/car_wash

uv run python scripts/render_starter_labels.py \
  --element-config configs/elements/beverage.yaml \
  --starter-dir data/starter_packs/beverage \
  --output-dir reports/label_previews/beverage

echo
echo "Data ready:"
echo "  data/yolo/car_wash_starter/data.yaml"
echo "  data/yolo/beverage_starter/data.yaml"
