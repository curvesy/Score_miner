#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install with:"
  echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "then run:"
  echo "source \$HOME/.local/bin/env"
  exit 1
fi

export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"

uv venv --clear --python "${PYTHON_VERSION:-3.12}"
uv sync --all-extras
uv run python scripts/check_gpu_env.py

echo
echo "Environment ready. Activate with:"
echo "source .venv/bin/activate"
