"""Environment checks for GPU training machines."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from typing import Any


def _run(cmd: list[str]) -> dict[str, Any]:
    if shutil.which(cmd[0]) is None:
        return {"cmd": cmd, "available": False}
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    return {
        "cmd": cmd,
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def check_environment(require_cuda: bool = True) -> dict[str, Any]:
    import torch
    import ultralytics
    from ultralytics import YOLO

    models: dict[str, str] = {}
    for model_name in ("yolo11n.pt", "yolo26n.pt", "yolo26s.pt", "yolov8n.pt"):
        try:
            YOLO(model_name)
            models[model_name] = "ok"
        except Exception as exc:  # pragma: no cover - environment-specific
            models[model_name] = f"error: {type(exc).__name__}: {exc}"

    cuda_available = bool(torch.cuda.is_available())
    result = {
        "python_torch": {
            "torch": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
        },
        "ultralytics": ultralytics.__version__,
        "models": models,
        "nvidia_smi": _run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }
    if require_cuda and not cuda_available:
        result["status"] = "fail"
        result["reason"] = "torch.cuda.is_available() is false"
    elif any(value != "ok" for value in models.values()):
        result["status"] = "fail"
        result["reason"] = "one or more baseline model weights failed to load"
    else:
        result["status"] = "ok"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-require-cuda", action="store_true")
    args = parser.parse_args()
    result = check_environment(require_cuda=not args.no_require_cuda)
    print(json.dumps(result, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
