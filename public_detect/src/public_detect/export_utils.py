"""Export and size-gate helpers for public Detect deploy artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    total = 0
    for item in root.rglob("*"):
        if ".git" in item.parts:
            continue
        if item.is_file():
            total += item.stat().st_size
    return total


def check_size_gate(path: str | Path, max_mb: float = 30.0) -> dict[str, Any]:
    root = Path(path)
    size_bytes = directory_size_bytes(root)
    max_bytes = int(max_mb * 1_000_000)
    files = []
    for item in sorted(root.rglob("*")):
        if ".git" in item.parts or not item.is_file():
            continue
        files.append(
            {
                "path": str(item.relative_to(root)),
                "bytes": item.stat().st_size,
            }
        )
    return {
        "path": str(root),
        "size_bytes": size_bytes,
        "size_mb_decimal": size_bytes / 1_000_000,
        "max_mb_decimal": max_mb,
        "max_bytes": max_bytes,
        "passes": size_bytes <= max_bytes,
        "files": files,
    }


def copy_export_to_repo(exported_model: str | Path, output_dir: str | Path) -> Path:
    source = Path(exported_model)
    if not source.exists():
        raise FileNotFoundError(source)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "weights.onnx"
    shutil.copy2(source, target)
    return target


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
