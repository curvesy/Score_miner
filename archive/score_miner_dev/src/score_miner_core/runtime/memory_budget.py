from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover - psutil is a runtime dependency
    psutil = None


GB = 1024**3


@dataclass(frozen=True)
class MemorySnapshot:
    rss_gb: float
    cuda_allocated_gb: float | None = None
    cuda_reserved_gb: float | None = None
    cuda_free_gb: float | None = None
    cuda_total_gb: float | None = None

    @property
    def loaded_estimate_gb(self) -> float:
        cuda = self.cuda_reserved_gb
        if cuda is None:
            cuda = self.cuda_allocated_gb or 0.0
        return self.rss_gb + cuda


@dataclass(frozen=True)
class MemoryBudget:
    hard_limit_gb: float = 5.0
    warning_limit_gb: float = 4.5

    def snapshot(self) -> MemorySnapshot:
        rss_gb = 0.0
        if psutil is not None:
            rss_gb = psutil.Process().memory_info().rss / GB

        cuda_allocated_gb = None
        cuda_reserved_gb = None
        cuda_free_gb = None
        cuda_total_gb = None
        torch_module = _maybe_import_torch()
        if torch_module is not None and torch_module.cuda.is_available():
            cuda_allocated_gb = torch_module.cuda.memory_allocated() / GB
            cuda_reserved_gb = torch_module.cuda.memory_reserved() / GB
            try:
                free_bytes, total_bytes = torch_module.cuda.mem_get_info()
            except Exception:
                free_bytes, total_bytes = None, None
            if free_bytes is not None and total_bytes is not None:
                cuda_free_gb = free_bytes / GB
                cuda_total_gb = total_bytes / GB

        return MemorySnapshot(
            rss_gb=rss_gb,
            cuda_allocated_gb=cuda_allocated_gb,
            cuda_reserved_gb=cuda_reserved_gb,
            cuda_free_gb=cuda_free_gb,
            cuda_total_gb=cuda_total_gb,
        )

    def status(self, snapshot: MemorySnapshot | None = None) -> dict[str, Any]:
        snap = snapshot or self.snapshot()
        estimate = snap.loaded_estimate_gb
        return {
            "rss_gb": round(snap.rss_gb, 4),
            "cuda_allocated_gb": _round_optional(snap.cuda_allocated_gb),
            "cuda_reserved_gb": _round_optional(snap.cuda_reserved_gb),
            "cuda_free_gb": _round_optional(snap.cuda_free_gb),
            "cuda_total_gb": _round_optional(snap.cuda_total_gb),
            "loaded_estimate_gb": round(estimate, 4),
            "warning_limit_gb": self.warning_limit_gb,
            "hard_limit_gb": self.hard_limit_gb,
            "is_warning": estimate >= self.warning_limit_gb,
            "is_over_limit": estimate >= self.hard_limit_gb,
        }

    def assert_within_limit(self, snapshot: MemorySnapshot | None = None) -> None:
        status = self.status(snapshot)
        if status["is_over_limit"]:
            raise MemoryError(
                f"Loaded memory estimate {status['loaded_estimate_gb']} GB exceeds "
                f"{self.hard_limit_gb} GB Chutes limit."
            )


def _maybe_import_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)
