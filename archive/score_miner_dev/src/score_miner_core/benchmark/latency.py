from __future__ import annotations

from statistics import mean, median


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(latencies_ms),
        "mean_ms": round(float(mean(latencies_ms)), 4),
        "median_ms": round(float(median(latencies_ms)), 4),
        "p50_ms": round(percentile(latencies_ms, 50), 4),
        "p95_ms": round(percentile(latencies_ms, 95), 4),
        "p99_ms": round(percentile(latencies_ms, 99), 4),
        "min_ms": round(float(min(latencies_ms)), 4),
        "max_ms": round(float(max(latencies_ms)), 4),
    }
