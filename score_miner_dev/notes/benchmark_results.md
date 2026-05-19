# Benchmark Results

```json
{
  "batch_size": 4,
  "batches": 2,
  "boxes_per_frame_max": 0,
  "boxes_per_frame_mean": 0.0,
  "boxes_total": 0,
  "detector": "empty",
  "frames_processed": 8,
  "latency": {
    "count": 2.0,
    "max_ms": 0.6924,
    "mean_ms": 0.499,
    "median_ms": 0.499,
    "min_ms": 0.3055,
    "p50_ms": 0.499,
    "p95_ms": 0.6731,
    "p99_ms": 0.6886
  },
  "memory_after_load": {
    "cuda_allocated_gb": null,
    "cuda_free_gb": null,
    "cuda_reserved_gb": null,
    "cuda_total_gb": null,
    "hard_limit_gb": 5.0,
    "is_over_limit": false,
    "is_warning": false,
    "loaded_estimate_gb": 0.5929,
    "rss_gb": 0.5929,
    "warning_limit_gb": 4.5
  },
  "memory_after_predict": {
    "cuda_allocated_gb": null,
    "cuda_free_gb": null,
    "cuda_reserved_gb": null,
    "cuda_total_gb": null,
    "hard_limit_gb": 5.0,
    "is_over_limit": false,
    "is_warning": false,
    "loaded_estimate_gb": 0.6449,
    "rss_gb": 0.6449,
    "warning_limit_gb": 4.5
  },
  "memory_before": {
    "cuda_allocated_gb": null,
    "cuda_free_gb": null,
    "cuda_reserved_gb": null,
    "cuda_total_gb": null,
    "hard_limit_gb": 5.0,
    "is_over_limit": false,
    "is_warning": false,
    "loaded_estimate_gb": 0.1337,
    "rss_gb": 0.1337,
    "warning_limit_gb": 4.5
  },
  "n_keypoints": 32,
  "optimize_for_inference": true,
  "schema_check": {
    "errors": [],
    "expected_frame_count": 8,
    "frame_count": 8,
    "valid": true
  },
  "threshold": 0.35,
  "video": {
    "end": 8,
    "fps": 25.0,
    "height": 540,
    "path": "../turbovision/tests/test_data/videos/example_football.mp4",
    "sampled_frames": 8,
    "start": 0,
    "stride": 1,
    "total_frames": 750,
    "width": 960
  }
}
```
