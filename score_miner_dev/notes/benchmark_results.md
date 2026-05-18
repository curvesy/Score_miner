# Benchmark Results

```json
{
  "batch_size": 4,
  "batches": 4,
  "frames_processed": 16,
  "latency": {
    "count": 4.0,
    "max_ms": 0.7765,
    "mean_ms": 0.457,
    "median_ms": 0.3717,
    "min_ms": 0.3082,
    "p50_ms": 0.3717,
    "p95_ms": 0.7187,
    "p99_ms": 0.7649
  },
  "memory_after_load": {
    "cuda_allocated_gb": null,
    "cuda_free_gb": null,
    "cuda_reserved_gb": null,
    "cuda_total_gb": null,
    "hard_limit_gb": 5.0,
    "is_over_limit": false,
    "is_warning": false,
    "loaded_estimate_gb": 0.5946,
    "rss_gb": 0.5946,
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
    "loaded_estimate_gb": 0.6592,
    "rss_gb": 0.6592,
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
  "schema_check": {
    "errors": [],
    "expected_frame_count": 16,
    "frame_count": 16,
    "valid": true
  },
  "video": {
    "end": 16,
    "fps": 25.0,
    "height": 540,
    "path": "turbovision/tests/test_data/videos/example_football.mp4",
    "sampled_frames": 16,
    "start": 0,
    "stride": 1,
    "total_frames": 750,
    "width": 960
  }
}
```
