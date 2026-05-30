# Benchmark Results

```json
{
  "batch_size": 1,
  "batches": 32,
  "boxes_per_frame_max": 9,
  "boxes_per_frame_mean": 6.5938,
  "boxes_total": 211,
  "detector": "rfdetr_m",
  "frames_processed": 32,
  "latency": {
    "count": 32.0,
    "max_ms": 663.212,
    "mean_ms": 343.2653,
    "median_ms": 313.4954,
    "min_ms": 258.2601,
    "p50_ms": 313.4954,
    "p95_ms": 465.2923,
    "p99_ms": 613.3327
  },
  "memory_after_load": {
    "cuda_allocated_gb": null,
    "cuda_free_gb": null,
    "cuda_reserved_gb": null,
    "cuda_total_gb": null,
    "hard_limit_gb": 5.0,
    "is_over_limit": false,
    "is_warning": false,
    "loaded_estimate_gb": 1.5073,
    "rss_gb": 1.5073,
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
    "loaded_estimate_gb": 1.5414,
    "rss_gb": 1.5414,
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
    "loaded_estimate_gb": 0.1383,
    "rss_gb": 0.1383,
    "warning_limit_gb": 4.5
  },
  "n_keypoints": 32,
  "optimize_for_inference": true,
  "schema_check": {
    "errors": [],
    "expected_frame_count": 32,
    "frame_count": 32,
    "valid": true
  },
  "threshold": 0.35,
  "video": {
    "end": 32,
    "fps": 25.0,
    "height": 540,
    "path": "../turbovision/tests/test_data/videos/example_football.mp4",
    "sampled_frames": 32,
    "start": 0,
    "stride": 1,
    "total_frames": 750,
    "width": 960
  }
}
```
