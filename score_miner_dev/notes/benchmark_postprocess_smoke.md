# Benchmark Results

```json
{
  "batch_size": 4,
  "batches": 4,
  "boxes_per_frame_max": 0,
  "boxes_per_frame_mean": 0.0,
  "boxes_total": 0,
  "detector": "empty",
  "frames_processed": 16,
  "latency": {
    "count": 4.0,
    "max_ms": 0.3341,
    "mean_ms": 0.1821,
    "median_ms": 0.1407,
    "min_ms": 0.1129,
    "p50_ms": 0.1407,
    "p95_ms": 0.3082,
    "p99_ms": 0.3289
  },
  "memory_after_load": {
    "cuda_allocated_gb": null,
    "cuda_free_gb": null,
    "cuda_reserved_gb": null,
    "cuda_total_gb": null,
    "hard_limit_gb": 5.0,
    "is_over_limit": false,
    "is_warning": false,
    "loaded_estimate_gb": 0.5896,
    "rss_gb": 0.5896,
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
    "loaded_estimate_gb": 0.6543,
    "rss_gb": 0.6543,
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
    "loaded_estimate_gb": 0.1394,
    "rss_gb": 0.1394,
    "warning_limit_gb": 4.5
  },
  "n_keypoints": 32,
  "optimize_for_inference": true,
  "schema_check": {
    "errors": [],
    "expected_frame_count": 16,
    "frame_count": 16,
    "valid": true
  },
  "threshold": 0.75,
  "video": {
    "end": 16,
    "fps": 25.0,
    "height": 540,
    "path": "../turbovision/tests/test_data/videos/example_football.mp4",
    "sampled_frames": 16,
    "start": 0,
    "stride": 1,
    "total_frames": 750,
    "width": 960
  }
}
```
