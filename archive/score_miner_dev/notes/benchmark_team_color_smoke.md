# Benchmark Results

```json
{
  "batch_size": 1,
  "batches": 8,
  "boxes_per_frame_max": 8,
  "boxes_per_frame_mean": 5.25,
  "boxes_total": 42,
  "detector": "rfdetr_m",
  "frames_processed": 8,
  "latency": {
    "count": 8.0,
    "max_ms": 463.3688,
    "mean_ms": 292.9006,
    "median_ms": 272.6932,
    "min_ms": 246.2484,
    "p50_ms": 272.6932,
    "p95_ms": 399.516,
    "p99_ms": 450.5982
  },
  "memory_after_load": {
    "cuda_allocated_gb": null,
    "cuda_free_gb": null,
    "cuda_reserved_gb": null,
    "cuda_total_gb": null,
    "hard_limit_gb": 5.0,
    "is_over_limit": false,
    "is_warning": false,
    "loaded_estimate_gb": 1.506,
    "rss_gb": 1.506,
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
    "loaded_estimate_gb": 1.5381,
    "rss_gb": 1.5381,
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
    "loaded_estimate_gb": 0.1385,
    "rss_gb": 0.1385,
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
  "threshold": 0.75,
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
