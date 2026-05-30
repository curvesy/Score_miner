# Memory Budget

Date: 2026-05-19

The TurboVision Chute template has:

```text
MAX_LOADED_MODEL_SIZE_GB = 5.0
```

Treat:

```text
< 4.5 GB: acceptable
4.5-5.0 GB: warning
>= 5.0 GB: failure
```

## What Counts

Track:

```text
CPU RSS
CUDA allocated
CUDA reserved
loaded estimate = RSS + CUDA reserved when CUDA is available
```

The first smoke miner must record memory:

```text
before model load
after model load
after one predict_batch
after repeated predict_batch
```

## Practical Implication

Do not load RF-DETR-L, DEIMv2-L, D-FINE-L, YOLO keypoints, a ball refiner, ReID, and jersey OCR together until measured. The first deploy should prove the package path and memory guard with RF-DETR-M or an empty runtime, then add candidates one at a time.
