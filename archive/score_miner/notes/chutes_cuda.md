# Chutes CUDA Config

Date: 2026-05-19

Primary config:

```text
score_miner/chute_config.yml
```

uses:

```text
torch==2.11.0
torchvision==0.26.0
CUDA 12.8 wheels
rfdetr --no-deps
```

Reason:

- TurboVision's open-source Chute template checks for CUDA `>= 12.8`.
- Chutes newer examples use CUDA 12.8 backends for some GPU workloads.
- Installing `rfdetr --no-deps` prevents it from silently replacing the pinned Torch build.

Fallback config:

```text
score_miner/chute_config_cu126.yml
```

uses CUDA 12.6 PyTorch wheels. Use this only if the Chutes build/runtime reports CUDA 12.8 incompatibility with the selected GPU/base image.

Local warning:

If local runs show `NVIDIA driver too old`, do not judge Chutes latency from that machine. Treat local RF-DETR runs as schema/box-count checks only.
