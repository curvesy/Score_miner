# Score Miner

Thin TurboVision deploy package. `miner.py` imports `score_miner_core` from the
wheel installed by `chute_config.yml`.

Current state: Phase 1 scaffold. The runtime is schema-safe and memory-budgeted.
Detector interfaces exist for RF-DETR, DEIMv2, and D-FINE, but the adapters are
not wired to model libraries yet.
