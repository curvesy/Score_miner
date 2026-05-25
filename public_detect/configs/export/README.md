# Export Configs

Planned export targets:

```text
ONNX FP16 where supported
minimal HF repo layout
full repo size gate <= 30MB
```

Do not keep both `.pt` and `.onnx` in the final public Detect HF repo unless the combined revision still fits under the cap and the deployed miner needs both.

