# After Model A finishes — full playbook (register tomorrow)

Two models:
- **Model A** (training now): on 404 Score frames, holdout kept secret. Job = give an HONEST score.
- **Model B** (the register model): on ALL 533 Score frames. Job = max performance to deploy.

Laptop vars (set once on LAPTOP):
```bash
KEY=~/Downloads/temporary.pem
LOCAL=/home/sina/projects/validator_improve/score_miner_project/public_detect
GPU=ubuntu@185.216.22.184
```

---

## PHASE 1 — Model A done → get the honest score (on the GPU)

You know it's done when the log prints `60 epochs completed` and `Results saved to runs/beverage/yolo26s_iter2`.

```bash
cd ~/Score_miner/public_detect
source .venv/bin/activate
python scripts/score_winner_style.py \
  --model runs/beverage/yolo26s_iter2/weights/best.pt \
  --data data/yolo_candidates/beverage_holdout_v2 --profile drink
```

Read the **final number** (`0.6*map50 + 0.4*fp`):
- **≥ 0.78** → recipe works → go to PHASE 2.
- **< 0.78** → STOP. Don't waste a Model B run or TAO. Tell me the number; we tune (conf thresholds / more epochs) first.

---

## PHASE 2 — Train Model B on ALL 533 frames (only if Phase 1 ≥ 0.78)

### 2a. From your LAPTOP — ship Model B's dataset + config
```bash
rsync -avP --mkpath -e "ssh -i $KEY" \
  "$LOCAL/data/yolo/beverage_phase4_modelB_v1/" \
  $GPU:~/Score_miner/public_detect/data/yolo/beverage_phase4_modelB_v1/

rsync -avP -e "ssh -i $KEY" \
  "$LOCAL/configs/training/beverage_yolo26s_modelB.yaml" \
  $GPU:~/Score_miner/public_detect/configs/training/beverage_yolo26s_modelB.yaml
```

### 2b. On the GPU — fix path + train
```bash
cd ~/Score_miner/public_detect
sed -i "s|^path:.*|path: $(pwd)/data/yolo/beverage_phase4_modelB_v1|" \
  data/yolo/beverage_phase4_modelB_v1/data.yaml

python scripts/train_baseline.py --config configs/training/beverage_yolo26s_modelB.yaml --batch 16
```
Model B warm-starts from Model A's `best.pt` and adds the 100 holdout + 29 new frames. ~2–3 hrs.

> Note: Model B trains on everything, so there's no clean holdout left to test it on — that's expected. You already proved the recipe with Model A's score; Model B uses the same recipe with more data, so it's ≥ Model A. That's the standard pro workflow.

---

## PHASE 3 — Export the deploy weights (on the GPU)
```bash
python scripts/export_yolo_onnx.py \
  --model runs/beverage/yolo26s_modelB/weights/best.pt \
  --output-dir deploy/modelB_onnx --imgsz 1280 --half
```
Confirm ~19 MB and `passes: true` (must be FP16 to fit the 30 MB HF cap).

---

## PHASE 4 — Pull results to LAPTOP, then DESTROY the box
```bash
rsync -avP -e "ssh -i $KEY" $GPU:~/Score_miner/public_detect/runs/beverage/yolo26s_modelB/ \
  "$LOCAL/runs/beverage/yolo26s_modelB/"
rsync -avP -e "ssh -i $KEY" $GPU:~/Score_miner/public_detect/deploy/modelB_onnx/ \
  "$LOCAL/deploy/modelB_onnx/"
```
Then terminate the A6000 so it stops charging.

---

## PHASE 5 — Build deploy repo + register (tomorrow, on LAPTOP)
```bash
python scripts/build_deploy_repo.py \
  --weights deploy/modelB_onnx/weights.onnx \
  --element-config configs/elements/beverage.yaml \
  --output-dir deploy/manak0_Detect-beverage_modelB \
  --input-size 1280 --conf 0.60,0.45,0.50 --rescue 0.0,0.0,0.20
```
Then: push to HuggingFace → deploy on Chutes → `btcli subnet register --netuid 44` → commit the miner (hotkey + element_id + hf-repo + hf-revision + chute_id). Full steps in `Importan_claudesad.md`.

Don't register unless Phase 1 was ≥ ~0.55 (ideally ≥ 0.74 to beat the winner) — registering a weak model risks deregistration + lost TAO.
