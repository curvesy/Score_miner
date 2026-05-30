# After training — full no-miss checklist

Do steps 1-7 ON THE GPU (env + images live there), then pull back + destroy.
Laptop vars: KEY=~/Downloads/temporary.pem  LOCAL=.../public_detect  GPU=ubuntu@185.216.22.184

Stopping training early loses NOTHING — `best.pt` is saved every time val improves.
Ctrl-C is safe; best.pt (strongest) and last.pt (latest) are both on disk.

## ☁️ 1. Stop training
Ctrl-C in the training terminal.

## ☁️ 2. Honest score (Model A, clean holdout) — the trustworthy number
cd ~/Score_miner/public_detect && source .venv/bin/activate
python scripts/score_winner_style.py \
  --model runs/beverage/yolo26s_iter2-2/weights/best.pt \
  --data data/yolo_candidates/beverage_holdout_v2/data.yaml --profile drink
# baseline was 0.8147 (beats winner ~0.74). Model B is >= this but can't be clean-tested.

## ☁️ 3. (Optional) Tune thresholds to maximize score
python scripts/score_threshold_sweep.py \
  --model runs/beverage/yolo26s_iter2-2/weights/best.pt \
  --data data/yolo_candidates/beverage_holdout_v2/data.yaml \
  --name modelA_sweep --imgsz 1280 --per-class
# if it beats conf 0.60/0.45/0.50, use the winners in step 5 --conf/--rescue.

## ☁️ 4. Export Model B -> FP16 ONNX (must be FP16, <30MB)
python scripts/export_yolo_onnx.py \
  --model runs/beverage/yolo26s_modelB/weights/best.pt \
  --output-dir deploy/modelB_onnx --imgsz 1280 --half

## ☁️ 5. Build deploy repo (miner.py + weights.onnx + chute_config.yml)
python scripts/build_deploy_repo.py \
  --weights deploy/modelB_onnx/weights.onnx \
  --element-config configs/elements/beverage.yaml \
  --output-dir deploy/manak0_Detect-beverage_modelB \
  --input-size 1280 --conf 0.60,0.45,0.50 --rescue 0.0,0.0,0.20

## ☁️ 6. SMOKE TEST the deploy repo -- DO NOT SKIP (prevents deregistration)
python scripts/smoke_deploy_miner.py \
  --repo deploy/manak0_Detect-beverage_modelB \
  --images data/yolo_candidates/beverage_holdout_v2/images/train --limit 5
# must print cup/bottle/can detections, no errors.

## ☁️ 7. Size gate
python scripts/check_repo_size.py deploy/manak0_Detect-beverage_modelB --max-mb 30

## 🖥️ 8. Pull everything back to laptop (BEFORE destroying box)
cd "$LOCAL"
rsync -avP -e "ssh -i $KEY" $GPU:~/Score_miner/public_detect/runs/beverage/yolo26s_modelB/ runs/beverage/yolo26s_modelB/
rsync -avP -e "ssh -i $KEY" $GPU:~/Score_miner/public_detect/runs/beverage/yolo26s_iter2-2/ runs/beverage/yolo26s_iter2-2/
rsync -avP -e "ssh -i $KEY" $GPU:~/Score_miner/public_detect/deploy/ deploy/

## 🖥️ 9. Verify then DESTROY the A6000
ls -lh deploy/manak0_Detect-beverage_modelB/   # weights.onnx + miner.py + chute_config.yml
# then terminate the box in the provider dashboard.

## 🖥️ 10. Push code to GitHub (when internet back) -- code+config only, weights are gitignored
cd /home/sina/projects/validator_improve/score_miner_project && git push -u origin main

## 🚀 11. TOMORROW -- register on SN44
# 1) push deploy/manak0_Detect-beverage_modelB/ to a HuggingFace model repo
# 2) deploy on Chutes -> chute_id
# 3) btcli subnet register --netuid 44 (costs TAO)
# 4) commit miner: hotkey + manak0/Detect-beverage-detect + hf-repo + hf-revision + chute_id
# full steps in Importan_claudesad.md
