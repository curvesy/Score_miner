# iter2 on a Rental GPU — copy-paste playbook

Everything is prepped on the laptop. You only fill in the rental IP + SSH port,
then paste blocks in order. Warm-start from your epoch-104 brain (`seed_best.pt`),
train on the new 42%-Score dataset. **NOT a resume, NOT from zero.**

What ships:
- code + scripts + configs + `seed_best.pt` (your 0.78 brain)  ← one rsync
- `data/yolo/beverage_phase4_iter2_v1/` (3502 train / 219 val) ← one rsync
- `data/yolo_candidates/beverage_holdout_v2/` (100 clean frames, to probe)  ← one rsync

---

## 0) Set these once (on the rental's terminal AND your laptop terminal as needed)

On your LAPTOP, fill in the rental box details:
```bash
RENTAL=root@PUT_RENTAL_IP_HERE
PORT=PUT_SSH_PORT_HERE        # RunPod/Vast give you a port, often not 22
LOCAL=/home/sina/projects/validator_improve/score_miner_project/public_detect
```

---

## 1) SHIP IT (run on your LAPTOP)

```bash
cd "$LOCAL"

# 1a. code + seed_best.pt  (excludes the heavy stuff)
rsync -avz -e "ssh -p $PORT" \
  --exclude 'runs/' --exclude 'data/' --exclude '.git/' \
  --exclude '.uv-cache/' --exclude '__pycache__/' --exclude '*.pyc' \
  ./ "$RENTAL:~/pd/"

# 1b. the iter2 training dataset
rsync -avz -e "ssh -p $PORT" \
  data/yolo/beverage_phase4_iter2_v1/ \
  "$RENTAL:~/pd/data/yolo/beverage_phase4_iter2_v1/"

# 1c. clean holdout (to honestly probe after training)
rsync -avz -e "ssh -p $PORT" \
  data/yolo_candidates/beverage_holdout_v2/ \
  "$RENTAL:~/pd/data/yolo_candidates/beverage_holdout_v2/"
```

`seed_best.pt` sits at the project root, so block 1a carries it automatically.

---

## 2) SET UP THE RENTAL (run on the RENTAL, once)

SSH in: `ssh -p $PORT $RENTAL`

```bash
cd ~/pd

# Fix the dataset's absolute path so it points at the rental, not the laptop:
sed -i "s|^path:.*|path: $HOME/pd/data/yolo/beverage_phase4_iter2_v1|" \
  data/yolo/beverage_phase4_iter2_v1/data.yaml

# Install deps (needs Python 3.12). uv is easiest:
pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# Sanity: GPU visible?
python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `uv` gives trouble, plain pip works too:
```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -e .
```

---

## 3) DRY-RUN FIRST (run on the RENTAL) — proves the config is correct, trains nothing

```bash
cd ~/pd
python scripts/train_baseline.py \
  --config configs/training/beverage_yolo26s_iter2.yaml --dry-run
```
Read the JSON it prints. Check:
- `train_args.model` → `seed_best.pt`  (warm-start ✓)
- `train_args.data` → ...beverage_phase4_iter2_v1/data.yaml
- `resume_checkpoint` → `null`  (good — this is warm-start, NOT resume)
- `expected_best_checkpoint` → runs/beverage/yolo26s_iter2/weights/best.pt

---

## 4) TRAIN (run on the RENTAL) — the ONE command

```bash
cd ~/pd
python scripts/train_baseline.py \
  --config configs/training/beverage_yolo26s_iter2.yaml
```
~1.5–2 hrs on a 4090 for 60 epochs at imgsz 1280, batch 8.
Tip: run inside `tmux` so an SSH drop doesn't kill it:
```bash
tmux new -s t        # start session
# (paste the train command)
# detach: Ctrl-b then d   | reattach later: tmux attach -t t
```
If you hit CUDA out-of-memory, lower batch on the fly:
```bash
python scripts/train_baseline.py \
  --config configs/training/beverage_yolo26s_iter2.yaml --batch 4
```

Output lands at: `runs/beverage/yolo26s_iter2/weights/best.pt`

---

## 5) PROBE — is iter2 actually better than 0.78? (run on the RENTAL)

```bash
cd ~/pd
python scripts/score_winner_style.py \
  --model runs/beverage/yolo26s_iter2/weights/best.pt \
  --data data/yolo_candidates/beverage_holdout_v2 \
  --profile drink
```
Look at the printed `0.6*map50 + 0.4*fp` final score.
- **Higher than 0.78 → keep iter2, go to step 6.**
- Lower/equal → iter2 didn't help; keep iter1, skip to deploy with the old best.pt.

---

## 6) EXPORT FP16 ONNX (must be FP16 — FP32 is 38MB, over the 30MB cap)

```bash
cd ~/pd
python scripts/export_yolo_onnx.py \
  --model runs/beverage/yolo26s_iter2/weights/best.pt \
  --output-dir deploy/iter2_onnx \
  --imgsz 1280 --half
```
Confirm it prints ~19MB and `passes: true`.

---

## 7) PULL THE RESULTS BACK TO LAPTOP (run on your LAPTOP)

```bash
cd "$LOCAL"
# the trained weights + onnx
rsync -avz -e "ssh -p $PORT" \
  "$RENTAL:~/pd/runs/beverage/yolo26s_iter2/" \
  runs/beverage/yolo26s_iter2/
rsync -avz -e "ssh -p $PORT" \
  "$RENTAL:~/pd/deploy/iter2_onnx/" \
  deploy/iter2_onnx/
```

---

## 8) BUILD THE DEPLOY REPO (run on LAPTOP, after pulling) — winner recipe baked in

```bash
cd "$LOCAL"
python scripts/build_deploy_repo.py \
  --weights deploy/iter2_onnx/weights.onnx \
  --element-config configs/elements/beverage.yaml \
  --output-dir deploy/manak0_Detect-beverage_yolo26s_iter2 \
  --input-size 1280 --conf 0.60,0.45,0.50 --rescue 0.0,0.0,0.20
```
Then push to HuggingFace + Chutes + register on SN44 — see
`Importan_claudesad.md` (the deploy→register→iterate playbook).

---

## Cheat sheet — what the words mean
- **seed_best.pt** = your epoch-104 brain (the 0.78). Config starts FROM it.
- **warm-start** = load that brain, teach it more on new data. Weights carried over.
- **resume** = finish the same run on the same data. We are NOT doing this.
- **probe on holdout** = honest score; never train on holdout_v2.
- **FP16 ONNX** = the only format that fits the 30MB HF cap (19.4MB = winner size).
