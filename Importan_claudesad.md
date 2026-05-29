p by step. Three phases: deploy, monitor, iterate.

  PHASE 1 — When iter1 training finishes tonight

  Step 1: Probe your model

  In your terminal (training will have stopped):

  cd /home/sina/projects/validator_improve/score_miner_project/public_detect

  UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
  PYTHONPATH=src uv run python scripts/score_winner_style.py \
    --model runs/beverage/yolo26s_phase4_winner_v1_local/weights/best.pt \
    --data data/yolo_candidates/beverage_winner_proxy_v2/data.yaml \
    --imgsz 1280

  Read the manifest score line. Three branches:

  ┌─────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Probe score │                                                        What to do                                                         │
  ├─────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ ≥ 0.55      │ Go to Step 2 (deploy live) — your model is good enough to start earning                                                   │
  ├─────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 0.40–0.55   │ Run iter2 (one command, ~1.5 hr): bash scripts/iter2_rebuild_and_warmstart.sh. After it finishes, re-probe and re-decide. │
  ├─────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ < 0.40      │ Run iter2 anyway. Something off in training, but warm-start will fix most of it.                                          │
  └─────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Step 2 — IF score ≥ 0.55: build deploy artifact

  # export the .pt weights to ONNX at the validator's input size
  UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
  PYTHONPATH=src uv run python scripts/export_yolo_onnx.py \
    --model runs/beverage/yolo26s_phase4_winner_v1_local/weights/best.pt \
    --output-dir /tmp/yolo26s_export --imgsz 1280

  # bundle ONNX + winner-style miner.py + config into a deployable HF repo
  UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
  PYTHONPATH=src uv run python scripts/build_deploy_repo.py \
    --weights /tmp/yolo26s_export/weights.onnx \
    --element-config configs/elements/beverage.yaml \
    --output-dir deploy/manak0_Detect-beverage_yolo26s_v1 \
    --input-size 1280 \
    --conf 0.60,0.45,0.50 \
    --rescue 0.0,0.0,0.20

  After this you'll have:
  deploy/manak0_Detect-beverage_yolo26s_v1/
  ├── weights.onnx        ← your model
  ├── miner.py            ← winner-style inference (TTA + rescue bonus + cross-class dedup)
  ├── miner_config.json   ← inference settings
  └── size_report.json    ← confirms < 30 MB
  
  This folder IS your HuggingFace repo. You push it to HF as-is.

  ---
  PHASE 2 — Going live (1-2 hours setup, one-time)

  This is the part that uses tools OUTSIDE this project. I'll lay out the flow but you'll execute these manually because they involve wallets, money, and external accounts.

  What you need before you start

  1. Bittensor wallet with TAO (registration on SN44 costs ~real money in TAO — check current burn cost at taostats.io). Today it's typically $20–$200 USD equivalent.
  2. HuggingFace account with write access (free).
  3. Chutes account at chutes.ai (Score's GPU serving platform — also free to sign up).

  The registration flow (5 steps)

  1. Push deploy/ folder to HuggingFace
     git lfs install
     huggingface-cli login
     huggingface-cli repo create your-hf-username/beverage-v1
     cd deploy/manak0_Detect-beverage_yolo26s_v1
     git init && git lfs track "*.onnx"
     git remote add origin https://huggingface.co/your-hf-username/beverage-v1
     git add . && git commit -m "v1"
     git push -u origin main

     note your-hf-username/beverage-v1 and the commit hash

  2. Deploy to Chutes (one-time)
     - log in at chutes.ai
     - "New Chute" → point at your HF repo
     - Chutes auto-reads chute_config.yml from your repo
     - wait ~5 min for deploy
     - copy the chute_id (looks like 970ae2be-206d-540c-a1c0-2b091834e87c)

  3. Register on subnet 44 (this costs TAO)
     btcli subnet register --netuid 44 --wallet.name <yourwallet>
     - prompts for TAO burn confirmation
     - on success you get a UID (e.g., 23)

  4. Commit your miner to the subnet
     You need to tell the subnet "this hotkey serves this HF repo at this revision via this chute_id".
     The score-vision repo has a tool for this:
     git clone https://github.com/score-technologies/score-vision
     cd score-vision
     # follow their miner registration docs — usually a python command like:
     python -m score_vision.miner.commit \
       --hotkey <yourwallet> \
       --element-id manak0/Detect-beverage-detect \
       --hf-repo your-hf-username/beverage-v1 \
       --hf-revision <commit-hash> \
       --chute-id <chute_id>

  5. Wait one window (~10-30 min depending on subnet block time)
     - Validators see your commitment
     - Next scoring window picks you up
     - Check console.scorevision.io for your hotkey's score

  For the EXACT steps and the latest commands, check the score-vision repo's README:
  - github.com/score-technologies/score-vision
  - Their docs/ folder has miner setup guides

  If you get stuck on registration, the SN44 community Discord is your best resource — most miners help each other through the first-time setup.

  ---
  PHASE 3 — Once you're live (this is where the real winning happens)
  
  Day 1 (after registering): VERIFY you're scoring

  Pull your own score:

  /usr/bin/curl -sL "https://console.scorevision.io/api/v2/elements/manak0%2FDetect-beverage-detect" \
    | python3 -c "
  import sys, json
  d = json.load(sys.stdin)
  for m in (d.get('leaderboard') or []):
      if m['hotkey'].startswith('<your-hotkey-prefix>'):
          print(f\"score={m['score']:.4f} earned=\${m.get('usdEarned',0):.2f}\")
  "

  Replace <your-hotkey-prefix> with the first 8 chars of your hotkey. You should see a score within 1-2 windows after registration.

  Compare your live score to your probe score. Usually live > probe by 0.05-0.10 because SAM3 GT is more forgiving than winner-proxy.

  Day 2-3: HARVEST YOUR OWN SCORING DATA

  Now that you're live, Manako logs YOUR predictions on YOUR scored frames. Pull them:

  UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
  PYTHONPATH=src uv run python scripts/probe_via_winner.py \
    --model runs/beverage/yolo26s_phase4_winner_v1_local/weights/best.pt \
    --output-dir data/yolo_candidates/beverage_self_v1 \
    --max-evals 2000 --min-eval-score 0.30 --gt-conf 0.40

  But change the hotkey filter in the script — or, simpler, just re-run harvest_all_manako.py and the new evals (including yours) will be in the index:

  UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
  PYTHONPATH=src uv run python scripts/harvest_all_manako.py \
    --output-dir data/yolo_candidates/beverage_all_miners_v2 \
    --min-eval-score 0.30 --gt-conf 0.40 --workers 60

  The key: every challenge frame the validator has scored you on is now a known frame URL with at least 26 miners' predictions. Aggregate them. The top-3 miners' boxes per frame
  become very high-quality proxy GT (~80-90% accurate vs SAM3).

  Day 3-4: ITER3 — train better while still live

  YOU ARE STILL MINING WHILE YOU TRAIN. The deployed model keeps earning. Training happens OFFLINE on your laptop. When iter3 is better, you push it.

  # build iter3 dataset with the freshest harvest
  UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
  PYTHONPATH=src uv run python scripts/build_phase4_dataset.py \
    --element-config configs/elements/beverage.yaml \
    --output-dir data/yolo/beverage_phase4_iter3_v1 \
    --source starter:data/yolo/beverage_starter \
    --source winnerproxy:data/yolo_candidates/beverage_winner_proxy_v2::3x \
    --source allminers:data/yolo_candidates/beverage_all_miners_v2::3x \
    --source oiv7:data/yolo_candidates/beverage_oiv7_v1 \
    --source drinkwaste:data/yolo_candidates/beverage_drinkwaste_v1:600 \
    --val-fraction 0.10

  # warm-start from iter2's best.pt (faster)
  cat > configs/training/beverage_yolo26s_iter3.yaml <<EOF
  element_config: configs/elements/beverage.yaml
  data: data/yolo/beverage_phase4_iter3_v1/data.yaml
  model: runs/beverage/yolo26s_phase4_iter2_v1_local/weights/best.pt
  project: runs/beverage
  name: yolo26s_iter3
  epochs: 40
  imgsz: 1280
  batch: 2
  patience: 15
  device: 0
  workers: 4
  seed: 44
  optimizer: auto
  cos_lr: true
  close_mosaic: 5
  cache: false
  amp: true
  plots: true
  val: true
  EOF

  UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
  PYTHONPATH=src uv run python scripts/train_baseline.py \
    --config configs/training/beverage_yolo26s_iter3.yaml \
    --epochs 40 --batch 2 --imgsz 1280 --name-suffix local

  # probe iter3
  UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
  PYTHONPATH=src uv run python scripts/score_winner_style.py \
    --model runs/beverage/yolo26s_iter3_local/weights/best.pt \
    --data data/yolo_candidates/beverage_all_miners_v2/data.yaml \
    --imgsz 1280

  Day 4: PUSH NEW MODEL TO HF (rolling update)

  When iter3 probe is better than what's currently deployed:

  # export new ONNX
  UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
  PYTHONPATH=src uv run python scripts/export_yolo_onnx.py \
    --model runs/beverage/yolo26s_iter3_local/weights/best.pt \
    --output-dir /tmp/iter3_export --imgsz 1280

  # build new deploy
  UV_CACHE_DIR=/home/sina/projects/validator_improve/.uv-cache \
  PYTHONPATH=src uv run python scripts/build_deploy_repo.py \
    --weights /tmp/iter3_export/weights.onnx \
    --element-config configs/elements/beverage.yaml \
    --output-dir deploy/manak0_Detect-beverage_yolo26s_v3 \
    --input-size 1280 \
    --conf 0.60,0.45,0.50 \
    --rescue 0.0,0.0,0.20

  # push to HF (UPDATING the same repo, new commit hash)
  cd deploy/manak0_Detect-beverage_yolo26s_v3
  cp -r .git ../manak0_Detect-beverage_yolo26s_v1/  # reuse the git history from your original repo
  cd ../manak0_Detect-beverage_yolo26s_v1
  cp ../manak0_Detect-beverage_yolo26s_v3/weights.onnx .
  cp ../manak0_Detect-beverage_yolo26s_v3/miner.py .
  cp ../manak0_Detect-beverage_yolo26s_v3/miner_config.json .
  git add . && git commit -m "iter3"
  git push

  # update your commitment to the new revision
  # (use score-vision's commit tool with the new commit hash)

  The validator picks up the new revision within ~10-30 min. Your score updates on the next scoring window.

  Day 5-7: KEEP THE LOOP RUNNING

  ┌─────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬──────────────────────┐
  │ Day │                                                  What you do                                                   │     What changes     │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ 5   │ harvest fresh evals (more frames now), train iter4, push if better                                             │ live score creeps up │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ 6   │ check leaderboard ranking. If you're top-5, you're earning.                                                    │ iterate              │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ 7   │ If you've stalled around 0.65, try YOLO26m (still under 30 MB), or tune per-class conf/bonus more aggressively │ break the plateau    │
  └─────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴──────────────────────┘

  "Can I keep retraining while live?" — YES

  Critically important to understand:
  - Live mining is JUST your committed HF revision + chute serving it. Static. Validator polls it, sends frames, gets predictions.
  - Training happens entirely on your laptop, offline. Nothing on-chain about your training.
  - You push a new HF revision whenever your local probe is better. Update commitment, done.
  - The winners have been doing this for weeks. That's why their HF "stayed static for 10 days" — they had no improvement worth pushing. The moment they have one, they push.

  Bottom line: you mine WITH iter1/iter2 weights while you train iter3/iter4/iter5 offline. Money keeps coming in while you improve.

  What can go wrong + how to recover

  ┌──────────────────────────────────────┬──────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────┐
  │               Failure                │                Cause                 │                                               Fix                                               │
  ├──────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Score below baseline (0.195) for 3+  │ Wrong class order in ONNX or HF repo │ Verify class_names = ["cup","bottle","can"] in miner.py matches manifest, and HF repo is        │
  │ windows                              │  gating                              │ public, non-gated. Re-push.                                                                     │
  ├──────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Chute timing out                     │ Your inference too slow at 1280      │ Drop to input_size=1024 in chute_config and re-push. Latency cap is in the manifest             │
  │                                      │                                      │ (latency_p95_ms gate).                                                                          │
  ├──────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Deregistered (lost UID)              │ Score below baseline_theta too long  │ Re-register (costs TAO again). Don't deploy until probe ≥ 0.50.                                 │
  ├──────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Score wiggles a lot day-to-day       │ Normal — different challenges per    │ Don't chase noise. Compare 3-day moving average.                                                │
  │                                      │ window.                              │                                                                                                 │
  └──────────────────────────────────────┴──────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────┘

  Honest final summary

  You are NOT in this race to win in one shot. You're in to iterate faster than the winners.

  - They iterate roughly once every 7-10 days (based on HF revision dates).
  - You can iterate once every 1-2 days because the entire pipeline is now scripted (harvest → build → warm-start train → probe → deploy).
  - After 3-4 iterations (≈ 1 week) you'll be in the same ballpark as the top miners. After 7-10 iterations you have a real shot at winner status.

  Sleep tonight after iter1 finishes. Run the probe in the morning. If it's ≥ 0.55, deploy live and the iteration loop begins.

  Sources:
