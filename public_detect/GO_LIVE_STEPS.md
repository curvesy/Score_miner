# Go-live steps (run on the LAPTOP) — beverage Detect miner, netuid 44

Two folders:
- public_detect/  = YOUR detector + deploy artifact
- turbovision/    = the `sv` CLI you run to deploy (subnet tool)

The deploy artifact you ship:
  ~/projects/validator_improve/score_miner_project/public_detect/deploy/manak0_Detect-beverage_modelB
  (weights.onnx + miner.py + miner_config.json)

SECURITY: never paste API keys in chat. If you did, revoke + regenerate them. Keys live ONLY in turbovision/.env.

------------------------------------------------------------------
## ONE-TIME SETUP (tonight)

# 1. clean a stray venv if you made one in the parent
rm -rf ~/projects/validator_improve/.venv

# 2. install the sv CLI INSIDE the turbovision repo (must cd in first!)
cd ~/projects/validator_improve/turbovision
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv sync
sv --help        # confirms install

# 3. bittensor wallet (your miner identity)
pip install bittensor-cli
btcli wallet new_coldkey --n_words 24 --wallet.name my-wallet
btcli wallet new_hotkey  --wallet.name my-wallet --n_words 24 --wallet.hotkey my-hotkey
# SAVE THE SEED PHRASES SAFELY.

# 4. .env (in turbovision/)
cp env.example .env
#   set: BITTENSOR_WALLET_COLD, BITTENSOR_WALLET_HOT, SCOREVISION_NETUID=44,
#        CHUTES_API_KEY, CHUTES_USERNAME, HUGGINGFACE_USERNAME, HUGGINGFACE_API_KEY

# 5. accounts:
#   - HuggingFace: write token (rotate the one leaked in chat!)
#   - Chutes: chutes register (done) + add ~$25 balance on chutes.ai
#   - check TAO cost:  btcli subnet burn-cost --netuid 44
#     fund your coldkey with that + buffer

------------------------------------------------------------------
## GO LIVE (tomorrow)

cd ~/projects/validator_improve/turbovision
source .venv/bin/activate

# 6. register hotkey on subnet 44 (BURNS TAO)
btcli subnet register --netuid 44 --wallet.name my-wallet --wallet.hotkey my-hotkey

# 7. clean caches from the deploy folder
rm -rf ~/projects/validator_improve/score_miner_project/public_detect/deploy/manak0_Detect-beverage_modelB/__pycache__

# 8. TEST deploy (uploads to HF + builds chute, NO on-chain commit yet)
sv deploy-os-miner \
  --model-path ~/projects/validator_improve/score_miner_project/public_detect/deploy/manak0_Detect-beverage_modelB \
  --element-id manak0/Detect-beverage-detect \
  --no-commit

# 9. REAL deploy (same, without --no-commit) — writes on-chain commitment
sv deploy-os-miner \
  --model-path ~/projects/validator_improve/score_miner_project/public_detect/deploy/manak0_Detect-beverage_modelB \
  --element-id manak0/Detect-beverage-detect

# 10. within ~10-30 min check your score:
#   https://console.scorevision.io  (element manak0/Detect-beverage-detect)

------------------------------------------------------------------
## What happens (mental model)
sv uploads model -> HuggingFace (<user>/ScoreVision @ revision)
sv builds a Chute that downloads that HF repo and serves /predict on a GPU
sv writes on-chain: {hotkey, element_id, revision, chute_id}
validators read it -> send frames to your chute -> score you -> you earn

Costs: TAO burn (one-time, register) + Chutes balance (ongoing, cheap, sleeps when idle). HF = free.
Your honest score = 0.8147; live winner = 0.7208. You enter as top scorer.
