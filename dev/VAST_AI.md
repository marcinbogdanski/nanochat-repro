# VastAI Setup

SSH to the instance, something like:

```bash
# Copy local repo to remote
rsync -av --delete --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude '.pytest_cache/' -e "ssh -i ~/.ssh/xxxxxxxx -p 4932" /home/user/Projects/the-nanochat/nanochat-repro/ root@xx.xxx.xx.xxx:/workspace/nanochat-repro/
# SSH to remote
ssh -i ~/.ssh/xxxxxxxx -p 4932 root@xx.xxx.xx.xxx -L 8080:localhost:8080
```

Setup instance

```bash
touch ~/.no_auto_tmux
nano ~/.bashrc              # comment line that activates conda
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Repo setup

```bash
uv sync
uv run python -m scripts.download_dataset -n 10
uv run python -m scripts.download_eval_bundle
uv run python -m scripts.train_tokenizer
```

Test runs

```bash
# WandB Login
uv run wandb login

# nanochat-repro
uv run python -m scripts.base_train --depth=12 --device-batch-size=32 --log-wandb-every=1 --eval-every=-1 --eval-tokens=524288 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --num-iterations=20 --run=h100m

# nanochat
uv run --extra gpu python -m scripts.base_train --depth=12 --device-batch-size=32 --eval-every=-1 --eval-tokens=524288 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --num-iterations=20 --fp8 --run=h100k
```
