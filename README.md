# My NanoChat

Reproduction of Karpathy nanochat. All credit to the Great Sensei!

# Run

```bash
uv sync
uv run python -m scripts.download_dataset -n 10
uv run python -m scripts.download_eval_bundle
uv run python -m scripts.train_tokenizer
uv run ./test_speed_d12_solo.sh
```

or

```bash
torchrun --standalone --nproc_per_node=2 -m scripts.base_train
```

## On VastAI

If running on vast.ai:

SSH to the instance, something like:

```bash
# Copy local repo to remote
rsync -av --delete --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude '.pytest_cache/' -e "ssh -i ~/.ssh/mb-vastai-cVxX -p 4932" /home/user/Projects/the-nanochat/my-nanochat/ root@20.119.175.17:/workspace/my-nanochat/
# SSH to remote
ssh -i ~/.ssh/mb-vastai-cVxX -p 4932 root@20.119.175.17 -L 8080:localhost:8080
```

Setup instance

```bash
touch ~/.no_auto_tmux
nano ~/.bashrc              # comment line that activates conda
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Repo setup here, uv sync, dataset download, train tokenizer

Test runs

```bash
# WandB Login
uv run wandb login

# my-nanochat
uv run python -m scripts.base_train --depth=12 --device-batch-size=32 --log-wandb-every=1 --eval-every=-1 --eval-tokens=524288 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --num-iterations=20 --run=h100m

# nanochat
uv run --extra gpu python -m scripts.base_train --depth=12 --device-batch-size=32 --eval-every=-1 --eval-tokens=524288 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --num-iterations=20 --fp8 --run=h100k
```