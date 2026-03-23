# My NanoChat

Reproduction of Karpathy nanochat. All credit to the Great Sensei!

# Run

```bash
uv sync
uv run python -m scripts.download_dataset -n 10
uv run python -m scripts.download_eval_bundle
uv run python -m scripts.train_tokenizer
python -m scripts.base_train
```

or

```bash
torchrun --standalone --nproc_per_node=2 -m scripts.base_train
```

