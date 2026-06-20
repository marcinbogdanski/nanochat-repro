# Nanochat Repro

This repo is a from-scratch, by-hand reproduction of the pretraining stage of Andrej Karpathy [nanochat](https://github.com/karpathy/nanochat). It is built to understand training of LLMs from the ground up. This repo includes training scripts, GPT model, distributed AdamW/Muon, FP8 and some other tricks to bring performance on par with reference nanochat. I also reproduced scaling laws experiments.

Two extensions beyond original nanochat include:

**Logging Metrics**: To explore deeper training dynamics, with `--log-metrics` training run will generate additional local logs with data required to later construct RMS/norms plots of post-block activations, gradients and param updates, etc.

**Deterministic Validation**: Runs can be made bit-for-bit deterministic by using `--deterministic` flag. With small patch to Andrej nanochat, it is possible to match some of the runs bit-for-bit with Andrej nanochat. This was the main correctness test when developing this repo.

To more deeply internalize core concepts I wrote most of the code by hand. I used AI agents as educational resource and for code review, but not to edit code. In similar spirit I used `nanochat` for learning, and tried not to overuse it during coding.

I would like to deeply thank Andrej and everyone who supported him in building original `nanochat`. In my opinion, it is the best resource currently available for learning LLM training.

## Quick Run

```bash
uv sync
uv run python -m scripts.download_dataset -n 10
uv run python -m scripts.download_eval_bundle
uv run python -m scripts.train_tokenizer
uv run ./runs/train_d12.sh
```

The `-n 10` is good for quick test. Longest scaling run requires approx 230 shards. Inspect `train_d12.sh` to ensure correct values for `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` param.

