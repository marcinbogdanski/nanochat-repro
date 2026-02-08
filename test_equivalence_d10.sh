#!/usr/bin/env bash
set -euo pipefail

# Run the training script with specified parameters
TOTAL_BATCH_SIZE=$((524288/128))
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
  torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
    --depth=10 \
    --total-batch-size="${TOTAL_BATCH_SIZE}" \
    --device-batch-size=1 \
    --block-size=1024 \
    --num-iterations=10 \
    --save-every=99999999 \
    --deterministic

# To check checkpoints:
# md5sum ~/projects/my-nanochat/models/model_000010.pt ~/.cache/nanochat/base_checkpoints/d10/model_000010.pt
