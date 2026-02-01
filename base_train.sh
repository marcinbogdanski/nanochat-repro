#!/usr/bin/env bash
set -euo pipefail

TOTAL_BATCH_SIZE=$((524288/128))

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
    --num-layers=10 \
    --total-batch-size="${TOTAL_BATCH_SIZE}" \
    --micro-batch=1 \
    --block-size=1024 \
    --max-steps=10

# Equivalent nanochat run:
# CUBLAS_WORKSPACE_CONFIG=:4096:8 \
#   torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
#     --depth=10 \
#     --max_seq_len=1024 \
#     --num_iterations=10 \
#     --device_batch_size=1 \
#     --total_batch_size="${TOTAL_BATCH_SIZE}" \
#     --eval_every=-1 \
#     --core_metric_every=-1 \
#     --sample_every=-1 \
#     --save_every=-1

# To check checkpoints:
# md5sum ~/projects/my-nanochat/model_000010.pt ~/.cache/nanochat/base_checkpoints/d10/model_000010.pt
