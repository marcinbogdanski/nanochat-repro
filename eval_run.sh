#!/usr/bin/env bash
set -euo pipefail

# Reproduce the base_train.py overrides without editing the script.
# Original run: CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc_per_node=2 -m scripts.base_train

TOTAL_BATCH_SIZE=$((524288/64))
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
  torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
    --num-layers=10 \
    --block-size=1024 \
    --max-steps=10 \
    --micro-batch=2 \
    --total-batch-size="${TOTAL_BATCH_SIZE}" \
    --eval-every=-1

