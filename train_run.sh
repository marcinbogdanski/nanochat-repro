#!/usr/bin/env bash
set -euo pipefail

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
  -- --depth=4 --device-batch-size=8 \
  --eval-every=100 --core-metric-every=100 --sample-every=100 --save-every=100 \
  --log-every=1 --deterministic --run=d4m


# Run the training script with specified parameters
# TOTAL_BATCH_SIZE=$((524288/16))
# CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
#   torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
#     --num-layers=12 \
#     --block-size=2048 \
#     --max-steps=100 \
#     --micro-batch=4 \
#     --total-batch-size="${TOTAL_BATCH_SIZE}" \
#     --eval-every=100 \
#     --core-metric-every=100 \
#     --generate-every=100 \
#     --save-every=999999999
