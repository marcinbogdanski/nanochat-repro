#!/usr/bin/env bash
set -euo pipefail

TARGET_FLOPS="$1"
MODEL_DEPTH="$2"
DEVICE_BATCH_SIZE="$3"
TAG="scaling_${TARGET_FLOPS}_d${MODEL_DEPTH}"

# Multi GPU
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
  -- --depth="$MODEL_DEPTH" --target-flops="$TARGET_FLOPS" --device-batch-size="$DEVICE_BATCH_SIZE" \
  --eval-every=-1 --eval-tokens=524288 --core-metric-every=99999999 --sample-every=-1 --save-every=-1 \
  --run="$TAG"

# Solo GPU
# python -m scripts.base_train \
#   --depth="$MODEL_DEPTH" --target-flops="$TARGET_FLOPS" --device-batch-size="$DEVICE_BATCH_SIZE" \
#   --eval-every=-1 --eval-tokens=524288 --core-metric-every=99999999 --sample-every=-1 --save-every=-1 \
#   --run="$TAG"
