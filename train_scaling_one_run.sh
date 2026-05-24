#!/usr/bin/env bash
set -euo pipefail

TARGET_FLOPS="$1"
MODEL_DEPTH="$2"
DEVICE_BATCH_SIZE="$3"
TAG="scaling2_${TARGET_FLOPS}_d${MODEL_DEPTH}"
RUN_DIR="${HOME}/.cache/mynanochat/runs/${TAG}"

if [ -d "$RUN_DIR" ]; then
  echo "Skipping existing run: $RUN_DIR"
  exit 0
fi

# Multi GPU
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
  -- --depth="$MODEL_DEPTH" --target-flops="$TARGET_FLOPS" --device-batch-size="$DEVICE_BATCH_SIZE" \
  --eval-every=99999999 --eval-tokens=524288 --core-metric-every=99999999 --sample-every=-1 --save-every=-1 \
  --log-metrics --log-every=1 --log-wandb-every=10 \
  --run="$TAG"

# Solo GPU
# python -m scripts.base_train \
#   --depth="$MODEL_DEPTH" --target-flops="$TARGET_FLOPS" --device-batch-size="$DEVICE_BATCH_SIZE" \
#   --eval-every=99999999 --eval-tokens=524288 --core-metric-every=99999999 --sample-every=-1 --save-every=-1 \
#   --log-metrics --log-every=1 --log-wandb-every=10 \
#   --run="$TAG"
