#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
  -- --depth=12 --device-batch-size=16 \
  --core-metric-every=99999999 --sample-every=-1 --save-every=99999999 \
  --log-metrics --log-every=1 \
  "$@"
