#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
  -- --depth=16 --device-batch-size=8 \
  --save-every=250 \
  --log-metrics --log-every=1 \
  --run=d16 \
  "$@"
