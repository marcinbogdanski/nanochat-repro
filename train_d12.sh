#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=1,3 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
  -- --depth=12 --device-batch-size=8 \
  --eval-every=250 --core-metric-every=250 --sample-every=250 \
  "$@"
