#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
  -- --depth=12 --device-batch-size=8 --fa3 \
  --eval-every=20 --eval-tokens=524288 --core-metric-every=20 --sample-every=20 --save-every=0 --num-iterations=20 \
  --log-every=1 \
  "$@"
