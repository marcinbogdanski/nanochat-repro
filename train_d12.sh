#!/usr/bin/env bash
set -euo pipefail

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
  -- --depth=12 --device-batch-size=16 \
  --eval-every=250 --core-metric-every=250 --sample-every=250 \
  --run=d12m
