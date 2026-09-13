#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="" python -m scripts.base_train \
  --depth=4 --total-batch-size=2048 --device-batch-size=1 --no-fa \
  --eval-every=99999999 --eval-tokens=2048 --core-metric-every=-1 --sample-every=-1 --save-every=10 --num-iterations=4 \
  --compute-dtype=fp32 \
  --log-metrics --deterministic \
  "$@"
