#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python -m scripts.base_train \
  --depth=12 --device-batch-size=8 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
  "$@"
