#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python -m scripts.base_train \
  --depth=12 --device-batch-size=8 \
  --eval-every=99999999 --eval-tokens=524288 --core-metric-every=99999999 --sample-every=99999999 --save-every=99999999 --num-iterations=20 \
  "$@"
