#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.base_train \
  --depth=4 --total-batch-size=262144 --device-batch-size=8 --no-fa3 --fp8=false \
  --eval-every=99999999 --eval-tokens=524288 --core-metric-every=-1 --sample-every=99999999 --save-every=10 --num-iterations=4 \
  --log-metrics --deterministic \
  "$@"
