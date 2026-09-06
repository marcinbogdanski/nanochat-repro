#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
  -- --depth=4 --total-batch-size=262144 --device-batch-size=8 --no-fa3 \
  --eval-every=99999999 --eval-tokens=524288 --core-metric-every=-1 --sample-every=-1 --save-every=10 --num-iterations=4 \
  --log-metrics --deterministic --muon-params-per-bucket=-1 \
  "$@"
