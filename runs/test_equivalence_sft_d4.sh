#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft \
  --no-fa3 --total-batch-size=32768 \
  --eval-every=99999999 --eval-tokens=524288 --num-iterations=4 \
  --log-metrics --deterministic \
  "$@"
