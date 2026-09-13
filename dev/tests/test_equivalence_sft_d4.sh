#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft \
  -- --total-batch-size=32768 --no-fa3 --fp8=false \
  --eval-every=99999999 --eval-tokens=524288 --chatcore-every=-1 --sample-every=-1 --num-iterations=4 \
  --log-metrics --deterministic \
  "$@"
