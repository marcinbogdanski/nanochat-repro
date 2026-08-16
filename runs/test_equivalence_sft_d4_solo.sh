#!/usr/bin/env bash
set -euo pipefail

# Use --total-batch-size=16384 to pin grad_accum=1, see PR #816 on Nanochat side
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.chat_sft \
  --no-fa3 --total-batch-size=16384 \
  --eval-every=99999999 --eval-tokens=524288 --chatcore-every=-1 --sample-every=-1 --num-iterations=4 \
  --log-metrics --deterministic \
  "$@"
