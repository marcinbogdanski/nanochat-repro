#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.chat_sft \
  --no-fa3 \
  --eval-every=99999999 --eval-tokens=524288 --num-iterations=4 \
  --log-metrics --deterministic \
  "$@"
