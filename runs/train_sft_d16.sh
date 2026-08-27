#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
  -- --run=d16 \
  "$@"
