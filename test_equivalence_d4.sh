#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
  -- --depth=4 --total-batch-size=262144 --device-batch-size=8 \
  --eval-every=10 --eval-tokens=524288 --core-metric-every=0 --sample-every=0 --save-every=10 --num-iterations=4 \
  --log-every=1 --deterministic --window-pattern=SSSL
#  --run=d4m

# To check checkpoints:
# md5sum ~/projects/my-nanochat/models/model_000010.pt ~/.cache/nanochat/base_checkpoints/d4/model_000010.pt
