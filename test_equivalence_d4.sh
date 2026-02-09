#!/usr/bin/env bash
set -euo pipefail

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
  -- --depth=4 --device-batch-size=8 \
  --eval-every=10 --core-metric-every=10 --sample-every=10 --save-every=10 --num-iterations=100 \
  --log-every=1 --deterministic
#  --run=d4m

# To check checkpoints:
# md5sum ~/projects/my-nanochat/models/model_000010.pt ~/.cache/nanochat/base_checkpoints/d4/model_000010.pt
