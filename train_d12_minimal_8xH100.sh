#!/usr/bin/env bash
set -euo pipefail

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=12 --device-batch-size=32 --run=h100m
