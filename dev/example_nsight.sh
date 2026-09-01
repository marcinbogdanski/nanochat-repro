#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 nsys profile \
  --trace=cuda,nvtx,nccl \
  --cuda-trace-scope=process-tree \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-event-trace=true \
  --sample=none \
  --cpuctxsw=none \
  --wait=all \
  --force-overwrite=true \
  --output="example_nsight" \
    torchrun --standalone --nproc_per_node=2 -m dev.example_nsight
