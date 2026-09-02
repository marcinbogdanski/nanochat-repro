#!/usr/bin/env bash
set -euo pipefail

# Part 0: manually install Nsight Systems from the NVIDIA website
# nsys --version

# Part 1: profile the model and save the trace to 'example_nsight.nsys-rep'
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
    torchrun --standalone --nproc_per_node=2 -m dev.example_nsight_part1_profile

# Part 2: post-process 'example_nsight.nsys-rep' to inject GPU-side phase spans
# uv run python3 dev/example_nsight_part2_postprocess.py example_nsight.nsys-rep example_nsight_gpu_spans.nsys-rep
