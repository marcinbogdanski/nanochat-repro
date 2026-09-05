#!/usr/bin/env bash
set -euo pipefail

# Part 1: profile the model and save the trace to 'nsight_trace.nsys-rep'
NANOREPRO_TRACE=1 CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 nsys profile \
  --trace=cuda,nvtx,nccl \
  --cuda-trace-scope=process-tree \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-event-trace=true \
  --sample=none \
  --cpuctxsw=none \
  --wait=all \
  --force-overwrite=true \
  --output="nsight_trace" \
    torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
      -- --depth=12 --device-batch-size=16 --num-iterations=5 \
      --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
      --log-every=1 \
      "$@"

# Part 2: post-process 'nsight_trace.nsys-rep' to inject GPU-side phase spans
python3 dev/nsight_postprocess.py nsight_trace.nsys-rep nsight_trace_gpu_spans.nsys-rep
