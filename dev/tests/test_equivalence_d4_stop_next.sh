#!/usr/bin/env bash
set -euo pipefail

run_path="${NANOREPRO_BASE_PATH:-$HOME/.cache/nanorepro}/runs/test_equivalence_d4_stop_next"

# Remove run path
if [ -d "$run_path" ]; then
    rm -f "$run_path"/*.pt "$run_path"/*.json "$run_path"/*.jsonl "$run_path"/git_diff_*.patch "$run_path"/STOP_NEXT
    rmdir "$run_path"
fi

# Start training in the background.
CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
  -- --run=test_equivalence_d4_stop_next --depth=4 --total-batch-size=262144 --device-batch-size=8 --no-fa --fp8=false \
  --eval-every=99999999 --eval-tokens=524288 --core-metric-every=-1 --sample-every=-1 --save-every=2 --num-iterations=4 \
  --log-metrics --deterministic &
train_pid=$!

# Wait for initial evaluation, so startup has finished clearing stop files.
until rg -q '"event": "bpb_eval", "step": 0' "$run_path/train_log_rank0.jsonl" 2>/dev/null; do
  if ! kill -0 "$train_pid" 2>/dev/null; then
    wait "$train_pid"
    exit 1
  fi
  sleep 0.2
done
touch "$run_path/STOP_NEXT"

# Wait for the checkpoint and graceful exit, then resume training.
wait "$train_pid"
CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
  -- --run=test_equivalence_d4_stop_next --depth=4 --total-batch-size=262144 --device-batch-size=8 --no-fa --fp8=false \
  --eval-every=99999999 --eval-tokens=524288 --core-metric-every=-1 --sample-every=-1 --save-every=2 --num-iterations=4 \
  --log-metrics --deterministic --resume
