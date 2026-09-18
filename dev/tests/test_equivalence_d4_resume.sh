#!/usr/bin/env bash
set -euo pipefail
run_path="${NANOREPRO_BASE_PATH:-$HOME/.cache/nanorepro}/runs/test_equivalence_d4_resume"

# Remove run path
if [ -d "$run_path" ]; then
    rm -f "$run_path"/*.pt "$run_path"/*.json "$run_path"/*.jsonl "$run_path"/git_diff_*.patch "$run_path"/STOP_NEXT
    rmdir "$run_path"
fi

CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
  -- --run=test_equivalence_d4_resume --depth=4 --total-batch-size=262144 --device-batch-size=8 --no-fa --fp8=false \
  --eval-every=99999999 --eval-tokens=524288 --core-metric-every=-1 --sample-every=-1 --save-every=2 --num-iterations=4 \
  --deterministic

rm "$run_path/meta_000004.json" "$run_path/model_000004.pt" \
  "$run_path/optim_000004_rank0.pt" "$run_path/optim_000004_rank1.pt" \
  "$run_path/dataloader_000004_rank0.pt" "$run_path/dataloader_000004_rank1.pt"

CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
  -- --run=test_equivalence_d4_resume --depth=4 --total-batch-size=262144 --device-batch-size=8 --no-fa --fp8=false \
  --eval-every=99999999 --eval-tokens=524288 --core-metric-every=-1 --sample-every=-1 --save-every=2 --num-iterations=4 \
  --deterministic --resume
