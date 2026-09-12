#!/usr/bin/env bash
set -euo pipefail

echo "--------------------------------------------------------------------------------"
echo "                                Base Train"
echo "--------------------------------------------------------------------------------"
./dev/tests/test_equivalence_d4.sh "$@"

echo "--------------------------------------------------------------------------------"
echo "                                Train Eval"
echo "--------------------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_eval \
  -- --eval-tokens=524288

echo "--------------------------------------------------------------------------------"
echo "                                Chat SFT"
echo "--------------------------------------------------------------------------------"
./dev/tests/test_equivalence_sft_d4.sh "$@"

echo "--------------------------------------------------------------------------------"
echo "                                Eval SFT"
echo "--------------------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.chat_eval \
  -- --data-mixture=ext --eval-tokens=524288
