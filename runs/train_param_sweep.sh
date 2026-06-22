#!/usr/bin/env bash
set -euo pipefail

# Matrix LR sweep
for LR in 010 015 020 025 030 035; do
  OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
    -- --depth=12 --target-param-data-ratio=12 --device-batch-size=16 \
    --eval-every=250 --core-metric-every=99999999 --sample-every=-1 --save-every=99999999 \
    --log-every=10 --log-wandb-every=10 \
    --matrix-lr=0.$LR --run=sweep_d12_mlr_$LR
done

# Embedding LR sweep
for LR in 20 30 45; do
  OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
    -- --depth=12 --target-param-data-ratio=12 --device-batch-size=16 \
    --eval-every=250 --core-metric-every=99999999 --sample-every=-1 --save-every=99999999 \
    --log-every=10 --log-wandb-every=10 \
    --embedding-lr=0.$LR --run=sweep_d12_embd_lr_$LR
done

# Warmdown Ratio sweep
for WR in 50 65 80; do
  OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
    -- --depth=12 --target-param-data-ratio=12 --device-batch-size=16 \
    --eval-every=250 --core-metric-every=99999999 --sample-every=-1 --save-every=99999999 \
    --log-every=10 --log-wandb-every=10 \
    --warmdown-ratio=0.$WR --run=sweep_d12_warmdown_$WR
done

# Batch Size sweep
for BS in 262144 524288 1048576; do
  OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
    -- --depth=12 --target-param-data-ratio=12 --device-batch-size=16 \
    --eval-every=250 --core-metric-every=99999999 --sample-every=-1 --save-every=99999999 \
    --log-every=10 --log-wandb-every=10 \
    --total-batch-size=$BS --run=sweep_d12_batch_size_$BS
done

# Weight Decay sweep
for WD in 20 28 40; do
  OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
    -- --depth=12 --target-param-data-ratio=12 --device-batch-size=16 \
    --eval-every=250 --core-metric-every=99999999 --sample-every=-1 --save-every=99999999 \
    --log-every=10 --log-wandb-every=10 \
    --weight-decay=0.$WD --run=sweep_d12_weight_decay_$WD
done
