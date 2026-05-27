#!/usr/bin/env bash
set -euo pipefail

# Andrej Params
# FLOPS_BUDGETS=(
#   1e17     | 
#   2.15e17  | 4x3090 d18  0:20h    *6 = 2h
#   4.64e17  | 4x3090 d18  0:42h    *6 = 4h12m
#   1e18     | 4x3090 d18  1:30h    *6 = 9h
#   2.15e18  | 4x3090 d18  3:20h
#   4.64e18  | 4x3090 d18  7:00h
#   1e19     | 4x3090 d18 15:30h
# )
# DEPTHS=(10 12 14 16 18 20)

# Micro Batch Size on 1x3090
# d10 -> 16
# ...
# d14 -> 16
# d15 -> 8
# ...
# d18 -> 8
# d20 -> 4

# Micro Batch Size on H100
# d10-d18: 32
# d20-d26: 16
# d28+:     8

FLOPS_BUDGETS=(6e18 3e18 1e18)
DEPTHS=(10 12 14 16 18 20)

for TARGET_FLOPS in "${FLOPS_BUDGETS[@]}"; do
    for MODEL_DEPTH in "${DEPTHS[@]}"; do
        if [ $MODEL_DEPTH -ge 28 ]; then
            DEVICE_BATCH_SIZE=8
        elif [ $MODEL_DEPTH -ge 20 ]; then
            DEVICE_BATCH_SIZE=16
        else
            DEVICE_BATCH_SIZE=32
        fi

        echo "=================================================================="
        echo "FLOPs: $TARGET_FLOPS, Depth: $MODEL_DEPTH, Device Batch Size: $DEVICE_BATCH_SIZE"
        echo "=================================================================="
        ./train_scaling_one_run.sh "$TARGET_FLOPS" "$MODEL_DEPTH" "$DEVICE_BATCH_SIZE"
    done
done
