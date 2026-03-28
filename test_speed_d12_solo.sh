#!/usr/bin/env bash
set -euo pipefail

python -m scripts.base_train \
  --depth=12 --device-batch-size=8 --window-patter=L --fa3 --fp8 \
  --eval-every=10 --eval-tokens=524288 --core-metric-every=20 --sample-every=20 --save-every=0 --num-iterations=20 \
  --log-every=1
