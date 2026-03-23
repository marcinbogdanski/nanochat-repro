#!/usr/bin/env bash
set -euo pipefail

python -m scripts.base_train \
  --depth=12 --device-batch-size=16 \
  --eval-every=0 --eval-tokens=524288 --core-metric-every=0 --sample-every=0 --save-every=0 --num-iterations=20 \
  --log-every=1
