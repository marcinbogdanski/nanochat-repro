#!/usr/bin/env bash
set -euo pipefail

python -m scripts.base_train \
  --depth=4 --total-batch-size=262144 --device-batch-size=8 \
  --eval-every=10 --eval-tokens=524288 --core-metric-every=0 --sample-every=0 --save-every=10 --num-iterations=4 \
  --log-every=1
