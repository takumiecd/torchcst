#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: MNIST_ROOT=/path/to/MNIST/raw $0 SEED [SEED ...]" >&2
  exit 2
fi

: "${MNIST_ROOT:?set MNIST_ROOT to the directory containing MNIST IDX files}"

for seed in "$@"; do
  echo "START_SEED=${seed} $(date -Iseconds)"
  PYTHONPATH=src python -u experiments/mnist_current_api.py \
    --device cuda \
    --seed "$seed" \
    --atoms 64 \
    --steps 128 \
    --batch-size 128 \
    --train-size 8192 \
    --test-size 2000 \
    --trust-radius 0.25 \
    --solver-starts 4 \
    --solver-max-iter 80 \
    --data "$MNIST_ROOT" \
    --out "output/mnist_current_api_k64_seed${seed}_a100.json"
  echo "END_SEED=${seed} $(date -Iseconds)"
done
