#!/usr/bin/env bash
set -euo pipefail
data=${1:?MNIST raw directory}
out=${2:?output directory}
mkdir -p "$out"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
for seed in 17 29 43; do
    modes='raw alpha_diagonal'
    if [[ "$seed" == 29 ]]; then modes='alpha_diagonal raw'; fi
    for mode in $modes; do
        timeout 900 python -m experiments.local_tangent_learning \
            --data "$data" --output "$out/$mode-s$seed.json" --approximation full \
            --parameter-rms "$mode" --seed "$seed" --steps 512 --eval-every 8 \
            > "$out/$mode-s$seed.log" 2>&1
    done
done
