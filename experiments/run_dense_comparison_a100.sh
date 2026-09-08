#!/usr/bin/env bash
set -euo pipefail
data_root=${1:?MNIST raw-data directory}
results_dir=${2:?Output directory}
mkdir -p "$results_dir"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
for seed in 17 29 43; do
    case "$seed" in
        17) methods='dense-adam cst-adam cst-adam-fast cst' ;;
        29) methods='cst cst-adam-fast cst-adam dense-adam' ;;
        43) methods='cst-adam cst dense-adam cst-adam-fast' ;;
    esac
    for name in $methods; do
        method=$name
        lr=0.001
        if [[ "$name" == cst-adam-fast ]]; then method=cst-adam; lr=0.05; fi
        if [[ "$name" == cst ]]; then lr=0.05; fi
        timeout 900 python -m experiments.local_tangent_learning \
            --data "$data_root" --output "$results_dir/$name-s$seed.json" \
            --method "$method" --lr "$lr" --approximation full \
            --solver spectral --recompression-action pair \
            --seed "$seed" --steps 512 --eval-every 8 \
            > "$results_dir/$name-s$seed.log" 2>&1
    done
done
