#!/usr/bin/env bash
# Run from the repository root, with CUDA PyTorch and the native build tools ready.
set -euo pipefail
data_root=${1:?Pass the MNIST raw-data directory}
results_dir=${2:?Pass an output directory}
mkdir -p "$results_dir"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
for seed in 17 29 43; do
    case "$seed" in
        17) methods='spectral-pair spectral-stream krylov-stream' ;;
        29) methods='krylov-stream spectral-pair spectral-stream' ;;
        43) methods='spectral-stream krylov-stream spectral-pair' ;;
    esac
    for method in $methods; do
        case "$method" in
            spectral-pair) solver=spectral; action=pair ;;
            spectral-stream) solver=spectral; action=jvp_vjp ;;
            krylov-stream) solver=krylov; action=jvp_vjp ;;
        esac
        timeout 900 python -m experiments.local_tangent_learning \
            --data "$data_root" --output "$results_dir/$method-s$seed.json" \
            --approximation full --solver "$solver" --recompression-action "$action" \
            --basis-size 256 --seed "$seed" --steps 512 --eval-every 8 \
            > "$results_dir/$method-s$seed.log" 2>&1
    done
done
