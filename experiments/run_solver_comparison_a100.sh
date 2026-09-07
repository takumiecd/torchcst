#!/usr/bin/env bash
set -euo pipefail
# DATA must identify the MNIST raw IDX directory on this host.
: "${DATA:?Set DATA to the MNIST raw IDX directory}"
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
assert 'A100' in torch.cuda.get_device_name(), 'Expected A100'
print(torch.cuda.get_device_name(), torch.__version__)
PY
export PYTHONPATH=src
python -m experiments.mnist_solver_comparison \
    --device cuda --evaluation visible --data "$DATA" \
    --steps 32 --seeds 17 --output output/solver_comparison/learning
python -m experiments.mnist_solver_comparison \
    --device cuda --evaluation visible --data "$DATA" \
    --profile --steps 8 --checkpoints 1 8 --repeats 3 --seeds 17 \
    --output output/solver_comparison/profile
