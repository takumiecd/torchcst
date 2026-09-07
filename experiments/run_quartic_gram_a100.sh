#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Fail before timing if CUDA is unavailable.
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable; restore the A100 allocation first"
name = torch.cuda.get_device_name()
assert "A100" in name, f"Expected A100, got {name}"
print(name, torch.__version__)
PY
mkdir -p output
for precision in float32 float64; do
    PYTHONPATH=src python -m experiments.quartic_gram_benchmark \
        --device cuda --dtype "$precision" --atoms 64 \
        --starts 4 --iterations 80 --repeats 3 \
        --output "output/quartic_gram_a100_${precision}.json"
done
