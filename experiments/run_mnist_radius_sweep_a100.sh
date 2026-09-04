#!/usr/bin/env bash
set -u

: "${MNIST_ROOT:?set MNIST_ROOT to the directory containing MNIST IDX files}"

seed="${SEED:-17}"
steps="${STEPS:-32}"
radii=("$@")
if [[ ${#radii[@]} -eq 0 ]]; then
  radii=(0.25 0.5 1.0 2.0)
fi

mkdir -p output
pids=()
labels=()

for radius in "${radii[@]}"; do
  label="${radius//./p}"
  labels+=("$label")
  (
    PYTHONPATH=src python -u experiments/mnist_current_api.py \
      --device cuda \
      --seed "$seed" \
      --atoms 64 \
      --steps "$steps" \
      --batch-size 128 \
      --train-size 8192 \
      --test-size 2000 \
      --trust-radius "$radius" \
      --solver-starts 4 \
      --solver-max-iter 80 \
      --data "$MNIST_ROOT" \
      --out "output/mnist_radius_${label}_seed${seed}_steps${steps}_a100.json"
  ) >"output/mnist_radius_${label}_seed${seed}_steps${steps}_a100.log" 2>&1 &
  pid="$!"
  pids+=("$pid")
  echo "START radius=${radius} pid=${pid}"
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "DONE radius=${radii[$index]}"
  else
    echo "FAILED radius=${radii[$index]}" >&2
    status=1
  fi
done

for index in "${!labels[@]}"; do
  log="output/mnist_radius_${labels[$index]}_seed${seed}_steps${steps}_a100.log"
  echo "RESULT radius=${radii[$index]}"
  tail -n 8 "$log"
done

exit "$status"
