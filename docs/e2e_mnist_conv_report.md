# Continuous CSTLinear convolution on the CONV-GROW-4 protocol

## Scope

`experiments/e2e_mnist_conv.py` reproduces the CONV-GROW-4 training protocol
with a different target-filter family. The MNIST split, 8→16 target convolution,
batch-tape seeds, 600 updates, K=4→36 ladder, birth steps, and AdamW learning
rates are retained. The original per-offset free rank-one blocks are replaced
by `CSTConv2d`, backed by a continuous Gaussian `CSTLinear` over input-tap
coordinates `(channel, dy, dx)` and output-channel coordinates.

This is therefore a protocol reproduction and a new continuous-family arm,
not a same-family numerical replication of the G4 rank-one result.

## Result

| Arm | Mean eval accuracy | Seed 0 | Seed 1 | Seed 2 | Active params | Final K |
|---|---:|---:|---:|---:|---:|---:|
| dense | 75.08% | 72.61% | 75.68% | 76.95% | 1418 | — |
| CST static K9 | **44.69%** | 48.34% | 44.58% | 41.16% | 311 | 9 |
| CST static K18 | 37.58% | 32.67% | 30.62% | 49.46% | 356 | 18 |
| CST static K36 | 37.17% | 31.01% | 48.78% | 31.74% | 446 | 36 |
| CST gradient grow | 34.68% | 29.59% | 33.35% | 41.11% | 446 | 36 |
| CST random grow | 34.24% | 31.35% | 32.81% | 38.57% | 446 | 36 |
| CST matched jump | 40.22% | 31.64% | 47.31% | 41.70% | 446 | 36 |

The dense rows reproduce the original G4 values to the reported four decimal
places for all three seeds. All arms within a seed have the same saved tape
hash. Both progressive growth arms perform six structural events with counts
`(6, 6, 5, 5, 5, 5)` and end at K=36; the jump arm reaches K=36 in one event
at the compute-matched step 219.

## Interpretation

The continuous CST result reproduces the earlier Gaussian-convolution wall:
static K9 reaches 44.69%, close to the 44.25% Gaussian reference quoted by the
G4 report. More atoms do not rescue it: static K18/K36 fall to 37.58/37.17%.
The current-batch first-order candidate selector raises progressive growth only
0.44 percentage points over random growth (34.68% versus 34.24%); both remain
below static K9 and static K36. A one-shot matched jump is better at 40.22%, but
still below static K9.

This strengthens the prior conclusion rather than overturning it: imposing a
Gaussian metric on channel indices is the dominant restriction, and schedule
or atom count does not repair that representation. In the original G4 family,
free per-offset channel factors raised static K36 to about 63.6%; the gap to the
continuous `CSTLinear` arm is therefore the expected cost of the channel-index
geometry, not evidence that the new convolution adapter is numerically wrong.

## Reproduction

```bash
python experiments/e2e_mnist_conv.py \
  --device cpu \
  --data-dir ../cst/data/MNIST/raw
```

Raw output: `results/e2e_mnist_conv.json`. A 20-update path check is saved as
`results/e2e_mnist_conv_smoke.json`.
