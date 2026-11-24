# Fourier Conv2d on the WideResNet-22-2 layer shapes

**English** | [日本語](radius-channel-fourier-wideresnet-20261124.ja.md)

The nine distinct layer shapes cover the 21 convolutions eligible for masking
in the public RigL CIFAR-10 launcher. The RGB stem is dense in that setting.
The measurements use the channel sizes, spatial sizes and strides from
Evci et al., §4.3, and the [pinned RigL model](https://github.com/google-research/rigl/blob/d39fc7d46505cb3196cb1edeb32ed0b6dd44c0f9/rigl/cifar_resnet/resnet_model.py).
For 3×3 stride 2, both arms pad one pixel on the bottom and right before a
padding-zero convolution, matching TensorFlow SAME on the even stage inputs.

## Fourier computation

Each spatial tap has its own Atoms. Input and output channels form two
unit-spaced tori with their own periods. Both use one-cell base tiles and
merge by two, up to a dyadic radius cap that divides both chart periods and is
at most half the shorter period. The trial uses caps 8, 16, 32 and 64. At the
shorter axis's half-period, overlapping images of a local window are added
through the periodic fold.

The Linear code integrates each M8 harmonic over a cell and supplies its radius
derivative. Since both axes use the same radius and selected bank, forward
shares the density coefficients and backward shares the integrated coefficients
and radius derivatives. Center phase and owner wrap are evaluated separately
for each axis. The backward pass projects the convolution weight cotangent
onto the folded Fourier basis and contracts it with the Atom coefficient
derivatives. The existing optional stable owner sort preserves Atom order
in the saved parameters and gradients.

The CUDA decoder evaluates the rational envelope and analytic amplitude and
radius-fraction derivatives on device. It encodes the two radius endpoints
exactly as in the Torch decoder. Static Fourier with this decoder can capture
the whole layer step. The eager occupied-bank version reads a live bank mask
and processes only the selected banks.

## Matched training-step measurements

Two independent processes on an RTX 3090 (compute capability 8.6) used
PyTorch 2.6.0+cu126, CUDA 12.6, cuDNN 9.5.1 and driver 580.95.05. They used
seeds 17 and 23 with reversed arm order. Each case uses synthetic inputs at
batch 128, channels-last FP32 with TF32 disabled, and about 10% as many
trainable Atom scalars as dense weights for that layer. The initial radius
fraction is uniform on [0, 0.25] for narrow and [0.65, 0.98] for wide.
Timings include radius decoding, weight assembly and layout, any input
padding, convolution, dX and parameter gradients, fused SGD with momentum
0.9, learning rate 0.1 and weight decay 5e-4, and Atom projection. Dense
starts from the same represented weight. Each arm used 10 warmups and 30
timing samples. Graph arms also replay once before timing. The numbers are
ranges of CUDA-event medians from processes 5919 and 5934 in milliseconds.

| Shape | Input | Stride | Dense eager | CST eager | Dense Graph | CST Graph |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1×1, 16→32 | 32×32 | 1 | 0.326–0.344 | 0.673–0.836 | 0.294–0.297 | 0.467–0.473 |
| 3×3, 16→32 | 32×32 | 1 | 0.526–0.534 | 0.937–0.962 | 0.492–0.493 | 0.699–0.702 |
| 3×3, 32→32 | 32×32 | 1 | 0.674–0.676 | 1.082–1.102 | 0.607–0.636 | 0.813–0.862 |
| 1×1, 32→64 | 32×32 | 2 | 0.274–0.278 | 0.624–0.681 | 0.233–0.234 | 0.459–0.461 |
| 3×3, 32→64 | 32×32 | 2 | 0.828–0.833 | 1.112–1.233 | 0.782–0.790 | 0.994–0.995 |
| 3×3, 64→64 | 16×16 | 1 | 0.651–0.688 | 0.985–0.991 | 0.632–0.634 | 0.888–0.893 |
| 1×1, 64→128 | 16×16 | 2 | 0.217–0.227 | 0.704–1.334 | 0.181–0.181 | 0.425–0.431 |
| 3×3, 64→128 | 16×16 | 2 | 0.957–0.961 | 1.066–1.239 | 0.866–0.878 | 1.220–1.231 |
| 3×3, 128→128 | 8×8 | 1 | 0.805–0.809 | 1.034–1.058 | 0.750–0.752 | 1.550–1.553 |

The table shows wide radii. CST eager uses CUDA decoding with occupied banks,
and CST Graph uses CUDA decoding with static banks. The same-source eager
occupied-bank comparison with the Torch decoder was 1.18–2.75× faster with
CUDA decoding across individual paired conditions. Dense was faster in both
execution modes at every shape and radius regime. The smallest Graph gap was
1.26× on 3×3 32→64, stride 2. The wide 3×3 128→128 fixture benefited from
eager bank selection. Static Graph projects and expands all reachable banks,
and in that fixture took about twice the dense replay time. The JSON retains
both radius regimes, the legacy and sorted arms, and synchronized wall times.

The multi-case timing processes retained allocator memory after Graph captures.
For a fresh-process memory comparison, the wide 3×3 128→128 fixture was rerun
with one arm per process at seed 17. The table reports total PyTorch allocator
peaks in MiB, including capture and replay for Graph. GPU process memory was
unmeasured.

| Arm | Peak allocated | Peak reserved |
| --- | ---: | ---: |
| Dense eager | 63.99 | 86 |
| CST eager, occupied banks | 63.41 | 90 |
| Dense Graph | 63.99 | 120 |
| CST Graph, static banks | 102.48 | 184 |

## Validation and evidence

The two final processes checked W, Y, dX and all four rational Atom fields
against a CPU FP64 reference on every fixture. The targeted GPU suite also
checked independent scalar Chebyshev quadrature, exact dyadic boundaries,
periodic overlap, graph replay across changes in radius bank and center,
and the existing rational Wide Linear plans.

Maximum relative L2 error was 5.17e-6. The GPU suite passed 65 tests.
The CPU suite passed 2,219 tests with 2,640 skips, and the wheel and sdist
built. These are synthetic single-layer steps. Model integration, the public
dispatcher and CIFAR-10 training are outside these measurements.

The final source is `74194a5d7ed45227a6c9c31af9f22fd69049835a`. The source
archive, driver, logs and raw timings are preserved in the worktree's ignored
`output/conv-wideresnet-fourier-native-20261124/` and in the Kyutech
home-relative `cst-conv-channel-20261124/74194a5d/`. The first paired eager
comparison is in `output/conv-wideresnet-fourier-20261124/`, mirrored at
`cst-conv-channel-20261124/33236517/` and pinned to
`33236517aa1e7e92c883dc9a7745fc7d07d8e95c`. In the final snapshot, extract
the archive into `source/`, set the CUDA 12.6 header paths and Ninja as in
`run.sh`, and execute:

```sh
bash run.sh test
bash matched-eager/run.sh
bash isolated-memory/run.sh
```

| File | SHA-256 |
| --- | --- |
| `source.tar.gz` | `5feb5ad6564e16cf0e507339fb98bb80a9cc8b45be0ebf9991dcbe504ac7ea93` |
| `test.log` | `a4ba85addaac83e0ffc5b485f7c1155e962267877eddd499d5bc24bf7b58fda9` |
| `matched-eager/measure.py` | `3a26bd0e2ebe68b39bdee28e304da14e393edd9a3ad515c519d32eb624fae0a0` |
| `matched-eager/run.sh` | `97d4f0e592f3c6263da62aa3c5329ac78ec52abe0b6712e45211b6684b6ae377` |
| `matched-eager/run-17.json` | `fdfd67f8334904d223de07c579eb7ee36887028bb16a740e3e58e86243508b86` |
| `matched-eager/run-23.json` | `acaac2e2a7919ca73787c7eb5f18f1dd6b2b2e3c51c7d1e0eb38f47c98939f7c` |
| `matched-eager/final.log` | `ccfc5455dee2cb1fa2b82c1d5196a7ce58698654f9626d8efb6bcc4f17ec3eff` |
| `isolated-memory/measure.py` | `5c707afedfdf57e0a2e5d8d13745e181a00da8ca096aff170fd11f7354a3ee59` |
| `isolated-memory/run.sh` | `144ac7b6f300918ca226fa985695626634bab8ff188e3517d82fb4a3ced33cab` |
| `isolated-memory/run-dense.json` | `99b76244cd3433389ae74e74545f0ff14c7aa70eb8f854153f844d3911f2512d` |
| `isolated-memory/run-cuda-active.json` | `e2001d21eb7dd5f1d01cb143832b8eae70a7cd5d09d03908bd819bd080ab2518` |
| `isolated-memory/run-graph-dense.json` | `fb5a7331005bfabab2026b337864d7403cab6929ea4ed806eeadcf8a1d23fd3a` |
| `isolated-memory/run-graph-cuda-static.json` | `2cb7f30421be2fc1c7cd0362e5861de91c9b96498181f48cbc02fea4fd2bc670` |
