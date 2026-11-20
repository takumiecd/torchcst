# RTX 4090 rational-radius dispatcher refresh

**English** | [日本語](rational-wide-dispatch-refresh-4090-20261120.ja.md)

The public table selects Plans for square M8 rational-radius Linear in eager
IEEE FP32 with 128 input rows and both input and Atom gradients. It targets
RTX 4090, Torch 2.9.1+cu128 and Triton 3.5.1. The
[Neon generation request](../../../benchmarks/dispatch/requests/rational-wide-rtx4090-v2.json)
ranks two independent complete-step runs per Plan by the median of their
timing medians. The package uses the generated exact-match table locally.

## Catalog measurements

The standard runner used the initialized Atom tensor and a fixed MSE target,
with eight warmups, 24 measured updates and eager CSTAdamW. The ordinary
Dense control starts from the same effective weight and uses AdamW. Dense
is a timing and memory control. Each cell reports optimizer-inclusive time
in ms and the maximum observed peak allocated MiB. The original benchmark
JSON and leaderboard also include the reserved allocator peaks.

| Width and Atom budget | CST Fourier weight | Original CST direct | Tile-ordered CST direct | Ordinary Dense | Selected Plan |
| --- | ---: | ---: | ---: | ---: | --- |
| 2048 / 32N | 2.63 / 171 | 18.18 / 119 | — | 0.43 / 85 | Fourier weight |
| 4096 / 32N | 7.68 / 590 | 35.39 / 237 | 5.62 / 241 | 1.01 / 282 | Tile-ordered direct |
| 8192 / 16N | 22.10 / 2210 | — | 5.95 / 321 | 4.08 / 1060 | Tile-ordered direct |
| 8192 / 32N | 24.93 / 2243 | — | 9.57 / 482 | 4.07 / 1060 | Tile-ordered direct |
| 16384 / 32N | 85.83 / 8766 | — | 17.21 / 963 | 14.24 / 4152 | Tile-ordered direct |

All ten runner artifacts passed the full-site, all-Atom FP64 reference
checks. The width-16384 direct Plan peaked at 1252 MiB reserved, compared
with 9870 MiB for the Fourier weight Plan and 4198 MiB for ordinary Dense.

## Checkpoint replay

All five bundled entries picked the declared Plan through the ordinary
`CSTLinear(x)` API on the RTX 4090. Output, input gradient and Atom gradient
agreed with the same explicitly fixed Plan, with a maximum relative L2
difference of `2.54e-8`. The first check required bit equality and caught
ordinary FP32 accumulation-order variation in the width-2048 weight Plan.
The repeated comparison used `rtol=4e-5, atol=2e-6`.

The replay used the TinyStories seed-17 Query Atom tensor from update 4,882,
with synthetic inputs. The wider fixtures tile the same saved Atoms onto
the larger grids. Weight and tile-ordered direct outputs and gradients
agreed within a relative L2 difference of `2.12e-6` across the four replay
conditions. Timed pairs use the mean of the two medians and maximum peak
allocation, with the execution order reversed.

| Checkpoint replay | Fourier weight step / MiB | Tile-ordered direct step / MiB |
| --- | ---: | ---: |
| 4096 / 32N | 6.25 ms / 614 | 6.80 ms / 281 |
| 16384 / 32N (tiled) | 78.55 ms / 8938 | 18.30 ms / 1111 |

At width 4096 the initialized fixture selects direct by speed, while the
learned Query replay measures a 0.55 ms increase and a 54% drop in allocated
memory. The exact table keys on operator and runtime metadata. The named
Swin-MLP, Keyword-MLP, MTGNN and PatchTST experiments use Polar coordinates
and different operators. The published residual CNN export uses materialized
classifier weights. Replay here covers the eligible Linear projection and
its short updates.

The selector and checkpoint replay ran at
`dd23468f8e860b9462fa90ea1bb45a6cc1682127` (PID 873357) with native SHA-256
`b2894baec63905d083f64fd081bf3dbc09a83dee33956db01b726e2d2f01a347`.
The local `auto-results.json` SHA-256 is
`9e875eb2789a2d8a56799ca5f19505f26b2d34606c84f0f6b383d00e58990ccc`.

## Evidence

The runner used source commit `97043337d69be760fc6a4e702149391af19610f6`,
source archive SHA-256
`f927d5990ce0350c04e0957a9279aec24b339dbdcc2ac952d3e71aee98118bf9`,
and GPU job PID 76111. The verified result archive SHA-256 is
`8c17830dd4c014d353c1b71151476f4289d7d1b8739163ee926e7a514d5f6a60`.
The source, driver, raw timing and correctness artifacts, Neon receipts,
and frozen dataset are in `output/rational-wide-axis-integration-20261120-01`.

The Neon source ID is
`b73f44c8dd82c6edd64dfc07f110533213833372e7030a9145c258425385809d`.
All ten stored originals exported byte-for-byte. The dataset snapshot is
`21cf0cc405ba7bfd09d4217ac36b6bded4db5419cb8f92b9b7800b52fc7bf6af`. Offline generation from that snapshot reproduced the
packaged selector and leaderboard.
