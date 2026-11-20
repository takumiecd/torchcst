# Atom ordering and rational-radius Linear bottlenecks on RTX 4090

**English** | [日本語](rational-q-atom-order-4090-20261120.ja.md)

Profiles use the saved-input direct Plan from the
[retention experiment](rational-q-saved-response-4090-20261120.md).
RTX 4090, SM89, PyTorch 2.9.1+cu128, eager IEEE FP32. The fixed square
projection has M8 cell mass, `rational_radius`, a unit periodic grid and,
unless specified, 128 input rows.

## Where the time went

Width 16384 and 524,288 Atoms, using the saved width-4096 Query Atom
distribution replicated onto the wider grid as in the
[width sweep](rational-q-crossover-4090-20261120.md).
Values below are CUDA kernel times from separate forward and backward traces,
averaged over three samples. The uninstrumented combined time was 129.75 ms.
The Fourier-weight comparator took 78.11 ms with 48.14 ms forward and 29.97 ms
backward.

| Stage | Forward (ms) | Backward (ms) |
| --- | ---: | ---: |
| Per-Atom response | 24.66 | 24.36 |
| Atom gradient | — | 38.86 |
| Wide-tile coefficient aggregation | 16.75 | 18.19 |
| Evaluation at destination sites | 2.29 | 2.34 |
| Input Fourier summaries | 0.46 | 0.47 |

At width 16384, 32, 64, 128 and 256 rows took 31.08, 64.61, 129.75 and
257.53 ms for combined forward and backward. At width 8192 and 128 rows,
doubling the Atom budget from 131,072 to 262,144 changed combined time from
23.97 to 53.61 ms.

The live distributions matter. The replay had median physical radius 95.63
sites and 32.8% of Atoms at level 9. Keeping x, dY, amplitude and centers
fixed, synthetic radius fractions of 0, 0.5 and 1 gave combined times of
63.93, 129.40 and 69.69 ms. At fraction 0.5 the combined response kernels
took 58.51 ms. At fraction 1 every radius was 512, the same kernels took
3.29 ms, and wide-tile aggregation took 59.03 ms. The synthetic variants
change radius and the bank distribution jointly.

`batched_parameter_kernel` owns one Atom per CUDA lane and traverses all
row groups, reading both responses and the input/output Fourier summaries.
The response kernel groups eight rows and assigns one Atom per lane.
With the original Atom order, nearby lanes can take different radius
branches and address distant tiles. Sorting within a radius bank changes
these conditions while retaining the same Atom values. The runs record
kernel duration. Hardware cache, executed divergence and DRAM counters were
unavailable. The locality explanation follows from the access pattern and
the controlled ordering experiment.

Compilation resource reports showed 127 registers and a 1,128-byte stack
frame for the physical Atom-gradient kernel, with zero reported register
spill loads/stores. The weighted-row-sum candidate retained the FP64 profile
coefficient construction, used 101 registers and zero reported spills,
and ran the width-16384 gradient kernel in 40.00 ms versus 38.90 ms for its
matched control. Its standalone optimizer step was 134.60 ms versus
134.17 ms for the control. The separate weighted research branch retains the
candidate and results.

## Ordering experiment

These times use the same Atoms and the same binary. The permutation was
applied before measurement in this probe. Pairwise output and first-order
gradient differences were zero after restoring gradient order.

| Physical Atom order | Forward + backward (ms) |
| --- | ---: |
| Original | 131.84 |
| Radius bank | 77.02 |
| Radius bank, input tile | 42.11 |
| Radius bank, output tile | 41.54 |
| Radius bank, input tile, output tile | 30.80 |

The input-first order reduced the single forward response launch from
24.26 to 1.32 ms. Its two Atom-related backward kernels took 9.66 ms
(response) and 9.33 ms (Atom gradient). An archived FFN-down variant had used
joint tile sorting in research commit `5dd4c263` on 4096→1024. The saved
square Plan had used sorted ranges to aggregate outputs and the original
physical Atom order for responses and Atom gradients.

## Per-call ordering in CSTLinear

The research Plan `cuda_rational_radius_wide_ordered` builds the key from
the current decoded radius and centers every call. It gathers the physical
Atoms before native execution. Autograd maps the gradients back to the
original Parameter rows. All three arms use the same native binary. Times
average medians from two fresh process orders with eight warmups and 24
measurements in each process. The ordered measurements include key
construction, sorting, the gather, and gradient remapping. Peaks are for
forward + backward and use the PyTorch allocator.

| Width / Atom budget | Saved F+B (ms) | Ordered forward (ms) | Ordered backward (ms) | Ordered F+B (ms) | Fourier weight F+B (ms) | Peak allocated saved / ordered / weight (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 / 32N | 21.91 | 3.21 | 5.92 | 9.15 | 6.23 | 234.74 / 235.74 / 588.11 |
| 8192 / 16N | 24.33 | 3.08 | 6.89 | 9.98 | 20.12 | 314.99 / 315.99 / 2211.76 |
| 8192 / 32N | 53.44 | 4.52 | 11.48 | 15.99 | 21.02 | 470.49 / 472.49 / 2238.96 |
| 16384 / 32N | 130.62 | 7.57 | 23.64 | 31.23 | 78.01 | 938.99 / 942.99 / 8757.88 |

At 16384, optimizer-inclusive times for saved, ordered and Fourier weight
were 134.25, 32.92 and 78.49 ms. Their allocated peaks were 962.97, 966.99
and 8773.88 MiB. All output and first-order gradient tensors matched the
saved Plan byte-for-byte in the four timing fixtures. The targeted CUDA suite
had 14 passes and the CPU suite had 2,206 passes and 2,585 skips.

In the [seven-layer capacity fixture](rational-q-saved-response-4090-20261120.md),
the ordered Plan completed all three optimizer updates and matched each saved
Plan loss. The mean of updates 2–3 was 0.219 s and the allocated peak was
4.06 GiB, compared with 0.915 s and 4.03 GiB for the previous saved Plan.
The ordinary Dense arm in the prior comparison ran out of memory at its first
optimizer step. This is an eager FP32 capacity result on synthetic data.

## Artifacts

Ordered source `7f69afa582dd2eaa9478a8bad16c4d5af4cd413f`, source snapshot
SHA-256 `a74dc04ffa5562383016e3536d00a7ddc39b450ced5a70499d93a862626b6d59`,
native SHA-256 `ba92efe246c09074b6a8993e95f1431ffd5fd038e48e9a04a20c13e6cf99e632`,
results archive SHA-256 `fb9f2b5967a3491eba584ff9d7e9abce8c937112e949e6f471de8d1f2a5b48db`.
The ordered job was PID 71098. The experiment directory is
`output/rational-q-tile-order-20261120-01`, with the isolated source,
`ordered-replay.py`, `ordered-profile.py`, and `ordered-capacity.py`.
The dispatch plan is research-only. The stage and offline ordering
diagnostics are `bottlenecks.json` and `atom-order.json` under
`output/rational-q-saved-response-20261120-01`. The weighted trial is in
`output/rational-q-weighted-adjoint-20261120-01`.
