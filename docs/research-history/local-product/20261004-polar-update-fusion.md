# Fused PolarAmpWidth update for small linear training

Research branch `codex/local-product-hybrid`, parent `b3edca58`.

The matched-size experiment identified the post-backward optimizer stage as a
large fixed cost: roughly80–92us for51–819 atoms, compared with7–12us for the dense
reference's ordinary fused AdamW. This experiment splits that stage into old
parameter snapshot, AdamW proposal and PolarAmpWidth update, then fuses the last
component into one Triton launch.

## Contract and implementation

Keep PyTorch's fused capturable AdamW, moments, step count, weight decay and
epsilon behavior unchanged. Clone the old canonical atom parameters, let AdamW
write its proposal, and apply the production Euclidean polar update in place.
The update uses `proposal-old`, including FP32 rounding. It projects the old
polar vector, removes radial task motion, computes finite-chord/time-energy
activity and dormant expansion, applies the exact radial flow and updates both
Euclidean centres. The operation order, normalization/fallback, radial bounds
and live device scalar values are retained. Floating-point contraction is
disabled. External writes increment the Torch tensor version.

The new research route accepts contiguous CUDA FP32[A,4] parameters and distinct
old/proposal buffers. The production optimizer API is unchanged. Sigma remains
live/trainable. Local-H/saved-H choices and persistent slot placement are
unchanged, with shared sigma, spacing1 and minimum sigma.25 in these fixtures.

Use the existing Linear runner with explicit `--polar-update fused`; the default
`torch` remains the control. Worker records contain the selection and source
hashes. The runner's correctness gate exercises the selected update against the
production geometry. `--phase-diagnostics` adds update component events only
in a separate post-measurement graph; primary complete-step timing/memory has
no event instrumentation. Component times cannot be added/subtracted as an
exact decomposition of synchronized wall-clock training timings.

## Validation and experiment provenance

Local checks:129 pass,15 CUDA-only skips; Ruff and whitespace checks pass.
GPU validation and matched timing/memory results passed; verified job/source/result
hashes are recorded below. Raw evidence stays in ignored
`benchmarks/cuda/linear/evidence/local-update-20261004/` and pool job directories.

An initial test submission `l4job-78d3efb1fe7d4e5485ecac5a0022c5a6` failed before
kernel execution because its test-only `BandwidthBounds` omitted required
`upper_floor`. The fixture was corrected; its verified logs/receipt are retained
under `failed-config/`. This failure supplies no performance evidence.


L4 checks:67 tests pass in each successful job. They cover six direct update
comparisons (two activity modes, three step sizes), zero/tiny/under/over-radius
projection, amplitude-dependent dormant expansion with w_c.2,259 atoms with a
masked tail, version increments and changed device scalars during Graph replay.
Captured training compares Y, dX, every atom gradient, p, Adam moments and step
against the public FP64 factored CSTLinear/CSTOptimizer for20 updates atN32/64/128.
The final job also repeats20 updates with the original Torch update atN64.
Tolerances remain4e-4 for Y/dX/dP and3e-6 for parameters/optimizer state; direct
production update comparisons use2e-6. Widths change throughout training.

The nine full-shape independent FP64 scalar comparisons pass; maximum absolute
Y/dX/dP errors are1.745e-6/1.792e-6/2.491e-6. The final runner's selected fused
update correctness gates pass for all three repeatedN64 cases.

## Complete-step results

Same L4, batch32, FP32, TF32 off, A=floor(.05*N²), persistent hybrid mid4.
Compare fresh processes with matching initialization/input hashes,21 samples
and synchronized complete Graph replay. Dense uses the same shape/batch/AdamW
settings but ordinary trainable dense weights, a performance reference rather
than an equivalent parameterization. No outer GEMM is built.

First validated sweep (us; all sigmas remain trainable):

| N | initial mixture | Torch update | fused update | dense | speedup |
|---|---|---:|---:|---:|---:|
| 32 | middle | 123.463 | 53.554 | 36.948 | 2.305x |
| 32 | narrow-only | 110.705 | 40.385 | 36.527 | 2.741x |
| 64 | early | 170.493 | 92.610 | 39.130 | 1.841x |
| 64 | late | 136.472 | 58.705 | 39.395 | 2.325x |
| 64 | middle | 150.310 | 73.115 | 39.204 | 2.056x |
| 64 | narrow-only | 123.647 | 46.125 | 39.610 | 2.681x |
| 64 | sigma3 | 177.184 | 99.503 | 39.199 | 1.781x |
| 128 | middle | 236.125 | 153.411 | 45.877 | 1.539x |
| 128 | narrow-only | 147.195 | 66.168 | 45.663 | 2.225x |

Middle is a nominal50% live singleton initialization; early/late nominal10/95%,
and narrow-only is100% live singletons at initialization. These synthetic
mixtures are not a claim that training converges toward one-hot behavior.
Sigma3 initializes every atom to rho3 with the same amplitude/centres; it is
an ordinary-width case, never a fixed-sigma or singleton-only shortcut.

Final source remeasurement after wiring the selected update into the runner's
correctness gate:

| N64 fixture | Torch us | fused us (p10–p90) | dense us |
|---|---:|---:|---:|
| middle | 150.622 | 73.103 (72.929–75.646) | 39.204 |
| narrow-only | 123.484 | 46.274 (45.988–47.634) | 39.579 |
| sigma3 | 177.722 | 99.636 (99.246–101.823) | 39.373 |

The instrumentedN64 middle polar component changes81.920→5.120us; old snapshot
and AdamW remain3.072 and7.168us. Its aggregate instrumented update changes
97.280→19.456us. Extra event nodes change timing relative to the earlier
88us aggregate diagnostic: use matched event instrumentation, not a comparison
across those graphs. Uninstrumented complete-step median changes150.622→73.103us
(51.5% lower). Primary forward/backward algorithms are unchanged.

All nine fixtures still refresh both persistent views52 times with0 moved atoms,
0 incremental repairs and0 overflow rebuilds during the primary measurement.
All51/204/819 sigma values change at the respective sizes. No stable-layout
movement-speed improvement is claimed.

## Memory and remaining costs

Measured full-step peak allocated memory includes CUDA Graph capture/replay.
It is unchanged by update fusion: N32.069824MiB, N64.145020MiB, N128.397949MiB;
allocator reserved memory is6MiB in every candidate/control. Polar temporaries
are removed, but the existing backward/layout allocation peak dominates, so
this is not a measured peak-memory reduction. Total GPU process memory is
unmeasured; neither allocated nor reserved bytes represents it.

The dense reference peaks32.534/32.596/32.815MiB allocated and46MiB reserved in
this runner, including framework/library/capture workspace, not only weights.
Do not interpret that difference as a tensor compression ratio.

The complete step remains slower than dense in every fixture: N32 narrow-only
is40.385 versus36.527us and finalN64 narrow-only46.274 versus39.579us. N64 middle
and sigma3 remain1.86x and2.53x dense time. RemainingN64 mixed phase diagnostics
are forward/loss32.768 and backward28.672us; sigma3 costs48.128 and39.936us.
Next priorities are current-state preparation/refresh and contraction/gradient
traffic, followed by reducing the remaining old-snapshot/update launch count
without altering AdamW numerical state. Keep the fused route as an explicit
research option; do not promote experimental alternatives into main here.

## Provenance and reproduction

Hardware: NVIDIA L4, driver580.82.07,23,034MiB; Torch2.11.0+cu130, CUDA13.0,
Triton3.6.0, Python3.13.15. Both successful batches share one owned L4, run
sequentially in the host-wide pool; cleanup is verified after drain.

- Size sweep: `l4job-1489354653a24bcab078608ace71f065`.
  Source archive SHA256 `7fc56a40e4594b214e224a6c4d7f24a4a38deb5547b5d3640f8fb1b1a3c53f89`;
  verified result archive SHA256 `75961300a6ef04faec2f0a277e04eb8f9ec0e30c286e26eb3c33f1610cbe8f94`.
- Final runner gate/remeasurement: `l4job-8dd79fe3849b462995837a0586c7a6d6`.
  Source archive SHA256 `2c5916966f76bc4e444e8b705018ee6f5572fbd4b36c31341a24c507ede98c98`;
  verified result archive SHA256 `c8ce4b1862cacbd3a22c9a9503b472700f566e1ffa7b32a474b1d0a40a3115d6`.

All backend Python files are unchanged between the successful source snapshots.
The final local backend, runner and test hashes match the final measured
snapshot; only research notes were completed afterward. Results/receipts and
summaries are retained under ignored `size-results/`, `final-results/`,
`size-summary.json`, `final-summary.json`. Pool also retains frozen sources,
original drivers and verified archives.

Reproduce correctness and the same existing small Linear benchmark:

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_local_polar_update.py tests/test_local_size_comparison.py tests/test_cst_optimizer.py
python -m benchmarks.cuda.linear.run --case benchmarks/cuda/linear/cases/local-size-64-middle.json --plans benchmarks/cuda/linear/plans-local-persistent.json --polar-update fused --phase-diagnostics --output /tmp/local-middle-fused.json
```

The benchmark command also measures the other catalog routes. The sweep drivers
in ignored evidence reduce the existing case's selected plans to the persistent
route and rerun with`--polar-update torch` for matched controls. The pool's
frozen `__pool_driver__.py` and recorded source-files hashes make the exact
submitted selection reproducible without adding raw generated artifacts to Git.
