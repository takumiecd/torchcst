# Small Linear: exact owner indexing before payload/cache tuning

Measured implementation checkpoint on `kernel/local-product-onchip-h`.
Runtime source: `ea94d86b5e4b52e0cf679e816f8491437a926467`.

## Change and contract

The recompute-VJP control repeatedly traverses every general-atom band for each
16-site spatial owner and batch block. The new `_index` route builds an exact
GPU ID list for the atoms whose current support intersects that owner, split
into the existing three rho bands. Forward lists use output support; dX lists
use input support. The consumer visits only those IDs. Exact singleton buckets
keep their existing direct input/amplitude path.

Physical 13-field metadata views, canonical Parameter/Adam moments, wide forward
H order, coefficient/normalization arithmetic, atom32, batch16, output16 and
four contraction warps are held constant. The ID lists are derived every forward
and saved with that call's backward state. No center rounding, uniform weights,
constant-width shortcut or detached center derivative is introduced. Full-domain
normalization derivatives remain in the parameter VJP.

The index builder still scans canonical metadata once per owner/direction.
It removes repeated traversal and useless consumer blocks rather than all
exploration. This experiment deliberately precedes the separate
[support-ordered payload proposal](20261005-layout-design.md); no new physical
payload permutation, shared-input stage or cache policy is introduced.

## Protocol and verification

Eight Cases: N64/N128, B32, A204/A819 (~5% of dense elements), initial rho1.25/3/8
or mixed1.25/3/8 at fractions0.5/0.3/0.2, seed41, FP32 IEEE/TF32 off. Every actual
initial decoded rho is asserted >1. Production minimum/birth remain0.25 and
sigma changes during every measured step. Fused capturable AdamW and production
fused Polar updates match the existing runner. The measurement includes support
preparation, persistent layout maintenance, index rebuilding, forward/backward
and optimizer updates.

Two full independent pool jobs recreate each model/optimizer/worker. Repeat2
reverses shape, width and plan order. Each execution has21 timing samples;
those samples are not independent runs. An earlier smoke job measures rho3 and
runs60 L4-host tests (3 declaration tests and57 GPU tests): exact support coverage
before/after large center moves, empty/singleton/two-site/floor cases, sliced
Domains, batches1/32/64, repeated backward, outstanding old backward after a
new layout refresh, and20 captured updates against public CSTOptimizer at both
sizes. Parameter, AdamW moments, step and live-width behavior are checked.

The full jobs independently compare scalar FP64 Y/dX/all-atom gradients at the
actual complete shapes for both plans and check nonzero oracle center gradients.
The existing runner also applies its separate small correctness gates before
measurement. CPU:857 passed/389 skipped; Ruff, eight prepared comparisons,
Registry/Case/snapshot checks and wheel/sdist build pass. Local skips are not GPU
validation.

## Results

| N / initial rho | Control µs | Indexed µs | Dense µs | Step reduction | Peak allocated control → indexed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 / 1.25 | 73.44 | 60.05 | 39.20 | 18.2% | 154,112 → 161,792B |
| 64 / 3 | 82.38 | 67.36 | 39.25 | 18.2% | 154,112 → 161,792B |
| 64 / 8 | 72.60 | 66.25 | 40.80 | 8.7% | 154,112 → 161,792B |
| 64 / mixed | 84.53 | 66.18 | 39.13 | 21.7% | 154,112 → 161,792B |
| 128 / 1.25 | 151.89 | 75.88 | 45.78 | 50.0% | 419,328 → 473,088B |
| 128 / 3 | 173.90 | 83.66 | 45.62 | 51.9% | 419,328 → 473,088B |
| 128 / 8 | 157.57 | 92.63 | 45.77 | 41.2% | 419,328 → 473,088B |
| 128 / mixed | 175.60 | 90.50 | 47.51 | 48.5% | 419,328 → 473,088B |

All eight cases improve in both independent executions. N128 rho3 closes70.3%
of the control-to-dense time gap (173.90→83.66µs, dense45.62µs). N64 rho3 closes
34.8% (82.38→67.36µs, dense39.25µs). The candidate still takes about1.66–2.02×
dense at N128 and1.53–1.72× at N64 in these cases. This confirms the repeated
consumer scan/zero work is a useful optimization target; remaining preparation,
parameter VJP and output/dX latency still matter.

All32 full-shape plan/oracle comparisons pass across the two jobs. Maximum
absolute Y/dX/dP errors: 2.26e-06 / 2.49e-06 / 4.46e-06,
with tolerance4e-4. Nonzero center gradients are checked in every oracle fixture.
Compiler reports show zero spill slots for all candidate kernels. N64 control
dX reports2 spill slots, so its compiler resource usage also changes with indexed
consumption. N128 control/candidate both report zero spills. CST reserved peak remains6MiB; GPU process usage is
unmeasured. Hardware/runtime: NVIDIA L4 (58 SMs), Torch2.11.0+cu130, CUDA13.0,
Triton3.6.0. All owned L4 VMs are stopped.

Times below are medians of the two independent execution medians, measured by
uninstrumented complete-step CUDA Graphs. Diagnostic phase events use a separate
Graph; their durations are not summed or subtracted from the primary time.

At rho3, N128's initial forward consumer list has1,057 useful atom visits instead
of6,552 traversed entries; dX has1,041 instead of6,552. N64 forward has246 instead
of816, dX250 instead of816. Counts cover spatial owners once and exclude the
batch-block multiplier. The builder's own scans are additional and included in
step time.

Index tensor capacity is7,296B at N64 and53,504B at N128, including per-band
offsets. These are capacity budgets, distinct from measured capture/replay peaks.
Allocated memory increases are retained in the comparison. Reserved allocator
memory and GPU process memory are distinct. No physical-DRAM reduction or warm
cache-residency claim is made from elapsed times.

## Evidence and reproduction

Use `plans-local-index.json` with
`cases/local-index-{64,128}-rho{1_25,3,8,mixed}.json` in the existing Linear runner.

```bash
python -m tools.kernel_dev prepare \
  --plans benchmarks/cuda/linear/plans-local-index.json \
  --case benchmarks/cuda/linear/cases/local-index-128-rho3.json \
  --candidate persistent-supportprep-band-recompute-vjp-index-atom32 \
  --output output/owner-index-comparison
python -m tools.kernel_dev check \
  --plans output/owner-index-comparison/plans.json \
  --case output/owner-index-comparison/case.json
python -m benchmarks.cuda.linear.run \
  --plans output/owner-index-comparison/plans.json \
  --case output/owner-index-comparison/case.json \
  --source-commit ea94d86b5e4b52e0cf679e816f8491437a926467 \
  --polar-update fused --phase-diagnostics --kernel-diagnostics \
  --output output/owner-index-comparison/result.json
```

Run from the measured source checkout when using that source commit argument.
The ignored driver only orchestrates existing tests, scalar oracle and runner.
Preservation: `benchmarks/cuda/linear/evidence/owner-index-20261006/` stores driver,
CPU/build logs, analysis, comparison preparation and DB verification. Pool job
directories under `~/.local/state/colab-l4-pool/jobs/` retain source archives,
raw results, samples, receipts and submitted drivers.

The [machine summary](20261006-owner-index.json) records job IDs, source/result
SHA256, matched package source hashes, error bounds, per-execution medians and
memory. Completed runner artifacts, including the smoke run, are preserved in
PostgreSQL with byte-identical export and idempotent reimport. Source manifests
match all196 package Python files; runner source hashes are also verified.

This remains an explicit benchmark research route. Arbitrary geometry, larger
outer GEMM scheduling, source-layout/cache tuning and public dispatcher adoption
require separate validation.
