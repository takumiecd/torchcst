# Reusing per-Atom geometry across token rows

**English** | [日本語](multirow-geometry-20261108.ja.md)

The controlled N16384 row comparison's first order identifies forward/backward,
not the optimizer, as the large-R cost. At R256 the separate diagnostic medians
are41.59/59.91/1.42ms for forward+loss/backward/optimizer (they must not be summed
or subtracted to construct a primary time). Code inspection shows projection
and expansion decode the same radius, wrapped center and9 Fourier coefficients
for each8-row group, and again for backward.

Candidate `cuda_radius_large_multirow_geometry` computes the existing FP32
25-value geometry per Atom once per forward, then reuses it for projection,
expansion and gradient groups through backward. It is retained by that autograd
invocation, with the original input/Parameters and their normal version checks.
There is no cache on the Algorithm, no CPU atom read, no math/Polar/LR change,
and no dense W/dW. The extra forward retention is100A bytes (6.25/12.5MiB at
N8192/16384). Baseline already creates the same transient table in backward;
measure the peak rather than assuming this retention is free.

Keep the original saved and recompute routes selectable. Defaults, Tensor
transform fallback, first-order differentiation restrictions and old scopes
remain unchanged. New templates default to the original uncached route.
Check varying radius/periodic images, R1/9/256 tails, high owner keys, noncontiguous
input, both single-gradient branches, saved hooks, retained backward and stale
Parameter rejection. Qualify fullN2048/A8N Y/dX/dP and optimizer using existing
CPU FP64/independent quadrature and matched-state gates before performance.

Use R256/N8192/16384/A8N and exactly the frozen width-study initialization and
data. Existing worker loops:10 warmup+51 timed complete synchronized steps;
allocator peaks include all61. Fresh processes, two reversed width/arm orders:
A8192→16384, saved→geometry→stream64→Dense; B16384→8192, Dense→stream64→geometry→saved. Keep all
operator/gradient/Adam/Parameter tolerances and initial/updated full VJP checks.
Preserve failures; stop on any gate/OOM/nonfinite/timeout. Fresh source/build,
no concurrent GPU experiments. Compare same-kind final hashes across orders;
geometry-vs-saved byte equality is recorded, not assumed, since precomputing
can change compiler rounding. Public/default adoption remains a separate step.

This is a premeasurement hypothesis and protocol, not a speedup result.

## Windowed recomputation candidate (fixed before GPU measurement)

`cuda_radius_large_multirow_stream64` retains only x/P, owner keys and the same
geometry table. Forward emits64-row windows into Y; backward regenerates moments
and responses in64-row windows. It retains all ceil(R/8)×A×3 FP32 gradient partials
until one final reduction in exactly the original eight-row-group order. Thus
it does not sum independently reduced per-window dP (which would change order).
The unchanged two orientation passes and forward projection recomputation run
inside the synchronized complete-step clock; Y/dX window copies also count.
At R256 the Z/H/G scratch scales with64 rather than256 rows. Full x/Y/dX and the
partial table remain. No change to Atom count, Parameters, update state or batch
loss scaling. Check R65 as well as R1/9/256, and both gradient branches/retained
backward/version checks for both new candidates. Qualify N2048 for both before
all16 large observations. Record saved-vs-geometry-vs-stream final hash equality
without assuming it. This is not a public checkpointing API or an adopted default.

For stream64 at R256/N16384/A131072, those two Z/H/G pairs use about199.983MiB
in total, while the full partial table48MiB and geometry12.5MiB remain. The
shape-derived core working set is260.483MiB instead of860.434MiB. Input/output,
optimizer and owner metadata are additional; actual allocator peaks determine
the realized saving. A full-R loss/dY is preserved, so windowing does not change
batch-loss normalization or optimizer update frequency.
