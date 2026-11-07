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
A8192→16384, saved→geometry→Dense; B16384→8192, Dense→geometry→saved. Keep all
operator/gradient/Adam/Parameter tolerances and initial/updated full VJP checks.
Preserve failures; stop on any gate/OOM/nonfinite/timeout. Fresh source/build,
no concurrent GPU experiments. Compare same-kind final hashes across orders;
geometry-vs-saved byte equality is recorded, not assumed, since precomputing
can change compiler rounding. Public/default adoption remains a separate step.

This is a premeasurement hypothesis and protocol, not a speedup result.
