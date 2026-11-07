# Launching only the current moment level

**English** | [日本語](multirow-levels-20261108.ja.md)

Code inspection found that packed_moments launches R×(N-2)×17 threads for each
level, although each level writes only its own two banks. Inactive levels also
receive the full grid before returning on the device mask. For N8192/16384,
levels2..log2(N) require12/13 such launches. This is scheduling overhead, not a
change to the Fourier/cell-mass operation.

Two separate research Algorithms combine the previous per-invocation geometry
reuse with a restricted moment grid: `cuda_radius_large_multirow_levels` saves
responses; `cuda_radius_large_multirow_levels_stream64` uses64-row recomputation.
At level l, launch R×2(N/2^l)×17 threads, map each to the original compact Z slot,
and call the unchanged compact_moment_element. Preserve the device-side required
bank mask, child-bank closure, ascending level dependency, summation order and
full moment storage layout. Total nominal threads across levels become
R×(N-2)×17 instead of12/13 times that value. The speedup is not assumed to equal
the thread-count ratio. Old Algorithms retain their original grid and defaults.
The retained-storage/64-row policies are unchanged.

Before measurement, freeze source, protocol and input hashes. Use only A100 MIG
3g.40gb, FP32 IEEE/TF32 off, N2048 qualification then N8192/16384,R256,A8N,
Xavier/q0/minimum-001 and frozen P/x/target/dY from the width study. Run the
complete targeted CUDA regressions and N2048 full VJP/independent quadrature/
matched-update gates for both candidates. Maintain all tolerances; preserve and
stop on failure/OOM/nonfinite/timeout. No simultaneous GPU experiment.

Fresh processes, A widths8192→16384 and saved→levels→levels_stream64→Dense;
B widths16384→8192 and Dense→levels_stream64→levels→saved. Each observation uses
the existing worker's10 warmup+51 synchronized complete timed steps; all61 enter
allocated/reserved peaks. Full initial/updated VJPs, same initial operator for
Dense, unchanged optimizer, state hashes, source/extension hashes and all samples
remain mandatory. Record both orders, complete-step time and memory; do not
infer quality, hardware replication, Graph support or public adoption. Compare
with previous geometry/windowing observations as separate source studies.

This is a premeasurement protocol. Stop further speculative optimization after
this comparison and preserve the measured options for review.
