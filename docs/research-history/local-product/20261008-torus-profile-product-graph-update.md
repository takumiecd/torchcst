# Intrinsic Torus profile-product Graph update

Follow the owner-selected centre-fibre S1xS2 profile-product revision2.
This branch builds on reference PR #71 and adds an explicit research-only
Torch atom_update Algorithm, not a new public optimizer default.
Intrinsic [A,5] FP32/FP64 only; ambient/older metrics/low precision are rejected.
The common Dispatcher/AtomUpdateBinding owns no optimizer state. AdamW retains
its proposal, moments and clocks. No linear-specific updater in the optimizer.

Reuse the exact Polar activity law; express the public intrinsic retraction
with device operations: limit arc displacement, periodic wrap, and cap section
normal coordinates below the antipodal margin. Keep R/r live. The eager law's
Python-double multiplication pi*R is reproduced in FP64 before casting each
operand to the parameter dtype. No Tensor-to-host reads during capture.

CPU gate:13 passed/20 CUDA cases skipped. Covers both activity modes,
FP32/FP64, three live major radii, minor-radius changes, zero/under/over-radius
Polar points, arc crossings/cap and section cap; update matches the public law
bit-for-bit. CUDA gates additionally compare20 captured AdamW updates/moments
and complete Y/dX/dP/parameter steps with evolving widths, Product/Strip and
reversed logical axes. Strict optimizer and numerical checks remain unchanged.

The ordinary CSTOptimizer wrapper still requires eager execution; the research
benchmark uses an explicit base-optimizer proposal plus declared update Plan,
as existing Euclidean complete-step measurements do. Intrinsic tangent/state
transport is identity. Ambient GPU/Graph updates are outside this candidate.

GPU verification and complete-step time/peaks remain pending. This is a Graph
comparison baseline and a prerequisite for fused CUDA work, not a speed claim.
Raw drivers/logs/results remain in ignored evidence/ and the shared pool.
