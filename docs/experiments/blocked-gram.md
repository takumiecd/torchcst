# Exact separated Gram action

The internal `FactoredFrameGeometry.gram_matvec` applies the displaced Taylor
frame Gram plus explicit damping, without constructing a GramSystem. This initial milestone established the contraction oracle before changing
the linear-system algorithm. An opt-in moment-compression integration now
exists: see [PCG validation](pcg-gram.md). Existing default updates still use
the previous solver.

For B = J + H[d, .], each column has four separated rank-one terms. The
implementation evaluates all sixteen pairs between target and source column
blocks, multiplies their factor inner products elementwise, and accumulates
against the source vector. It retains cross-atom interactions, factor
coordinate derivatives, and displacement corrections. No rank truncation,
block diagonal approximation, or change of damping is made. Floating-point
summation order differs from explicit Gram multiplication.

With D=KP and block size b, each pairwise tile is at most b by b. Extra eager
workspace with gradient recording disabled is O(b*(I+O)*P + b^2 + D), in
addition to existing factor values/Jacobians/Hessians. Those supplied factors
still cost O(K*(I+O)*(1+P+P^2)). Thus K proportional to width still entails
quadratic factor storage; this milestone does not claim linear total memory.
If the caller records gradients through the contraction, autograd can retain
intermediates across blocks, invalidating the inference workspace bound.

This initial implementation uses Python loops with shape-only bounds and
GPU tensor operations. It introduces no data-dependent host reads, but has
many kernel launches and recomputes source blocks. Do not capture the whole
unrolled computation and assume the same memory bound: graph pool lifetimes
must be measured separately. A fused/tiled GPU implementation and iterative
solver are subsequent work, not established performance results.

Tests compare against the independent visible Jacobian/Hessian Gram for
float32/float64, two kernel variants, zero/nonzero frame displacements, and
block sizes 1/3/32 (including partial blocks). Float64 agreement uses
rtol=1e-11, atol=1e-12. A tensor-operation allocation audit with synthetic
factors rejects full visible matrices and full Gram matrices. A CUDA test
compares the result under synchronization-error mode after cold setup.

Local full suite: 233 passed, 24 skipped (CUDA unavailable locally).
No learning speed, peak training memory, or convergence improvement is claimed
until this operator is integrated and measured end to end.

A100 MIG3g.40gb targeted suite: 26 passed, including the CUDA action under
sync-debug error mode. This is not a full profiler synchronization audit.
