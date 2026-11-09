# Connected M16 FP32 and reuse: prespecified optimization protocol

[日本語](connected-m16-reuse-20261109.ja.md)

Status: prepared, not a GPU result. Baseline is the frozen C2-connected M16
implementation from5ade3859/PR159. Mathematical revision9, whole-cell K3 windows,
quintic radius/position mixtures, normalization and every blend derivative stay
unchanged. There is no public registration, default or optimizer change.

Two new research recipes are registered in the existing Linear runner:

- `connected16-fp32`: positive homogeneous Chebyshev16 coefficients and analytic
  radius derivatives in FP32, computed once per active scale and reused across
  both windows and axes. No FP64 task arithmetic or saved Tensor is required.
- `connected16-reuse`: the same preparation plus8-row response reuse, child-tile
  DC/even-harmonic moment reuse, constant K3 phase rotations, and tile-owner
  coefficient aggregation. Wide coefficients have one owner and no per-Atom
  global atomics; direct tiles reduce locally before one atomic per bank/site.

H remains a complete per-Atom response before amplitude. Forward saves X/P,
FP32 geometry and per-invocation routing ranges; backward recomputes H and its
partials. Atom Parameters remain A×4. No persistent virtual Atoms or dense W.
Routing is rebuilt from current geometry each forward. Sort/range construction
and all scratch are included in complete-step measurements.

M16 coefficients are obtained by squaring the homogeneous scaled Chebyshev
recurrence and converting each Fourier coefficient to the nonnegative basis
z^(16-k-j)(1-z)^j, z=cos²(pi R/P). Common scale cancels in normalization.
The generator and independent CPU FFT/quadrature checks are retained. FP32
rounding and the changed sum order must meet the same absolute/relative L2
3e-4 Y/dX/all-field gate as the original implementation.

Use the existing connected fixture and independent FP64 oracle at N128,2048,
8192,16384 before timing each width. Check four-step AdamW state parity, partial
row groups, retained/multiple outstanding forwards and SavedVariable mutation
protection. Eager Graph support stays unqualified; no capture-memory claim.

Compare baseline, FP32 preparation, full reuse, optimized M8 control and Dense
on one reserved idle RTX4090, B256,A8N, identical P/X/target hashes, physical
radii log-uniform[.5,32], FP32 IEEE, common fused AdamW/projection.5 warmups,
15 timed synchronized complete eager steps. Allocated/reserved peaks include
resident Parameters/gradients/Adam and transient geometry/layout/moments.
Measure profiling separately. A run's timing samples are not independent jobs.
Repeat a promising result in an independent run; preserve failures and sources.
