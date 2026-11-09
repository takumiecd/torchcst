# Connected M16: qualification protocol

[日本語](connected-m16-qualification-20261109.ja.md)

Prespecified protocol. The [completed screen](connected-m16-results-20261109.md)
passes correctness but fails the performance gate. No learning or production
adoption. The GPU connection is admitted only at measured default M16 settings.

The connected definition uses K=3, T=2^l, P=3T. Its radius bank is the smallest
T with R<=bT, b=.9. Between aT and bT, a=b/1.25=.72, blend T and 2T with
S(log(R/(aT))/log(1.25)), S(u)=10u^3-15u^4+6u^5 extended by 0/1.
Within each bank let z=(mu+.5)/T-1.5, n=floor(z); blend windows n and n+1
with S((z-n-.4)/.2). Integrate entire cells over each immutable three-tile
window. Add repeated chart images. Input/output mixtures form a product,
including all cross terms. Both mixing derivatives enter the VJP.

The exact density remains Chebyshev M16, with unit mass per axis. This is not
exact compact support. The connected binding has physical revision9 (envelope
revision10 reserved for reference use); old revisions7/8 and checkpoints retain
their meaning. Checkpoint metadata rejects incompatible connection settings.

The initial research CUDA Algorithm accepts physical FP32 rows and first-order
VJPs, power-of-two flat-torus charts up to16384, B<=256 and A<=131072. It uses
cached local masses for T<16, shared tile moments for larger banks, atom response
H followed by coefficient aggregation/site expansion. Backward recomputes H
and derivatives; retained tensors are X/P and FP32 geometry, not a dense W.
Geometry scratch is not persistent trainable virtual atoms. Parameters stay A×4.
Atomic aggregation is nondeterministic. Higher-order gradients and Graph support
are not certified. No saved FP64 tensors; coefficient preparation uses local
FP64 arithmetic to qualify the generic M16 recurrence initially.

Compare legacy M8, legacy M16 and connected M16 in the *same new implementation*
to isolate degree and connection costs. These controls are not claimed to match
the fastest legacy production schedule. Square Linear B256, N2048 then8192
(and16384 if feasible), A=8N; identical seeded physical rows, X, target, fused
AdamW lr1e-4, zero decay, radius projection [.5,50], center wrap, dX included.
R is log-uniform in [.5,32] to exercise direct and hierarchical routes, rather
than only one sharp initialization. Dense is an IEEE-FP32 cost comparator with
standard nn.Linear initialization; it is not the same initial function.

Before timing: independent FP64 quadrature of the Chebyshev polynomial checks
Y, dX and all four physical gradients on20 atoms covering scale/position joins,
old boundaries, wide radii and repeated images. Gates: max absolute and relative
L2 errors <=3e-4. Check optimizer parameters, moments and counters against the
CPU definition over a short moving-state trajectory. Timing uses the existing
Linear runner and its isolated workers, synchronized whole eager steps,
3 warmups/9 rounds in this first screen. Record absolute allocated/reserved
peaks separately. This is one independent run per shape, not9 independent runs.
No Graph/capture memory claim is made. Repeat only promising small differences.

Learning is conditional on correctness and a usable complete-step cost. Preserve
negative results. The new fixture is explicitly research-only, rejected by the
existing submission/database adapter rather than mixed into old radius data.
A follow-up comparison to the existing optimized M8 route is required before
claiming competitiveness with the current implementation.
