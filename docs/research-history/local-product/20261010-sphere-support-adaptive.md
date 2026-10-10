# Sphere support-count adaptive fusion study

The user requested execution that distinguishes short and broad supports,
including support discovery and normalization rather than H arithmetic alone.
This study preserves the existing pair of intrinsic S² Explicit charts,
Polar/Triweight chord profiles, complete per-chart discrete L2 floor 1e-6,
live widths, task-width stop-gradient, and all six atom gradients. No atom or
nonzero support contribution is discarded. Public dispatch remains separate.

## Fixed candidate and exact fallback

One candidate uses a fresh 8³ device cell CSR for each chart on every forward.
It queries conservative coordinate boxes on the actual normalized FP32 sites;
it does not substitute unit-sphere distance identities. The original positive
gap predicate determines support, including positive gaps whose raw squares
underflow. A tiny CTA fuses support discovery, complete norms, short input H
and output accumulation only when both sides meet all guards: at most four
query rows, 128 candidates and 16 actual positive-gap support sites per side.
Invalid or unsafe bounds, query/count overflow and near-floor uncertainty
select separate flagged full scans and the original recompute G4/merged direct
forward. Tiny and fallback atom contributions must be disjoint and complete.

CAP64, tile16 and lossless int16 support IDs remain fixed. Original decoder,
intrinsic Jacobians and backward arithmetic remain unchanged. Backward consumes
saved sites/centres/precision/norm/count/index/H and amplitude snapshots rather
than mutable live values. H is retained for backward: this candidate does not
claim to remove its allocation. Full U/V, Phi and W/dW are not allocated.
FP32 IEEE, disabled TF32 and disabled floating-point fusion remain in force.

The preceding standalone cell preparation study did not establish a stable
sigma3 speed gain over compact full scans. The new mechanism instead fuses
query and contraction for confirmed tiny supports. Its CSR, classification,
fallback and launch overheads must be included in measured complete steps.

## Predeclared comparison before GPU execution

The four fixed cases are N1024/N2048 with batch32, floor(.05*N²) atoms,
seed41, and initial sigma1.25 or sigma3. Sigma3 is the ordinary primary
objective; sigma1.25 is a separately labelled sharp-support condition, away
from the sigma1 activity boundary. A gain in the sharp condition must not be
reported as a sigma3 gain. No synthetic mixed-width fixture is introduced.
Broad and mixed supports remain mandatory correctness witnesses.

Each case contains compact W, recompute G4/merged direct, the single tiny16
candidate, and dense: 16 isolated workers without filtering. The same Sphere
fixture, MSE, fused AdamW lr1e-4/weight-decay .01 and existing research Graph
geometry update apply. A benchmark-only CLI exposes the additional sigma1.25
case while calling the existing complete-step worker unchanged with explicit
plans. The public API and existing scaling CLI contracts are unchanged.

Independent FP64 checks cover full sites/all atoms for Y, dX and all-dP at
initial and after24 actual updates, with maxabs AND relative-L2 <=4e-4. The
same two warmups, one initial capture/replay and 21 timed replays total24.
Every atom's live width and optimizer clock must be checked. Peak allocated
and reserved include capture/replay; total process memory is not measured.
Independent phase/route-count diagnostics do not alter primary timing or
subtract preparation costs. Actual tiny/fallback fractions must be observed,
not inferred from sigma alone.

New regression tests must pass with zero skips, then all16 full correctness
workers must pass on the same frozen source before timing. Tests must include
exact positive support/count/norm, 16/17 and query/candidate limit transitions,
floor/FTZ and boundary witnesses, mixed side supports and exact full overflow,
requested gradients, regular zero amplitude, retained forwards, live buffers,
Graph replay, and complete optimizer state/clock checks. All inherited runtime
files must remain byte-identical to the measured parent. Compilation-heavy
tests may use disjoint partitions whose union is the exact collected suite.

Budgets are 900s pool outer, 875s driver and 350s child per job. Failures and
partial results remain preserved; no omitted cases, tolerance relaxation,
post-result retuning or partial winner selection is permitted. Qualification
is per size and sigma against compact W: strictly >3% speed improvement, or
at least5% allocated reduction with <=3% time regression. Any eligible case
requires independent reverse confirmation of the full16-worker cohort with
all controls retained. Public adoption is a separate decision.

## Provenance and result status

The research checkout is on `kernel/sphere-support-adaptive`, starting from
`53fcda1f19302e7c235dca0b0f814a72d1c0951e`. Implementation, tests and frozen
comparison tooling are in preparation. Source hashes, exact collections,
completed outcomes, job IDs and recovery proofs will be recorded before any
integration decision. No performance improvement has been established yet.
