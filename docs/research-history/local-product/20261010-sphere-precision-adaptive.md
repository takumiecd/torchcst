# Fixed guarded physical precision study

The original support-adaptive study passed its64 regression tests but failed
the initial N1024/sigma1.25 compact control's full physical FP64 all-dP gate.
The [original note](20261010-sphere-support-adaptive.md) retains that failed
cohort, source and diagnosis. This study has separate research Algorithm IDs
for precision-corrected compact, recompute and adaptive execution. Its control
is explicitly `precision-corrected-compact`; public dispatch is unchanged.

## Fixed repair and unchanged mathematical contract

The exact declared float `precision_norm_threshold=0.001` selects an atom
when either complete original FP32 chart norm is <=0.001 or NaN. Both sides
then use complete physical FP64 geometry/profile/norm and centre-VJP snapshots,
while the ordinary FP32 route excludes that atom. This is disjoint ownership,
not atom removal. Contractions, state, inputs and outputs remain FP32; the
guard, physical scans/refinement and their allocations count in complete steps.
The complete discrete L2 norm floor remains1e-6. Intrinsic S2 charts, chord
distance, Polar/Triweight profiles, live scalar buffers, task-width
stop-gradient and all six requested atom gradients remain unchanged.

Unselected atoms retain inherited FP32 arithmetic. An archived initial-state
analysis found a remaining normal-atom maxabs discrepancy approximately
0.0006248 when selecting solely by0.001. Therefore this fixed candidate may
still fail the unchanged4e-4 gate. The root will first run a small explicit
physical witness on the frozen source. That diagnostic is not the full16
correctness cohort, and does not authorize performance claims or a silent
threshold change. If it fails, preserve its evidence and predeclare any broader
guard as a separate study.

## Fixed protocol and prerequisites

Four cases are N1024/N2048 and initial sigma1.25/sigma3, in that order within
each size. Each runs corrected compact, corrected recompute, corrected
adaptive and dense: all16 isolated workers without filtering. Batch32,
floor(.05*N²) atoms, seed41, FP32 IEEE with TF32 disabled, MSE and fused AdamW
lr1e-4/weight-decay.01 remain fixed. Sigma1.25 is a sharp diagnostic;
sigma3 is the ordinary objective. No gain is transferred between conditions.

Every CST worker checks full sites/all atoms with the unchanged independent
physical FP64 oracle for Y,dX and all-dP initially and after24 actual updates.
Both maxabs and relative-L2 must be <=4e-4. Two warmups, one initial captured
replay and21 timed replays total24; live widths and exact optimizer clocks are
checked. Peak allocated/reserved include capture/replay and exclude oracle
scratch. Total GPU process usage is unmeasured.

All new regression tests must pass with zero skips, followed by all16
verify-only workers on identical frozen source and driver before all16 primary
workers. Partitions must exactly cover the frozen test collection once.
Independent phase/route diagnostics run after primary timing/peaks on a
separate trajectory; three event blocks measure normal preparation with fresh
CSR/tiny forward, guard/physical refinement and complete adaptive forward.
Their medians are not additive and do not reduce primary cost.

Budgets remain900s pool outer,875s driver and350s child per job. Qualification
per case against corrected compact is strictly>3% faster, or at least5%
allocated reduction with<=3% time regression. Any qualifying case requires a
single independent full16 reverse confirmation with every control/condition
retained. Failures and incomplete cohorts remain evidence; no tolerance
relaxation, retuning or selection from partial workers is permitted.

## Provenance and reproduction

Branch `kernel/sphere-precision-adaptive` retains all298 tracked inherited
runtime files byte-for-byte from `3766723347c90cfee88957f79479e02df4825fc0`;
their map SHA256 is
`134053499c2bb0cb64fccaea19749aeab67a2708d9744d3731995ed9ec06c4ad`.
New code is research-only. The catalog is
`benchmarks/cuda/linear/plans-sphere-precision-adaptive.json`; four matching
case declarations and the benchmark-only
`benchmarks.cuda.linear.precision_adaptive_comparison` wrapper are separate
from the original v1 tools.

Ignored `output/sphere-precision-adaptive/` contains the fixed protocol,
cohort driver, assessor, exact collected nodes, CPU synthetic integrity tests,
readiness hashes and reproducible pool commands. Each stage uses
`cohort_driver.py --stage <regressions|gate|primary|inverse> --expected-tests
<frozen stage count>` with `CST_JOB_OUTPUT` provided by the shared pool.
The assessor requires complete regression/gate jobs before accepting primary,
identical source dictionaries and driver/protocol, immutable parent hashes,
exact zero-skip JUnit coverage, valid source/archive/result hashes and the
full ordered cohort. Public adoption is a separate decision.

At this checkpoint only CPU declarations and harness readiness are being
prepared. Actual precision-regression GPU results, the root witness, complete
gate and performance evidence remain pending. No performance improvement or
adoption has been established.
