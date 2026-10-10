# Fixed guarded physical precision study

The original support-adaptive study passed its64 regression tests but failed
the initial N1024/sigma1.25 compact control's full physical FP64 all-dP gate.
The [original note](20261010-sphere-support-adaptive.md) retains that failed
cohort, source and diagnosis. This study has separate research Algorithm IDs
for precision-corrected compact, recompute and adaptive execution. Its control
is explicitly `precision-corrected-compact`; public dispatch is unchanged.

## Fixed repair and unchanged mathematical contract

The exact declared float `precision_norm_threshold=0.001` selects an atom
when either complete original FP32 chart norm is <=0.001 or nonfinite. Both sides
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
still fail the unchanged4e-4 gate. The predeclared next step was a small explicit
physical witness on the frozen source; its completed result appears below.
That diagnostic is not the full16
correctness cohort, and does not authorize performance claims or a silent
threshold change. Its failed evidence is preserved; any broader guard requires
a separately declared study.

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

## Frozen negative results and disposition

Source `e0ebfad7654b94e03a59f8eff685302567d229c2` passed the complete CPU suite
with1468 passes and2943 GPU/database skips. Its L4 regression job
`l4job-7c0b298adf794c5f982a5bf4ca79aa80` executed the exact78 collected tests:
77 passed, one failed, zero skipped and zero errors. All17 tests for each
compact/direct/adaptive route and all22 benchmark tests passed. The sole failure
was the shared physical profile VJP probe exactly at its declared norm floor.
The regression source archive SHA256 is
`27f7753fd16df7a3c19548877c2ebc480e8c987a90581ca6fd261b046b6a4452`.

The equality probe's FP64 norm and declared floor are both
`2.8742942542474784e-5`. Implicit conversion of its constexpr floor to FP32
instead produces `2.8742942959070206e-5`, selecting the wrong side of the
normalization derivative. Its observed derivative error195.9824676513672
matches an independent CPU reconstruction of that branch mistake. Below/equal/
above probes remain unchanged. Explicit FP64 floor materialization is a
separate implementation repair; it cannot turn this frozen run into a pass.

For a chart profile \(f_i\), \(n=\sqrt{\sum_i f_i^2}\), and
\(d=\max(n,\epsilon)\), the centre derivative retained by the oracle is

\[
\partial_c(f_i/d)=\frac{\partial_c f_i}{d}
-\mathbf{1}_{n\geq\epsilon}\frac{f_i\sum_j f_j\partial_c f_j}{d^3}.
\]

The equality branch follows the existing `clamp_min` derivative. In particular,
a singleton below the floor can have a nonzero derivative, whereas a positive
singleton above the floor has zero centre derivative.

Independent four-atom diagnostic job
`l4job-311014927b0e49e29ea168c2bccabb57` completed on the same implementation
source. Its source archive SHA256 is
`4e4b40b95d649b5712ce63d1e88deca42cf9088efc1a2a5afdf96743b1744f04`;
the archive differs because it contains a separate diagnostic driver. It uses
four actual original atoms, complete1024-site charts and the archived batch32
X/dY. Independent CPU rederivation matches its physical FP64 dP within
1.5526e-12. This is a diagnostic, not the full16-worker correctness cohort.

| Route | Four-atom all-dP maxabs | Fixed4e-4 gate |
| --- | ---: | --- |
| corrected compact | 0.0006214697069548691 | FAIL |
| corrected recompute | 0.000624807567062291 | FAIL |
| corrected adaptive | 0.0006238538927458848 | FAIL |

The selected low-norm atoms37087 and1552 improve from approximately0.0105
error to at most5.6e-6. However unselected atoms19080 and42223 still exceed
4e-4 on every route. Relative-L2 passes; qualification requires both metrics.
This residual is distinct from the exact-floor implementation defect.

Independent audits verified source/result/archive hashes, all1630 source
files, all298 immutable inherited runtime files, new runtime anchors and exact
zero-skip test coverage. Verified raw logs/tensors/source archives remain in
`~/.local/state/colab-l4-pool/jobs/<job-id>/`; the audit records are in ignored
`output/sphere-precision-adaptive/`. Hardware was NVIDIA L4, Torch2.11+cu130,
CUDA13 and Triton3.6. Both jobs finished without timeout; the pool stopped its
owned runtime after retrieval.

The fixed0.001 candidate is rejected for large-scale qualification. There is
no complete16-worker pass, primary timing, reverse confirmation or adoption.
Its implementation and tests remain on the recoverable research branch;
only the negative research notes are intended for main integration. The floor
repair will receive a separately frozen regression validation. Any broader
precision policy must first be declared as a new study with unchanged accuracy
gates; the next mechanism should establish a complete physical precision
reference before narrowing its coverage for cost.
