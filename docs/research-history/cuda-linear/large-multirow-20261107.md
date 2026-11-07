# Large-width multirow Linear comparison

**English** | [日本語](large-multirow-20261107.ja.md)

The user moved the primary performance question to widths 8192 and 16384.
The preceding small FFN record is [separate](multirow-packed-20261107.md): its
256 token rows and model width256 must not be confused with the large widths.
This study measures square CSTLinear, not a full width16384 language model.

Fixed cohort: A10080GBPCIe MIG3g.40gb, FP32 IEEE/TF32 off, R=256 resident rows,
N=8192 and16384, A=8N (65536/131072 Atoms), modes8/base1/merge2/spacing1/origin0.
The two-chart kernel is minimum-001: radius[.5,50], birth.5, upper_floor1,
rational_zero_opening tau.001, q0 at initialization. Use public Xavier gain1
(there is no ReLU in this isolated Linear). Centers and amplitudes are actual
public initialized Parameters. No Atom-count, learning-rate or envelope sweep.
This is fixed-c linear-Atom-count performance, not an accuracy/lower-bound study.

The original small Plans stay narrow and unregistered in production. Separate
large row8 and FP32-parallel IDs admit square N2048/8192/16384, R1..256, A4..8N;
small A is for indexing/independent quadrature only. Actual measurements use8N.
Old production admission, Graph rejection, support, normalization and the native
Polar finite_chord update remain unchanged. The CUDA arithmetic body is reused;
only the explicit large scope is added. Graph and other hardware are unqualified.

Large initialization evaluates the same sign-energy target through local cell
mass norms; repeated periodic images are added before squaring. Widths<=1024
keep the old factor-norm calculation. Bounded CPU local Fourier reference checks
all sites/all Atoms in64-Atom chunks. Dense starts from the same effective CST
operator, exported on CPU by local atom-major FP32 scatter (chunk64,
maximum_outer_product_values1048576). This explicit export policy differs in
rounding order from historical identity/GEMM export. Dense W is initialization
only; CST execution does not store a dense W/dW. Preserve input tensors and hashes,
source, export policy and initial Dense W content hash.

Before timing, run the existing multirow/global regression tests and new wide
index/tail tests, including a signed32-exceeding owner key atN16384. Qualify
N2048/A16384/R256 with full Y/dX/all-dP against public CPU FP64 local Fourier,
and an eight-Atom independent polynomial-quadrature supplement (public width
decode). Reuse the existing Linear runner's three matched-state CPU FP32 update
checks and same-gradient GPU optimizer-wrapper byte gate. The new fixture uses
the user-authorized Parameter atol3e-5/rtol3e-4; existing full operator gates
(max absolute and relativeL2<=5e-4), gradient/Adam checks remain unchanged.
Both measured widths also check all-Atom VJPs before and after their actual updates.
Stop on failed mathematical gates, nonfinite, OOM or timeout; retain failures.

Use the existing `benchmarks.cuda.linear.run` workers, no new training/timing
loop. The new fixture is explicitly research-only and rejected by the database
adapter, like the original radius fixture. Source hashes include the bounded
reference/exporter. Run each arm in a fresh process: FP64 row8, FP32 saved+
parallel and Dense. OrderA: widths8192→16384, armsrow8→FP32→Dense;
orderB: widths16384→8192, armsDense→FP32→row8. One allocation, two observations
per arm/width; do not treat them as independent hardware replication.

Each observation executes10 warmups plus51 timed, synchronized, complete steps:
zero_grad, clear dX, forward/MSE/backward including dX, fused AdamW/native Polar.
Fixed resident random x/target (seed41), lr1e-4, weight_decay0, biasFalse. Data
transfer, CPU initialization/export, initial proof, compile, updated-state proof
and hashing are outside the step clock. Peak allocated/reserved includes all61
training steps; total GPU process usage is unmeasured. Preserve all51 samples,
initial/updated radius distributions, final Parameter/gradient/Adam/clock Tensor
hashes and initial conditions. Report both orders and setup including initial
proof separately. Do not join these isolated-Linear timings to whole-LM timings.
These updates do not establish convergence or a large-model accuracy advantage.

This is preparation and a fixed protocol, not a completed large-width result.
