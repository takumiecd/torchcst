# Matched Sphere / Strip-Torus scaling protocol

`benchmarks.cuda.linear.scaling_comparison` is benchmark-only. It accepts a
registered caller-provided Linear Plan JSON; runtime algorithms and public
dispatch are unchanged. The baseline uses Sphere fused W64/patch32 for sigma3,
Sphere bounded H/G4096 for sigma8, or Torus sparse W C64/S256/T16.

The four fixed cases are N1024/2048, B32, floor(5% N²) atoms, seed41, initial
sigma3/8, FP32 IEEE without TF32. An independent CPU generator creates identical
X/target/fixed dy for both geometries. Dense weights use their own CPU generator.
Every mode owns a fresh initial model and optimizer. All training uses MSE loss,
AdamW lr1e-4/decay0.01, dX and all atom dP, the live activity/width law, moments,
exact step counter and geometry update. Equal sigma is not equal mathematical
work: Sphere retains two S² charts and per-side floors; Torus retains its single
S¹×S² centre-fibre chart, coupled circle radius and one whole-product floor.

Research Graph executes two warmup updates, one explicitly replayed capture
update and21 measured replays (capture recording itself executes no update).
Eager executes three warmup updates and21 measured updates. Each worker gates
initial and24-updated complete-site/all-atom Y/dX/dP against the existing
independent FP64 oracle with fixed dy, bounded atom chunk128, maxabs and relative
L2 <=4e-4; Torus also retains elementwise4e-4. MSE loss values are recorded. Actual/reference Y, dX, all dP, parameters,
moments, counters and MSE must be finite before error comparisons; NaN metrics
cannot pass the thresholds. Final training outputs, gradients and optimizer
state are checked outside timing, including Dense.
The separate20-step same-cotangent state gate compares the public Torch update:
parameters2e-6, Sphere moments4e-4/Torus moments2e-5, exact counters. Its AdamW
state is initialized by an eager update before capture; an explicit Graph replay
and synchronization precede copying the shared starting parameters/moments/clock.
The first failed common GPU preflight ran no primary worker: capturing lazy
optimizer initialization reset moments on replay. Corrective preflight retains
the original thresholds and precedes every primary retry.

Public eager retains synchronous guards and public gradient/state handling.
Research eager/Graph retain the previously declared explicit AdamW proposal
plus typed update Plan boundary. A public/Graph ratio is not a kernel-only gain.
This matched MSE protocol does not retroactively relabel older timings.

Peak allocated/reserved includes capture/replay and only the worker's training
state. Oracle scratch is destroyed before resetting peaks. Process usage and
physical cache residency are unmeasured. Support accounting is independent FP64
raw positive support (including zero-amplitude atoms); exact FP32 boundary
rounding may change runtime overflow counts. A separate fresh diagnostic model
copies the24-updated parameters/moments/clock, executes one capture plus five
instrumented replays, and records forward/loss, backward, optimizer and proposal
components. These phases neither mutate the primary model nor replace/add up to
uninstrumented complete-step timing. Correctness-only workers emit no timing.

Caller cohorts run each plan/mode in an isolated sequential worker, retain all
four primary cases and their baseline/candidates/dense, and stop on any failure.
Independent inverse selection is predeclared only for >3% time gain, or >=5%
allocated-memory reduction with <=3% time regression. Raw drivers/results and
frozen source inventories remain in ignored output. Cohorts match X/target,
fixed dy, atom initialization, full operator declaration and sorted hashes of
all actual model buffers and non-atom parameters (coordinates, chart radii,
Line/Grid start/spacing/pitch and live kernel scalars), with dtype/shape labels.
Raw-byte SHA remains identical to the previous NumPy-backed hash, using a
contiguous torch uint8 view so the benchmark needs no NumPy dependency. Snapshot import paths are
recorded and driver workers use the frozen src tree. Timed samples must be
finite and positive; capture/replay peak byte counts must be positive integers
with reserved at least allocated. These checks precede any primary GPU run; only the orchestrator owns
GPU submission and PR integration. Local CPU skips are not CUDA validation.
