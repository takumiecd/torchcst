# Exact sorted-axis Sphere support: reject the runtime candidate

The exact sorted-support candidate passes correctness but loses the complete
training step in all four large cases, with no allocated-memory reduction.
Do not add its runtime, registry, catalog or tests to main. Preserve the source
on `kernel/sphere-sorted-support` at `4b3a785005c647c4397d668ebd4678fdedc73d75`,
with verified raw jobs and recovery proof. This change records negative evidence
only; fused W from PR81 and bounded H from PR80 remain available.

## Complete-step result and decision

NVIDIA L4, B32, 5% atoms, FP32 IEEE, eager public CSTOptimizer AdamW with dX.
Each worker uses the same initial p/X/target hashes, 3 warmups and 21 evolving
updates, with independent full-size FP64 Y/dX/all 6 atom gradients before and
after those 24 updates. Initial and updated correctness pass every case. Dense
uses the same X/target and its own ordinary Linear parameters.

| N | Initial sigma | Fused W ms | Sorted W ms | Change | Bounded H4096 ms | Dense ms | W allocated / reserved MiB |
|---|---|---|---|---|---|---|---|
|1024|3|10.018192|10.661230|+6.42%|18.566222|0.708518|88.486 / 110|
|2048|3|29.979102|30.345146|+1.22%|102.341693|1.095325|307.209 / 334|
|1024|8|129.449341|147.093606|+13.63%|18.258633|0.702329|245.287 / 256|
|2048|8|541.350499|578.430817|+6.85%|102.029227|1.118366|918.409 / 950|

The W allocated/reserved peaks are identical for fused and sorted in each row.
Width3 uses support capacity64; width8 uses capacity256. H4096 is also much
faster and lower-memory for width8. The small N64 runner smoke was negative too:
sorted/fused complete eager time ratio1.03332333 (+3.33%).

The predeclared qualification is at least3% complete-step time gain, or at
least5% allocated-peak reduction with at most3% time regression. Every case
fails that condition; no independent inverse job is warranted. This is one
primary job per large condition, not two independent confirmations. Isolated
preparation or P99 timing was not used to override the complete-step result.
Sorting, gather, scanning, backward and the public optimizer are all included.
The measurements do not isolate which component causes the slowdown.

## Exact mathematical contract and implementation

The existing two explicit intrinsic S² charts, Polar parameterization,
separable Triweight and per-chart L2 floor1e-6 stay fixed. For embedded atom
centre q, actual normalized site s_i, and live precision P:

\[
v_i=\bigl[1-P\sum_d(s_i[d]-q[d])^2\bigr]_+^3,
\qquad
\phi_i=\frac{v_i}{\max(\sqrt{\sum_jv_j^2},10^{-6})}.
\]

In exact arithmetic, v_i>0 implies |s_i[0]-q[0]|<sqrt(1/P). Sorting actual sites
by that ambient component permits an inclusive candidate slab; every site that
can contribute remains in it. The CUDA implementation expands both ends by
32*FP32eps*(abs(q[0])+maxabs(sortedX)+sqrt_rn(div_rn(1,P))). Rounded subtraction,
square, nonnegative sum, product and interval endpoints are covered in the
finite normal reciprocal regime. Nonfinite endpoints, nonpositive/nonfinite P,
or reciprocal below128*FP32tiny take the original complete-vector pack.

Each forward freshly normalizes/snapshots live sites, sorts the axis, gathers
all3 coordinates contiguously, then binary-searches the conservative bounds.
Chunk64 scans still evaluate the original direct3-component chord distance.
They count all positive supports and accumulate the entire raw squared norm.
Only the firstCAP raw entries are temporarily stored with their original site
IDs; they are normalized and used only if the final complete count<=CAP. If
count>CAP, W/VJP use the existing complete original-site traversal. No support,
atom or normalization term is approximated or truncated.

Sorted accumulation changes FP32 reduction order. Within64eps*floor of the
sharp norm floor, the original full-vector pack replaces count/norm/packed
outputs. Backward keeps the original-order normalized site snapshot, q/J,
precision and Polar task Jacobian; later live p/site/radius/scalar changes do
not change retained backward. Algorithm instances have no numerical tensor
cache. Existing width detachment, live-width and snapshot contracts are retained.

The validated fused control is unchanged in meaning. Its W executor, W kernels,
complete-support kernels and fused decoder kernels are byte-identical to the
first fused study; preparation only adds an internal optional packer hook whose
None/default path invokes the existing full pack unchanged. The corrected gate
and primary match all1523 non-driver source-file hashes; only their frozen
`__pool_driver__.py` differs. Source equality proof is archived.

## Validation, provenance and recovery

CPU:1351 passed/2427 skipped; targeted after visibility correction7 passed/
160 CUDA skips. Ruff/AST, 18 Plan declarations/four cases, wheel/sdist pass.
Corrected L4 gate:167 passed/0 skipped/0 failures/0 errors,117.041 seconds;
all small runner records pass. Full FP64 ordinary gates remain max-absolute and
relative-L2<=4e-4. Existing near-sub-floor cases retain absolute FP64<=4e-4 plus
public FP32 atol4e-5/rtol4e-4 checks. No tolerance was relaxed for this candidate.

Additional tests check exact complete support index sets/normalized phi,
uncertain precision fallback, singleton/empty/sub-floor profiles, retained
mutated geometry/parameters, linear Graph live widths and 20 public optimizer
Parameter/moments/step agreement. A full captured optimizer step and GPU process
usage are not claimed by these eager results.

| Job | Scope | Source snapshot SHA256 | Result archive SHA256 |
|---|---|---|---|
|l4job-006defc03b1d4260a047b631142ebcdb|correctness/small runner|a158137a357724913ea3779b40cb864ef755e9dc66d409e3421389d30fc98811|6718c9bd52f4f603adfa86111592c1080ba34e9fc7ce93648c7bb815e7c2994c|
|l4job-45db0bdf2d2b4fb2bd5a48291f6322d5|all4 primary cases|38d53eb131e1a115f9bbf69e16d42a3a23fe7fc5b56ff9999bd401076aeef6d4|08da14cf7048fa43b5abfabdd4a2a158cde3056637ea3a7b982af6fc481b445f|

Runtime: Torch2.11.0+cu130, CUDA13.0, Triton3.6.0, NVIDIA L4. Reproduce from the
source branch/bundle with ignored `output/sphere-sorted-support/smoke_driver.py`
(timeout900) and `measure_driver.py` (timeout1500) through the shared pool.
The queued initial job `l4job-fa343894cffc41728ebaab99643c173a` was cancelled
before slot assignment or CUDA execution for code review. The corrected source
adds one CTA-uniform `tl.debug_barrier()` after rawPhi scatter and before packed
reload or floor replacement; reduction/order/math are unchanged. That cancelled
job is not a GPU failure or an independent run.

Evidence root:
`/Users/ware10sai/.codex/worktrees/profile-product-cuda/torchcst/output/sphere-sorted-support`.
Complete gate/primary/cancelled job copies, per-file SHA256 manifests, source
comparison proof, logs and build outputs remain ignored there. `history.bundle`
verifies a complete history; restoring the source branch into `recovery.git`
returns exactly4b3a785005c647c4397d668ebd4678fdedc73d75 and `git fsck --full`
exits0. Other bundled refs' dangling commits are recovery objects, not corruption.
No source branch or needed ignored evidence was deleted.

Next memory hypothesis is kept separate: N<=2048 site IDs fit signed int16
losslessly, provided every loaded ID extends to int32 before pointer arithmetic.
At A209715/CAP64 the two index arrays' theoretical saving is51.2MiB. That is a
tensor-byte budget, not measured complete-step memory or physical-cache evidence;
no int16 implementation or adoption is included here.
