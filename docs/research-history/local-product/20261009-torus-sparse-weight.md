# Torus centre-fibre sparse W candidate

The mathematical operator remains the intrinsic S¹×S² centre-fibre profile
product on one circle-output Strip, Triweight/Triweight, live Polar width,
combined complete-chart L2 norm with one floor. This research candidate packs
complete raw supports and norm derivatives once, assembles W through sparse
atom patches, computes Y/dX/shared dW with IEEE FP32 cuBLAS, and contracts every
atom VJP against dW. Coupled section q0 / circle-radius differentiation remains.

For regular circle sites a conservative angular chord bound defines a compact
index window. Positive spacing, full tiles, nonoverlapping pitch, sufficient
capacity, and a period/pitch discrepancy below one spacing are required. The
bound includes the entire pitch gap and two rounding indices; duplicate windows
are disabled when N is below capacity. Unsafe geometry uses complete-axis
packing. Packing overflow uses complete-axis contraction, never truncation.
The S² axis is always fully enumerated during packing. Forward snapshots own
p/scalars/queries/support and W; no forward state lives on Algorithm instances.

Predeclared stage: one 900-driver-second correctness job, 40 GPU tests plus
full-shape B32/N1024/2048/sigma3/8 all-atom physical FP64 oracle and public
optimizer update checks. max absolute, relative L2 and existing elementwise
4e-4 gates; public parameter update 2e-6; twenty captured evolving steps with
moments 2e-5. Gate failure prevents performance measurement. New corrective
source, if needed, is a separate job; failures and scope remain recorded.

Only after correctness, one primary performance job up to 1500 driver seconds
for all four cases, old W+GEMM baseline, sparse W, dense. Existing runner retains
21 full evolving-step samples, eager and Graph, capture/replay peak allocated
and reserved bytes. Process GPU memory is unmeasured. A separate inverse job
(up to 1500 seconds) is eligible only for >3% time improvement or >=5% allocated
memory reduction with <=3% time regression. CPU declarations are not GPU PASS.
Public default selector remains unchanged. Raw drivers/results belong in
ignored `output/torus-sparse-weight/` and the shared pool job directories.

Initial source f6a140b7 gate job `l4job-9e2177a6458a458995b2d795db1e81b7`
failed in driver setup before importing Torch or running any GPU test. The
Colab Python3.13 environment cannot bootstrap venv ensurepip; driver time
0.164 seconds. No numerical or performance result exists. Source archive
`fa62f19dbf1bed3715462043581fc842614bef1e97cce0894787361a5cbc497c`,
result archive `d691537135094321aacf98a8b948203c1a4ab386a5880453953d1628f62a25cb`.
Complete job copied to ignored `output/torus-sparse-weight/failed-setup-v1/`,
all files and archives hash-verified. Corrected driver uses installed pytest
first; only if absent, creates a no-ensurepip job-local environment and directs
pip explicitly to that environment. Corrected gate has 899 seconds, retaining
all tests, four cases, oracle and tolerance gates. This is a setup repair,
not a retry justified by numerical or performance outcomes.

Metadata checkpoint e1f6506 additionally rejects TF32 for cuBLAS contractions;
cohort already sets TF32 false and numerical execution files are unchanged.
Local CPU full suite: 1350 passed, 2418 skipped, 18 warnings,26.44 seconds;
wheel/sdist offline build and Ruff passed. CUDA skips are unvalidated.

Setup-repaired source3992aac job `l4job-474d449075174d26abcc402c59115c9d`
reached GPU tests:37failed/4passed/0skipped in11.50 seconds,16.046 driver
seconds. Nonempty cases failed Triton compilation because the circle pack's
conditional branches reused vector names at different CAP/full-axis shapes;
no numerical comparison or full-atom case completed. Entire job is retained in
`output/torus-sparse-weight/failed-compile-v1/` with every file hash verified.
Correction assigns branch-exclusive names to full-axis vectors, preserving
operations, norm, support, derivatives and all gates. Corrected gate has882
seconds, within the original900 after both failures; no performance runs have
started. Failed source/result hashes remain in the copiedspec/receipt.

Compiled source5bf4db4 gate `l4job-14ca8d30b90747e68d84365b13ea4c5b`
completed33tests but failed all8 large-axis window/overflow tests. The first
seam case returned incorrect Y (maxabs0.7653142); subsequent cases encountered
illegal GPU memory accesses. The GPU clock ran41.948 driver seconds,
37.33test seconds. No full-atom or performance cohort completed. Wholejob is
hash-verified and retained at `output/torus-sparse-weight/failed-seam-v1/`.
Triton's signed integer remainder retained negative indices across the seam,
unlike Python modulo. Corrected packing normalizes remainder to [0,N) before
any query/load/scatter. All existing seam/overflow/math tests and tolerances
remain; no result is selected from the failed numerical cohort. Revised gate
has840 seconds, deducting failures from the original900; performance remains
blocked until correctness passes.

## Correctness passed before performance

Final measured source `7a970a9954217d39d17664b8a5a6423a6b088fa2`, job
`l4job-b5400ed460f64e4f901b5beb1c3b9468`:41tests passed,0skipped,50.93seconds,
96.590driver seconds. All four full-A B32 cases passed physical FP64 Y/dX/allP
and public update checks. Worst maxabs: Y2.141064e-5, dX3.039439e-5,
dP2.973620e-5; all relative L2 below8.55e-7. Four update comparisons are exact.
Twenty captured evolving updates compare parameter2e-6/moments2e-5 and live
geometry/width. Scalar floor, empty atoms, full-axis overflow, retained forward,
independent gradient requirements and periodic seam are covered.
Source archive `920444dfbf49d256aa000837b7d11c620937a7de3e960976c22f833ed3b0366e`,
result archive `236754960eccd7c69c3e245c8a7c79940ad22154873e45373878057704dd4bb0`.
Complete job copied/hash-verified under ignored `output/torus-sparse-weight/gate-passed/`.

## Primary complete-step cohort

L4/Torch2.11.0+cu130/CUDA13.0/Triton3.6.0, B32, FP32 IEEE,5%atoms,
one intrinsic circle-output Strip, live widths. Existing Torch factor/W+GEMM
is the baseline; candidate CUDA sparse factor/W assembly/allP VJP uses cuBLAS
for Y,dX,dW. Values are ms; memory is binary MiB. The independent inverse is
pending, so these are one-job observations and not a confirmed winner.

| N | sigma | baseline Graph | candidate Graph | candidate eager | dense Graph | baseline allocated | candidate allocated | candidate reserved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1024|3|121.959247|11.676789|13.670464|0.083464|131.120|178.636|370|
|1024|8|124.136649|15.901810|17.714389|0.083299|131.120|178.636|370|
|2048|3|1363.371469|47.173414|49.453839|0.590511|258.314|611.955|1244|
|2048|8|1389.783154|83.660056|87.706464|0.590886|258.314|611.955|1244|

All four time improvements exceed the predeclared3% gate, qualifying one
inverse cohort. Allocated memory increases36.2% at1024 and136.9% at2048;
this is a speed/memory tradeoff. Dense remains140..191times faster at1024,
80..142times faster at2048. Every method's time/memory and negative failures
remain recorded. Memory includes CUDA Graph capture/replay; GPU process usage
is unmeasured. Separately instrumented phase medians are diagnostics, not
components to add or subtract from complete-step timing.

Primary `l4job-ba408eb87c2b4e038553de2333a9dab6`,428.515driverseconds,
source `d5c40731e45bd7f885889ae05420ca00511a7f6eccbbdf17b121b90e12daee0e`,
result `df1270e7ed70f504f8ac187adc320a2e25ee247db30368f686e1a66a1512f9a2`.
Each case passes both plans' all-atom oracle/update before timing21samples.
Every initial p/X/target hash matches within case; every atom's sigma changes
through current Polar activity, maxDelta0.01407(sigma3)/0.10873(sigma8).
Complete job under ignored `output/torus-sparse-weight/primary/`, files/archive
hash-verified. Inverse job `l4job-52d34b291b4d4c48882c7151e09242c2` is frozen:
all1515source file hashes match primary, including driver. Only plan order in
external Case changes; dense remains last. No source or protocol is modified
for inverse. Each job uses the same declared1500driver-second cap.

## Shared numerical calculation placement

After both measured anchors were frozen, exact bounded trigonometry/atom/fibre
helpers moved from onchip/kernels.py into operation-local `_shared/profiles.py`.
Onchip and sparse W import these actually shared helpers lazily. All seven
numerical helper/forward/backward ASTs match measured7a970a9 exactly. This
structural proof does not replace122-test GPU integration regression and all
four full-atom oracle cases for the relocated source. Public default dispatch
is unchanged. Source anchor7a970a9 and all failure/gate/performance raw snapshots
remain recoverable in this named research branch and ignored evidence.

For report reuse, raw circle/section profiles u/v obey
D_a=max(sqrt(sum_j u_aj²)*sqrt(sum_i v_ai²),epsilon),
W_ji=sum_a amp_a*u_aj*v_ai/D_a. For common centre coordinate c and shared dW Q,
t_a=sum_ji Q_ji*u_aj*v_ai and

    grad_c = amp_a/D_a * [ sum_ji Q_ji*(du_aj/dc*v_ai+u_aj*dv_ai/dc)
               - 1[norm_u*norm_v >= epsilon]*t_a
                 *(sum_j u_aj*du_aj/dc / norm_u²
                   +sum_i v_ai*dv_ai/dc / norm_v²) ].

The q0 term includes both circle-radius and section derivatives. Empty norms
use the inactive floor branch and zero raw derivatives; no0/0 correction is
formed. Shared parameter dependence does not invalidate the factorized norm
identity for each fixed atom, but both terms must be included in its VJP.
