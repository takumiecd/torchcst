# H reuse exploration: primary geometry scope

## Owner steering on 2026-10-08

The owner requested an active exploration goal for large Linear intermediate
reuse, then changed its primary geometry scope to an ordinary Sphere chart and
Strip + Torus. This instruction supersedes the Euclidean cohort in the original
goal text. Euclidean is a historical implementation/control, not the primary
optimization target or sufficient evidence for adoption.

The stated motivation is that atom centres can escape to infinity in Euclidean
geometry, whereas Sphere and Torus have compact centre domains. Preserve the
existing geometry projection, retraction and optimizer vector-state laws.
Compact geometry alone is not a proof that every centre has sampled support.
Do not add Euclidean clipping as a substitute for the selected geometries.

No new Euclidean diagnostic job, kernel mutation or GPU allocation had occurred
before this steering. The new branch started at main 09dec7486e71f308f4442fb4c31a211769ad093c.
Existing square-strip evidence is retained unchanged.

## Existing Torus contract and first target

Start with Strip + Torus profile product revision 2, selected by the owner and
implemented in the Torch reference. Geometry is S1 x S2, not a conventional
two-dimensional donut. For atom centre (theta_a, q_a) and radii R, r, define

\[
d_c(i;a)^2=4(R+r q_{a,0})^2\sin^2((\theta_i-\theta_a)/2),
\qquad d_s(j;a)^2=r^2\|q_j-q_a\|^2.
\]

With shared current Polar width sigma_a,

\[
u_a(i)=f_c(d_c(i;a)^2/\sigma_a^2),\quad
v_a(j)=f_s(d_s(j;a)^2/\sigma_a^2),\quad
K_a(i,j)=A_a\frac{u_a(i)v_a(j)}
 {\max(\|u_a\|_2\|v_a\|_2,\epsilon)}.
\]

Circle and section can exchange logical input/output roles. Their Cartesian
site indices permit norm factorization and H = X v followed by H u^T, while
parameter derivatives remain coupled: section-centre motion changes the circle
radius. Floor once over the whole chart. Do not independently floor factors,
drop radius derivatives or substitute the old joint radial Torus distance.
Widths remain detached in the task VJP and evolve through the Polar update.

The old Strip + Torus fused kernel is a separate radial/direct-activity control;
its timing is not an implementation comparison of the same new operator.
PR #74 contains the correctness-verified intrinsic Torus Graph update baseline.
It remains draft; complete-step performance/memory are not yet measured.

## Sphere contract investigation

The existing ordinary Sphere Product chart lifts Cartesian chart coordinates
gnomonically. Its ambient chord distance generally depends jointly on both
logical site indices. Public profile-product revisions 1 and 2 currently accept
Euclidean and S1 x S2 Torus respectively, and do not define Sphere composition.
Changing only geometry is therefore not an implementation of Sphere profile
product with the old independent-axis norm or H factorization.

A candidate on S2 is a product of three ambient-coordinate profiles:

\[
K_a(i,j)=A_a\frac{\prod_{k=0}^{2}
 f_k((s_{ij,k}-c_{a,k})^2/\sigma_a^2)}
 {\max(\sqrt{\sum_{i,j}(\prod_{k=0}^{2}
 f_k((s_{ij,k}-c_{a,k})^2/\sigma_a^2))^2},\epsilon)}.
\]

Centres remain on the Sphere; this is a proposed new composition, not an
existing declaration. It is orientation dependent and generally not separable
by logical input/output indices. Its whole-chart norm must not be replaced by
three independent norms. No Sphere metric or declaration is adopted by this
note. Investigate the construction before kernel integration; retain the
existing Sphere chord/radial operator as a mathematical reference.

## Comparison requirements

Keep saved versus recomputed versus tile-reused H/G comparisons distinct from
H-route versus generated-W/GEMM comparisons. Keep a single operator chart,
global normalization, live geometry, all atom gradients and optimizer moments.
Use separate instrumented Graphs for diagnostics and uninstrumented complete
training steps for adoption, including capture/replay allocated and reserved
peaks. Do not infer hardware traffic or cache residency from component timings.

The large square target remains N inputs and N outputs, initially 1024 and 2048,
with B32 and about 5 percent atoms, FP32 IEEE and a matched dense control. It is
not a 64-output Strip target. Define the Torus-specific radii, site spacing,
support-width cohort, finite run budgets and oracle gates before any new
performance submission; Euclidean rho values do not automatically define an
equivalent curved-geometry workload. Independent reverse-order confirmation is
required before a positive performance disposition. Preserve negative evidence.

This checkpoint records scope and mathematics only. No new performance result,
Sphere API, CUDA implementation or public dispatch policy is claimed.

## First implementation checkpoint: Torus block contractions

Add research_torch_torus_profile_product_chunked/v1, not public registration.
It accepts intrinsic FP32/FP64 Product/Strip centre-fibre Triweight/Gaussian
profiles, axes up to 2048 and batch up to 64. Each block enumerates every
circle and section site for exact normalization. Transient factors are not
saved across the forward/backward boundary. H/save keeps B*A values; H/recompute
recomputes the input contraction during parameter VJP. W uses the same factors
to assemble the aggregate matrix and saves W for dX, then uses dW for parameter
VJPs. All routes recompute factor VJPs in bounded atom blocks and retain both
normalization derivatives and section/circle-radius coupling.

Snapshot canonical source, current width precision, amplitude maximum, live
geometry and freshly built axis queries. No geometry state is frozen across
replays. Older VJPs retain their own forward state after source, pitch, radii
or amplitude changes. The source law still detaches width in task gradients.
No approximation, truncation, sampling or new coordinate update is introduced.

The first GPU submission is a correctness gate only, with a predeclared 600s
driver budget and a 300s pytest child budget. Run test_torus_profile_product.py,
test_torus_profile_product_graph_update.py and the new chunked tests together.
These cover independent FP64 physical full-matrix values/dX/all source VJPs,
both floor cases, Product/Strip/reversed axes, partial atom blocks, old snapshots,
empty atoms, separate input/parameter gradient requirements, retained tensor
lifetime, and 20 captured steps with live geometry, evolving widths and AdamW
moments. No performance repetition or numerical tolerance change is authorized
by this gate. Diagnose implementation failures before an explicit corrected
submission and retain the original failure evidence.

The large performance cohort will use equal N input/output lengths, circle
Strip tiles of 64/pitch68 with unit arc spacing, a centred section Grid32x32
for N1024 and32x64 for N2048, and minor radius sqrt(N). Major radius is
(ceil(N/64)*68)/(2*pi), giving a single circular period around the Strip.
The initial widths are physical sigma3/8, not a claim of equal Euclidean rho
or equal support counts. Shared sigma bounds1..16, B32, seed41, A=floor(N*N/20),
Triweight/Triweight, global floor1e-6 and AdamW lr1e-4/decay0.01 stay fixed.
Large oracle and finite performance budgets must be recorded before submitting
that later stage; correctness of this small gate does not prove large speed.

## Verified first gate

Measured source ee6bfbcb, job l4job-c6063ff6924141ccbcbbb096423e7d3c:
174 passed, zero skipped,15.39s, driver exit0/no timeout. NVIDIA L4,
Torch2.11.0+cu130/CUDA13.0/Triton3.6.0. Source archive SHA256
f31415ab7194f2a3fbb0d33ded01f3d16632666645b05113ce64fd4a2288d548;
result archive2dac7bb49f338a8f82df46805cbcc3c528acee4089997663de9419cd37f9b879.
All five result manifest entries and the receipt/archive hash were verified.
CPU complete suite at this checkpoint:1302 passed/2184 skipped/18 warnings.
Raw gate evidence is in ignored torus-profile-product-h-reuse-20261008/gate-l4
and the pool job directory. The supervisor finished and its owned slot stopped.
This is correctness evidence, not a measured speed or memory improvement.

## Large comparison preparation and predeclared budget

Use the existing Linear runner, new polar_torus_profile_product_strip fixture,
the four torus-profile-product-square-strip-{1024,2048}-sigma{3,8} cases and
plans-torus-profile-product-chunked.json. All three controls have atom_chunk1024:
h-saved, h-recompute, w-gemm. No contraction/chunk tuning based on primary times.
Select --polar-update torus explicitly. The benchmark update uses the verified
research update Plan with the common binding/Dispatcher; the public optimizer
wrapper remains eager. Adapter/protocol revision8 distinguishes intrinsic
Torus retraction and physical-fibre oracle from Euclidean revision7. Reject
crossed optimizer policies, oracle scopes and update identifiers. Generation
reconstructs five parameters and the actual Torus chart.

The FP64 oracle enumerates every circle/section site for every performance atom
in bounded blocks. It constructs actual embedded fibre queries and their chord
distances independently of the runtime distance expressions. Full-matrix norm
versus factored norm is checked separately. Y/dX/all source VJP checks retain
the 4e-4 max and relative-L2 gate and the 2e-6 optimizer-update gate. The
performance fixture additionally requires nonzero centre derivatives. Record
all source/result hashes, initial input/parameter hashes and full worker JSONs.

Predeclare two primary L4 jobs, one per N, each with a2100s driver budget:
300s correctness pytest budget and850s runner subprocess budget per sigma case,
plus setup. Each job reruns the complete Torus/chunk/fixture gate before timing.
Each case uses B32, warmup5, rounds21, seed41 and the unchanged optimizer stated
above. Record isolated-process eager and uninstrumented Graph complete-step
times and capture/replay allocated/reserved peaks; process usage is unmeasured.
--phase-diagnostics captures a separate external-event Graph after the primary
measurements and cannot replace those times or memory peaks.

Reserve at most two independent reverse-order confirmation jobs with the same
2100/300/850s budgets if a primary route has at least3 percent time improvement
against a named same-operator control or lower allocated peak without a time
regression. Reverse all three CST controls; dense remains last in the existing
runner. Confirm all four N/sigma cases and matching fixture/source/update
conditions, not only the favorable rows. Preserve all negative and partial
results. Numerical tolerances and these budgets are not increased in response
to an unfavorable timing or failure. No public dispatcher adoption is implied.

## Pre-timing fixture failure and causal correction

The first two primary attempts at source ec2ce997 passed196 tests with zero
skips, then failed the all-atom nonzero centre-gradient gate at sigma3 before
any timing. N1024 job l4job-d3fb0434ca5445579ee597526e3dab78 used28 rounded-up
driver seconds; N2048 job l4job-e403b7465c444b76922a2e309ba0d16f used27.
No sigma8 case was attempted. The result manifests and receipt hashes were
verified; full failures/source/proof files are preserved in the pool and ignored
failed-primary-{1024,2048} evidence directories.

The Line constructor with spacing1 defaulted to a centred origin (-511.5 or
-1023.5), but the seeded circle centre initializer used zero-origin Strip site
indices. Rotation placed some centres in the wrong part of the inter-tile gaps.
Independent physical-query CPU diagnosis of2048 atoms found54/51 near-zero
circle-centre derivatives, with zero empty atoms. A single live circle sample
has a constant normalized factor and zero circle-centre task derivative even
when the section has multiple samples; the old both-factors singleton count
did not diagnose this. This is a real normalization behavior, not a CUDA error.

Correct the fixture's circle Line to low0/highN-1, matching the initializer and
intended physical Strip layout. No kernel, geometry metric, norm floor, width,
update, numerical gate or route recipe changes. The same independent2048-atom
diagnosis then has0/0 failing centre components in both sizes. Add regression
checks for the zero origin and multiple live circle samples at sigma3.

Failed source/result archive hashes, N1024:
cca5ef8ceedaedfcec9e56565084d234853052d32c6dbebe7c3dbdcc5efcf72d /
73527a58ffb48aab6176a61c14d945e012110b9314380a5345db956d77f8053d;
N2048:b0243a13bb197a52e993cb07c180f8a396901e938d08b3caa49155fe6e06ca24 /
f58f01d6a3b0331ebf78ce90b0f23080403444d1451d815b90474cd7507a7ea0.

Explicitly resubmit corrected source using only remaining predeclared budget:
2072s for N1024 and2073s for N2048. Conservatively subtract the full failed
driver time from both the300s pytest budget and850s sigma3 runner budget too
(272/822s and273/823s). Sigma8 keeps its unused850s budget. Inverse confirmation
budgets remain unused. This is a fixture correction before timing; no timing
was discarded, selected or retried and no numerical tolerance is relaxed.

## Large-circle FP32 numerical failure and correction

Corrected-origin source0369c08b passed198 GPU tests, zero skips, in both jobs,
and passed the full-atom nonzero-centre gate. It then failed the unchanged
sigma3 Y comparison before timing:3/32768 mismatches at N1024 (maximum reported
absolute error9.1224e-4),161/65536 at N2048 (1.7046e-3). Sigma8 was not attempted.
Jobs l4job-0f7f52b4841844b18841ba4bd225e72a and
l4job-5d4b439591c640ffa4066c19d4d7bc3a consumed30/28 rounded driver seconds.
Cumulative prior consumption is58/55s. All9 manifest entries, source archive
and receipt/result archive hashes were verified and saved in ignored
failed-numerics-{1024,2048}; neither attempt produced timing evidence.

N1024 source/result hashes:
840bc9cf0b935a90ab20689d8e7c326731ed209ddf74f01aa695d214d805e2bd /
b4662cd26adfdfab6afdad1e4978e332909dd1ec67082b1cc738d8f5a5bc0683.
N2048:2ea31ec298a93c9e37c5caa64fdad85f6a266c06daff3a0c5beec8fe987d38a9 /
5c416d9aee48bc3f0947fe76106f100c8e72b90f0daa0ca45243510ca6b21359.

A2048 CPU factor comparison isolates the dominant error to subtraction of
rounded FP32 angular quotients, magnified by a large circle and thin support.
The section difference is already a stable physical chord difference. Snapshot
circle queries and form the angular difference in FP64, reduce to[-pi,pi], then
cast to the declared dtype before sine/profile evaluation. The measured
scaled circle factor maximum discrepancy against FP64 decreases from
1.3450e-5/2.7102e-5 to2.0623e-7/2.1344e-7 at N1024/2048. Add FP32 values and
centre-gradient regressions at the periodic seam and large angles. Profiles,
norms, contractions, retained H/W, parameters, updates and tolerances stay FP32
in the performance cohort; only angular difference preprocessing uses FP64.
This bounded reference implementation makes no claim of fast FP64 geometry or
public dispatch adoption. All three routes use this same numerical law.

Before a corrected primary submission, retain the unchanged gates and subtract
all prior consumed time:2042/2045s driver,242/245s pytest and792/795s sigma3
runner; sigma8 remains850s. No measurement was selected or rerun and no gate,
cohort, atom count, norm floor or optimizer contract is relaxed.

## Verified N1024 primary, confirmation pending

Frozen runtime0aeeff35dc32984f1ea21c1418427c7b18a78ecf: local complete CPU suite
1322 passed/2192 skipped/18 warnings, Ruff/diff checks and wheel/sdist build
passed. GitHub CPU Actions run37719038254/job113122163314 passed. N1024 primary
job l4job-465a50431de74ee5b1737bc08944dbf4 passed202 GPU tests without skips and
both complete cases, including the independent full-A FP64 Y/dX/all-parameter
VJP, strict nonzero centre derivatives and exact public optimizer proposal.
The max Y/dX/dP discrepancies over the three routes and both widths are below
1.3e-5. Every52428 atom changes width; initial/final empty/floor/singleton-both
counts are zero. All seven workers per case pass submission checking.

NVIDIA L4/Torch2.11.0+cu130/CUDA13/Triton3.6, uninstrumented Graph medians ms;
peaks include capture/replay, decimal MB. No total process measurement.

| sigma | route | Graph ms | eager ms | allocated MB | reserved MB |
| --- | --- | --- | --- | --- | --- |
|3|H saved|94.419|236.477|135.943|436.208|
|3|H recompute|93.789|237.410|129.363|402.653|
|3|W + GEMM|122.217|230.535|137.489|419.430|
|3|dense|0.0825|0.671|51.383|111.149|
|8|H saved|96.862|239.255|135.943|436.208|
|8|H recompute|96.492|236.038|129.363|402.653|
|8|W + GEMM|127.324|228.867|137.489|419.430|
|8|dense|0.0837|0.707|51.383|111.149|

Same seed/source/initial parameters/inputs/update across routes were verified.
The reference enumerates dense axis factors; its very large gap to dense is a
negative production-speed result. H recompute saves6.58 decimal MB relative to
H saved with only a sub-percent primary-time difference. H routes are faster
than generated W here; this cannot be generalized to the prior Euclidean CUDA
cohort or a fused Torus implementation. In the separate instrumented Graph,
H backward is70.9–73.6ms and optimizer0.23–0.24ms. This points to factor
regeneration/VJP/contractions as the next diagnostic target, without proving
hardware memory traffic or an isolated FP64 penalty.

The N1024 primary satisfies the predeclared confirmation trigger: H versus
W exceeds3 percent time improvement; H recompute also has lower allocated peak
without primary Graph regression. Queue both N1024 and N2048 independent jobs,
reverse all three CST routes, preserve dense last, repeat all four cases with
unchanged runtime0aeeff35 and2100/300/850s budgets. No later source tuning is
included. Jobs l4job-24e1a0f029b44aa1901733fe8e31c768 and
l4job-6cde9e65e784419587633b753c457388 are pending, as is N2048 primary
l4job-5ab0622e3f1f43fa8a2b39899e706745. Do not adopt a positive ranking until
these are checked.

N1024 primary source/result archives:
9fe0ecc95667238280b2bb4f99b3b8e9ddd5fefb8cf8373f0b4323c5491de2ba /
494f15bc64a00c982bb6a424006921d9df4466f2a9d6f8aa73def5d099273a90.
All25 result entries, source archive and receipt/result archive hashes verified.
Raw primary-1024 results and verification summary are preserved in ignored
H-reuse evidence and the pool job directory.

PR #74 merged through GitHub at772b7cdcaa4a2a307a7a02f0dec36e58d2d98ca4;
local main was fast-forwarded afterward. It integrates the correctness-verified
research update foundation, not an updater speedup or public fast path. Draft
PR #75 contains this bounded contraction/reference protocol and remains pending
complete disposition. No research worktree, ignored evidence or branch has been
removed during the live comparison.

## N2048 primary acquisition failed; no retry

N2048 primary l4job-5ab0622e3f1f43fa8a2b39899e706745 did not return a completion
acknowledgement or result archive. Its CLI request and supervisor remained live
beyond the driver budget in wall-clock time. After a documented graceful
SIGTERM to the supervisor (preventing subsequent dispatch), a read-only server
session listing authoritatively reported no active sessions and pruned the
stale owned session cst-pool-f5756d821539-1. Only then was the stale local CLI
request sent SIGTERM. The supervisor terminated with colab exec exit-15;
normal pool recovery marked the interrupted job failed and all slots stopped.
No force kill, queue/slot-state edit or replacement primary submission occurred.
The source archive SHA256 is verified:
a8a35ef2cbd8db56d40e115f7a65d1228c002605e20f499815d34a2a7d855d2a.
There is no result archive, receipt or full-A performance correctness proof.
The cause of remote session disappearance is unknown; do not label it an OOM,
numerical error, measured kernel timeout or measured speed regression.

Preserve spec/transport log in ignored lost-primary-2048, server-assignments.log,
supervisor-stop-request.json and stale-transport-stop.json; original source and
job records remain in the pool. Treat its full remaining2045s primary driver
budget as consumed; do not extend/reuse it. The two already queued inverse jobs
retain their predeclared2100/300/850s budgets and frozen runtime0aeeff35; they are
not relabelled replacement primary runs. Resume only after verified recovery
and stopped slots, still with one L4. A successful N2048 inverse can provide a
first complete correctness/time/memory observation, but cannot prove independent
N2048 replication or support a positive cross-size performance adoption.
N1024 can still be independently confirmed against its retrieved primary.

PR #75 remains draft. Whole-cohort positive adoption is unproven. The preserved
negative/acquisition evidence must remain in the final disposition rather than
being replaced by favorable rows or fresh primary budgets.
