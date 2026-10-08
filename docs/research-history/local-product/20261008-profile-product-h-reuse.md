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
