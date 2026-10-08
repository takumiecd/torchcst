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
