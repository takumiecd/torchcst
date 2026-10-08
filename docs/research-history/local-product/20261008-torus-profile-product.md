# Torus centre-fibre profile product

The owner selected the S1/S2 centre-fibre chord product on2026-10-08 after
reviewing the two proposed distances. This resolves the pending interpretation
in20261008-profile-product-torus-contract.md. The existing radial Torus remains
a separate mathematical control. Only profile-product optimization is pursued.

## Declaration and exact reference

Add presets.polar_torus_profile_product, composition profile_product/revision2,
with two raw profiles ordered circle then S2 section. Revision1 remains the
Euclidean coordinate product. A single S1xS2 Torus Product/Strip Line+2D Grid
Chart supplies input/output layout; circle may be either logical axis. Intrinsic
parameters have two Polar plus three centre coordinates; ambient parameters
have two Polar plus four embedded centre coordinates. Full-chart normalization
uses max(norm(circle)*norm(section),epsilon) once, including support gaps and
partial Strip tiles. Current shared Polar widths are detached in the task VJP
and evolve through the existing activity/update law.

The circle radius R+r*q_a0 depends on section-centre coordinates; Torch autograd
retains both factors and both normalization derivatives. The independent oracle
constructs actual embedded circle/section queries and normalizes all matrix
entries directly. A negative-control test shows detaching circle/section coupling
changes section gradients. Preserve intrinsic and ambient Geometry retraction,
gradient projection, moment transport and maximum arc step. Explicit eager
torch_polar_update now calls the one-chart update for non-Euclidean product
contexts, instead of invoking the two-chart separable update signature. No
optimizer state ownership or geometry law changes.

## Validation checkpoint

New CPU tests44 passed/36 CUDA variants skipped: Product/Strip, reversed logical
axes, intrinsic/ambient, both active/inactive norm floors, materialized/factored/
auto, full values/dX/all source parameters, coupled VJP, initialization/zero
atoms, five AdamW steps with moments/clocks and centre validity, checkpoint,
old/new declaration rejection and strict explicit/general update equivalence.
All FP64 oracle tolerances remain1e-10/1e-12 values and1e-9/1e-11 VJPs.
The earlier complete CPU run passed1240/2069 skipped/18 warnings; subsequent
changes add only CUDA parametrization and two initialization/empty tests.
Ruff and wheel/sdist pass; installed-wheel declarations import with Triton and
benchmarks blocked. The first no-isolation build lacked hatchling; retain its
environment error, then use the standard isolated build successfully.

This checkpoint implements a Torch reference, not a validated CUDA speedup.
Torus Graph updates/fused CUDA and same-protocol complete-step timing/memory
remain required next work. Existing Euclidean spatial primary measurements use
their separate frozen source/worktree and shared single-L4 queue. Raw local
logs/build products live in ignored output/torus-reference/.
