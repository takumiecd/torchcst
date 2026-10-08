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

## Completed CUDA reference gate

Job l4job-07ad83ce66314367a91e26b9bf6477b1 passed all80 tests (44 CPU and
36 CUDA variants),5.54s; driver exit0/no timeout. Frozen source is
 dd7a8bcd14950a825819c70e7b41db702bfe6235, source archive SHA256
55db9b7081c463d45842697b616fc8cf8c6300cbcb8a12a08e7bbccece9bbcb0;
result archive269c1e94a4af2fc60496dbb62171d8c89574cbc0b29759aba0ce24871ffabb1a.
Receipt/archive hashes and test proof verified. Hardware/runtime NVIDIA L4,
Torch2.11.0+cu130, CUDA13.0, Triton3.6.0. Raw results and the frozen source
remain in the corresponding host-wide pool job directory and ignored evidence.
This validates the exact reference on CUDA, not a fused kernel or Graph speedup.
The checkpoint after dd7a8bcd adds only this evidence note; runtime/tests unchanged.
