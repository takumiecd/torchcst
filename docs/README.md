# Design document status

The executable contract lives in `src/torchcst/` and `tests/torchcst/`. Design
notes explain why that contract exists, but several documents intentionally
preserve superseded proposals. Use this index before reading pseudocode as
current API documentation.

## User guides

- [`pullback-adam.md`](pullback-adam.md) — diagonal pullback geometry,
  parameter-space versus tangent-space moments, step calibration and the
  structural forces (rent / pair repulsion / wall), optimizer ownership, and
  the moving-frame limitation. Its final "Chart coordinates" section covers
  `ChartPullbackAdam` (AM3, 2026-08-23): the chart-point metric as a sum of
  per-incidence pullbacks, shipped in `torchcst.optim.chart` with the
  autograd-oracle mechanism tests in `test_chart_pullback.py`.
- [`cst-optimizer.md`](cst-optimizer.md) — `PullbackConfig` and `CSTOptimizer`:
  one owner per trainable parameter across a model with many CST sites in many
  layers, deterministic per-site overrides, shared-chart ownership, and why the
  coordinator is not a `torch.optim.Optimizer`. Current API.
- [`policy-authoring.md`](policy-authoring.md) — compose a policy, add a custom
  backward statistic without editing the engine, and test both capture modes.
- [`capture-lifecycle.md`](capture-lifecycle.md) — hook timing, deferred versus
  inline-reduced measurement, memory trade-offs, and the structural boundary.

## Current design reference

- [`framework-design.md`](framework-design.md) — current responsibility map,
  composed versus whole policies, storage behavior, implemented operation
  matrix, and extension checklist.
- [`winning_recipe_design.md`](winning_recipe_design.md) — v4 architecture and
  accepted invariants behind the current implementation. Some planned modules
  and distributed paths described there are still future work.
- [`fast-construction-family.md`](fast-construction-family.md) — contract for
  the cSFW / cVP methods and the Refit op (FASTCON arc). Shipped 2026-08-01;
  each section carries an "実装（2026-08-01）" note where the drafted contract
  and the implementation differ (`cELM` is still unimplemented, and the drafted
  `polish_iters=3` default lost to `0` plus periodic backfit).
- [`family-implementation-brief.md`](family-implementation-brief.md) — the
  handoff spec that fast-construction was built from: frozen defaults from the
  FC-0/1/2 experiments (§2) and the implementation decisions recorded while
  building it (§6). Read it before changing a family default.

## Historical decision records

- [`reviews/winning_recipe_v3_review_gpt56sol.md`](reviews/winning_recipe_v3_review_gpt56sol.md)
  — adversarial review of v3; its blocking findings were inputs to v4.

Earlier pre-v4 notes (`architecture_workbench.md`, `policy_engine_mvp_rfc.md`,
`cst_policy_embedding.md`) were removed; their conclusions are folded into
`winning_recipe_design.md` and remain recoverable from git history.

For usage, installation, supported features, and the live update order, use the
repository [`README`](../README.md).
