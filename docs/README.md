# Design document status

The executable contract lives in `src/torchcst/` and `tests/torchcst/`. Design
notes explain why that contract exists, but several documents intentionally
preserve superseded proposals. Use this index before reading pseudocode as
current API documentation.

## Current design reference

- [`winning_recipe_design.md`](winning_recipe_design.md) — v4 architecture and
  accepted invariants behind the current implementation. Some planned modules
  and distributed paths described there are still future work.

## Historical decision records

- [`reviews/winning_recipe_v3_review_gpt56sol.md`](reviews/winning_recipe_v3_review_gpt56sol.md)
  — adversarial review of v3; its blocking findings were inputs to v4.

Earlier pre-v4 notes (`architecture_workbench.md`, `policy_engine_mvp_rfc.md`,
`cst_policy_embedding.md`) were removed; their conclusions are folded into
`winning_recipe_design.md` and remain recoverable from git history.

For usage, installation, supported features, and the live update order, use the
repository [`README`](../README.md).
