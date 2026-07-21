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

- [`architecture_workbench.md`](architecture_workbench.md) — early lifecycle
  alternatives and responsibility-boundary exploration.
- [`policy_engine_mvp_rfc.md`](policy_engine_mvp_rfc.md) — pre-v4 Global Policy
  RFC. Names such as `UpdateRequest`, `MutationPlan`, and older `CSTEngine`
  sketches are not current APIs.
- [`cst_policy_embedding.md`](cst_policy_embedding.md) — feasibility exercise
  written against that pre-v4 RFC.
- [`reviews/winning_recipe_v3_review_gpt56sol.md`](reviews/winning_recipe_v3_review_gpt56sol.md)
  — adversarial review of v3; its blocking findings were inputs to v4.

For usage, installation, supported features, and the live update order, use the
repository [`README`](../README.md).
