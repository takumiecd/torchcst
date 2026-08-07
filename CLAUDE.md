# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The repo has a checked-in `.venv` (Python 3.11) and a `uv.lock`.

```bash
.venv/bin/pytest                                   # full suite (~260 tests, ~2s)
.venv/bin/pytest tests/torchcst/test_scored_birth.py
.venv/bin/pytest tests/torchcst/test_scored_birth.py::test_name
.venv/bin/pytest -k "quota and not neuron"
.venv/bin/ruff check .
python -m pip install -e ".[dev]"                  # fresh environment
```

`pyproject.toml` sets `pythonpath = ["src"]` and `testpaths = ["tests"]`, so pytest
works without installing the package.

## Repository scope

`torchcst` is the **framework** repository: reusable implementation plus executable
API contracts only. Preregistrations, training runners, raw results, figures, and
scientific reports belong in the sibling `cst` experiment repository. Do not add a
recipe, a runner, or a result artifact here just to run one experiment — assemble
a policy-tree root directly in `cst` instead.

`docs/` mixes current reference with preserved superseded proposals. Read
[`docs/README.md`](docs/README.md) before treating any design note as current API
documentation; `winning_recipe_design.md` in particular describes modules that are
still future work.

## Architecture

The library represents a linear map as learnable synapse atoms in continuous
coordinate domains composed through kernel matrices, instead of a dense weight
matrix:

```text
W = K_out(mu_out, t) diag(w) K_in(mu_in, s)^T
```

Coordinates `s`/`t`, amplitudes `w`, and kernel bandwidth are ordinary autograd
parameters. A separate clock-driven policy decides when atoms and neurons are
born, retired, merged, or ungated.

### Five responsibilities, five packages

| Package | Owns |
|---|---|
| `compute/` | forward computation and delivery of module-boundary tensors (`CSTLinear`, `CSTConv2d`, `DepthwiseCSTConv2d`, `OffsetCSTConv2d`, `CSTBoundary`, `EntryLinear`/`RankOneLinear` controls, `NeuronGatedLinear`, `BackwardContext`); `compute/backends/` is its pure-function world (`Factored`/`Materialized`/`NativeTruncated`, selected via `backend=`) — tensors in, tensors out, never stores or the engine |
| `instruments/` | backward-derived sufficient statistics and candidate fields |
| `policy/` | the policy tree: roots (`RentEconomy`/`QuotaRegime`), lifecycles, cadences, quotas, distributors, courts, recipes |
| `engine.py` | hooks, clocks, orchestration, validation, atomic apply, audit, optimizer reconciliation |
| `storage/` | entity IDs, physical slots, capacity, prepare/commit, followers |
| `audit/`, `lab/` | one-way event sinks; deterministic experiment/replay helpers |

The boundaries that are easy to violate:

- **Backward produces evidence; it never mutates structure and never selects
  top-K.** A scored birth reads finalized scores at `engine.step()`, after
  microbatch aggregation and after its quota is distributed.
- **Logical quota is not storage allocation.** `StructuralQuota` limits operation
  counts; `BudgetDistributor` divides that count across sites/actions. Physical
  placement stays private to `SynapseStore` and `SlotPool.prepare()`. A future CSR
  or block-sparse layout replaces the *storage allocator*, not the quota API.
- **The tree root, not the engine or a user-supplied planner, assembles the
  plan.** `root.bind(...)` produces the live `RuntimeTree`; per event its
  `EventDraft` stages run in fixed order (absorb → retention → interface
  retire/cascade → birth → response bundles), every stage reading simulated
  post-plan state, and the root then conducts prepare-all → commit-all →
  audit/lineage reconciliation. The engine only drives clocks, capture, and
  the transaction boundary (`engine -> root -> children -> storage`, one
  direction).
- **Ordinary policies are loss-blind.** Only an opt-in profit court
  (`policy/profit.py`, attached as `QuotaRegime(profit=...)`) may evaluate an
  objective, and `engine.step(objective=...)` raises when the tree has no
  profit court.
- **Audit subscribers are one-way sinks** and cannot feed back into policy.

### Update lifecycle (an API contract, not a convention)

```text
engine.begin_update()
  forward / loss.backward()
  engine.observe_microbatch(weight=...)   # once per accumulated microbatch
engine.finalize_backward()                # releases captured tensors & hooks
optimizer.step()
engine.step()                             # structural transaction
```

`finalize_backward()` must release the capture queue before `step()`; the engine
raises if it did not. A failed policy event may consume its clock tick, but
capture state is always closed before the next update.

### Observation timing

Each observation request declares its own stage; different instruments at one site
may differ.

- `backward_inline` → the tensor hook calls `instrument.reduce_backward(module, x,
  g_out)` and retains only the returned statistics (use for large conv
  activations, provided the reduction is chunked).
- `after_backward` → the hook keeps detached `x`/`g_out`, and `finalize_backward()`
  calls `instrument.measure_after_backward(module, x, g_out)`.

Both feed the common `instrument.finalize_update(weighted_measurements, view)`
boundary. That shared boundary is what makes gradient accumulation correct:
**signed** microbatch contributions are weighted and summed before any nonlinear
step such as `abs()`. Never take `abs()` inside a measurement method.

`StructuralEngine(capture_mode=...)` is only a global/per-site experiment override,
not the primary timing declaration.

### Storage: IDs vs slots

Logical entity IDs and physical tensor rows are deliberately different.
`IdAllocator` issues monotonically increasing IDs and never reuses them; `SlotPool`
reuses free holes and rows killed in the same atomic event; capacity grows
geometrically only when the live count no longer fits; store `version` invalidates
stale views and candidate snapshots; `OptimizerStateFollower` expands/resets
optimizer state when tensors grow or rows are replaced. There is no whole-tensor
reallocation per birth/prune event.

Mutation is two-phase and cross-store atomic: every affected store must
`prepare()` completely before any store `commit()`s. `ProposalBundle` is the
joint-operation primitive — e.g. a neuron ungate plus its incident synapse births
must be one bundle so a half-connected neuron cannot commit.

## Extending

The policy tree is the sole authoring path (`docs/policy-tree-phase2.md`; the
older composed-`Policy` and whole-`StructuralPolicy` surfaces described in
[`docs/policy-authoring.md`](docs/policy-authoring.md) are retired):

1. **A named method** — `cSET()`, `cRigL()`, `cRES()`, `RENT()` return a
   `SynapseLifecycle`; place it under a root (`RentEconomy(lam=...)` or
   `QuotaRegime(budget=...)`) with the root-owned `cadence=`, optional
   `quota=`/`distributor=`/`overrides={site-glob: lifecycle}`, and bind via
   `StructuralEngine`.
2. **A hand-composed `SynapseLifecycle`** — `birth_factory` /
   `prune_factory` / `absorb_factory` callables; `NeuronLifecycle` fills the
   interface seat (neuron court and/or the RESPONSE ungate+incident-birth
   capability). `policy/recipes.py` holds the few named, validated whole-tree
   assemblies in active research use (currently `FastConstruction`; the
   historical 5c presets were removed — the test suite keeps a private `LC`
   copy in `tests/torchcst/_recipes.py` as scaffolding).

A new backward statistic is a frozen request with `build(context)` plus an
instrument implementing `prepare`, one timing-specific measurement method, and
`finalize_update`. **No `StructuralEngine` edit and no instrument registration step
is required.** (`InstrumentSpec` remains only for the legacy built-in catalog.)

The migration aliases the composed-`Policy` surface once accepted
(`proposers=`/`schedule=`/`allocator=`/`BudgetAllocator`) were removed with it;
only the current names exist.

### Deliberately unsupported

These raise explicit errors rather than silently degrading: synapse/neuron **kick**
operations, entry-family merge, generic standalone neuron birth, and distributed
structural coordination. Coupled neuron+synapse birth is authored as a
`NeuronLifecycle` RESPONSE capability (`BundleComposer` + `IncidentOutputBirth`),
which emits the ungate and its incident births as one atomic `ProposalBundle` so a
half-connected neuron cannot commit. See the operation matrix in
[`docs/framework-design.md`](docs/framework-design.md).

## Tests

`tests/torchcst/` is the executable contract, one file per mechanism. New
structural behavior should come with mechanism tests for: quota limits, event
timing, immunity, atomic-failure rollback, optimizer-state following, and
deterministic replay. Dual-stage instruments must be tested to produce identical
ranking scores and operations under both observation timings —
`tests/torchcst/test_scored_birth.py` is the compact reference, and the suite runs
`examples/gaussian_gradient_birth.py` itself under both timings.
