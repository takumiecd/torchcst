# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The repo has a checked-in `.venv` (Python 3.11) and a `uv.lock`.

```bash
.venv/bin/pytest                                   # full suite (~165 tests, ~2s)
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
catalog entry, a runner, or a result artifact here just to run one experiment —
compose `Policy` directly in `cst` instead.

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
| `compute/` | forward computation and delivery of module-boundary tensors (`CSTLinear`, `CSTConv2d`, `EntryLinear`/`RankOneLinear` controls, `NeuronGatedLinear`, `BackwardContext`) |
| `instruments/` | backward-derived sufficient statistics and candidate fields |
| `policy/` | cadence, quota, action rules, distributors, courts, catalog |
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
- **The engine, not a user-supplied planner, assembles the plan** for composed
  policies (order: cadence → quota → prune courts + prune caps → budget requests
  → distribution → proposal rules → `StructuralPlan` → prepare-all → commit-all →
  optimizer/lineage/audit reconciliation).
- **Ordinary policies are loss-blind.** Only an opt-in profit trial
  (`policy/profit.py`) may evaluate an objective, and `engine.step(objective=...)`
  raises unless the composed policy declares one.
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

Two equally public authoring paths — see [`docs/policy-authoring.md`](docs/policy-authoring.md):

1. **Composed `Policy`** — `cadence=`, `quota=`, `observations=`, `actions=`
   (`ActionSpec.synapse_birth(...)` etc.), `distributor=`. Use when those
   boundaries fit the algorithm.
2. **Whole `StructuralPolicy`** — one object implementing `capture(clock)`,
   `plan(context) -> StructuralPlan | None`, optional `on_applied(...)` and
   `bind_instruments(...)`. Use when decomposition would hide coupling. It needs no
   cadence, distributor, or court.

A new backward statistic is a frozen request with `build(context)` plus an
instrument implementing `prepare`, one timing-specific measurement method, and
`finalize_update`. **No `StructuralEngine` edit and no instrument registration step
is required.** (`InstrumentSpec` remains only for the legacy built-in catalog.)

### Migration aliases still accepted

`proposers=` / `retention=` → `actions=`; `schedule=` → `cadence=`; `allocator=` →
`distributor=`; `BudgetAllocator` → `BudgetDistributor`. New code must use the
current names.

### Deliberately unsupported

These raise explicit errors rather than silently degrading: synapse/neuron **kick**
operations, entry-family merge, **generic composed-policy neuron birth**, and
distributed structural coordination. Note that `ActionSpec.neuron_birth(...)`
exists but its composed distribution/execution route is *not* implemented — coupled
neuron+synapse birth must be authored today as a whole `StructuralPolicy` emitting
an atomic `ProposalBundle`. See the operation matrix in
[`docs/framework-design.md`](docs/framework-design.md).

## Tests

`tests/torchcst/` is the executable contract, one file per mechanism. New
structural behavior should come with mechanism tests for: quota limits, event
timing, immunity, atomic-failure rollback, optimizer-state following, and
deterministic replay. Dual-stage instruments must be tested to produce identical
ranking scores and operations under both observation timings —
`tests/torchcst/test_scored_birth.py` is the compact reference, and the suite runs
`examples/gaussian_gradient_birth.py` itself under both timings.
