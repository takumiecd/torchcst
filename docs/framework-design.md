# Framework design and extension map

This document is the current implementation map for `torchcst`. It records the
boundaries that are easy to forget while writing an experiment: who owns time,
scores, decisions, storage, and mutation; which extension path to use; and
which structural operations are actually routed by the standard engine today.

`torchcst` is the framework repository. Preregistrations, training runners,
raw results, figures, and scientific reports belong in the sibling `cst`
repository.

## The central separation

One optimizer update and one structural event are different transactions.
Autograd produces evidence; policy code turns finalized evidence into proposed
operations; the engine validates and applies those operations at a later
structural boundary.

```text
PyTorch update
  compute module -> BackwardContext -> observation instrument
                                      |
                                      v
                              finalized evidence
                                      |
Structural event                     v
  cadence -> quota -> action rules -> distributor -> StructuralPlan
                                                    |
                                                    v
                                  Store.prepare -> Store.commit
```

Hooks never choose top-K and never mutate a store. A scored birth action reads
the score field at `engine.step()`, after microbatch aggregation and after its
logical quota is known.

## Ownership by component

| Component | Owns | Does not own |
|---|---|---|
| Compute module | forward computation and delivery of module-boundary tensors | structural timing or mutation |
| Observation request | immutable configuration and per-site instrument construction | hooks or structural quota |
| Observation instrument | backward-derived sufficient statistics and update aggregation | top-K budget or store writes |
| Cadence | event, observation-window, and phase timing | operation counts |
| Quota policy | logical limits for birth, prune, and merge families | site selection or physical slots |
| Action rule | prune decisions, candidates, scores-to-proposals, or merge proposals | transaction commit |
| Budget distributor | division of a logical quota across sites/actions | tensor memory allocation |
| Whole `StructuralPolicy` | an indivisible algorithm's complete plan | storage internals or commit protocol |
| Engine | hooks, clocks, orchestration, validation, atomic apply, audit, optimizer reconciliation | research metrics and experiment protocol |
| Store / `SlotPool` | IDs, physical slots, capacity, prepare/commit, followers | policy budgets |

The old `BudgetAllocator` name meant logical budget distribution. It is now a
compatibility alias for `BudgetDistributor`; it has never allocated storage.

## The composed policy path

Use `Policy` when the algorithm separates naturally into reusable parts:

```python
policy = Policy(
    cadence=PeriodicCadence(event_interval=200, observe_window=20),
    quota=CallableQuota(annealed_quota),
    observations=(score_request,),
    actions=(
        ActionSpec.synapse_prune(MagnitudeCourt(drop_fraction=0.1)),
        ActionSpec.synapse_birth(
            ScoredBirth(score_request, selector=TopKSelector())
        ),
    ),
    distributor=EvenBudgetDistributor(),
)
```

The engine, rather than a user-visible planner, performs the standard ordering:

1. ask the cadence whether an event fires;
2. obtain the event's `StructuralQuota`;
3. run prune courts and enforce explicit prune caps;
4. form site/action budget requests and distribute birth/merge quota;
5. invoke proposal rules and assemble a `StructuralPlan`;
6. prepare every affected store, then commit the atomic unit;
7. reconcile optimizer state, lineage/age state, and audit records.

The public `actions=` surface replaces the old `proposers=` and `retention=`
fields. `schedule=` and `allocator=` are likewise migration fields; new code
should use `cadence=`, `quota=`, and `distributor=`.

### Annealing structural change

Sigma annealing and structural annealing are independent. Keep the
kernel's sigma fixed on the compute module and return a time-varying
`StructuralQuota`:

```python
def annealed_quota(clock, phase):
    if phase is Phase.FROZEN:
        return StructuralQuota.zero()
    fraction = max(0.0, 1.0 - clock.event_index / 20)
    return StructuralQuota(
        synapse_birth=round(32 * fraction),
        synapse_prune=round(32 * fraction),
    )

quota = CallableQuota(annealed_quota)
```

`WindowedQuota` is preferable when the exact event-by-event supply must be
frozen in a preregistration. `CallableQuota` is preferable for cosine,
resource-aware, or distributed supply functions.

## The whole-policy path

Use a first-class `StructuralPolicy` when decomposition would hide coupling or
force one decision across artificial component boundaries. A whole policy has
only the lifecycle capabilities it needs:

```python
class MyPolicy:
    requires = ()

    def capture(self, clock):
        return False

    def plan(self, context):
        if context.clock.update_step % 200:
            return None
        return StructuralPlan((...operations_or_bundles...,))

    def on_applied(self, context, applied_operations):
        ...  # optional state update from what actually committed
```

`PolicyContext` provides immutable synapse/neuron views, aligned ages,
declared instruments, retired-lineage state, the candidate event clock, and
the policy RNG. The engine still owns validation and two-phase mutation.

`ProposalBundle` is the joint-operation primitive. For example, neuron birth
or ungating plus its incident synapse births should be one atomic bundle so a
half-connected neuron cannot commit. This is also the current public route for
custom coupled neuron+synapse birth algorithms.

## Observation and backward lifecycle

An observation request chooses one of two execution stages:

- `backward_inline`: the output tensor hook calls
  `instrument.reduce_backward(module, x, g_out)` and retains only the returned
  sufficient statistics;
- `after_backward`: the hook retains detached `x` and `g_out`, and
  `finalize_backward()` calls
  `instrument.measure_after_backward(module, x, g_out)`.

Both stages feed `instrument.finalize_update(weighted_measurements, view)`.
This common boundary is what makes gradient accumulation correct: signed
microbatch contributions can be weighted and summed before a nonlinear score
such as absolute value is committed.

```text
engine.begin_update()
  forward
  loss.backward()
  engine.observe_microbatch(weight=...)
  [repeat forward/backward/observe for gradient accumulation]
engine.finalize_backward()
optimizer.step()
engine.step()
```

Inline capture normally saves memory for large convolutional activations,
provided the reduction itself is chunked. Deferred capture preserves the full
boundary tensors until finalization and is useful when the statistic cannot be
expressed as a compact inline reduction. Timing is declared per request; the
engine-level `capture_mode` exists only as an experiment/debug override.

## Candidate fields and birth

The continuous candidate field used by `ContinuousGradientRequest` is a pool
of possible continuous `(s, t)` coordinates plus their accumulated scores. It
is not `EntryLinear` and is not a separate structural entity. The pool is
sampled on first use and refreshed when the associated store version changes.

Current standard birth choices include:

- `UniformBirth`: sample continuous coordinates uniformly and emit births;
- `ScoredBirth(ContinuousGradientRequest(...))`: score continuous
  coordinates from backward evidence and take top-K at the structural event;
- discrete-entry and rank-one controls, which are comparison families rather
  than the center of CST research.

Bandwidth belongs to the kernel object. Birth policy selects coordinates,
lineage, and initial amplitude; it does not silently anneal sigma.

### Continuous kernel families

`RepresentationSpec.continuous(..., kernel=...)` selects the profile, and the
compute module refuses a kernel whose `family` disagrees with the store's spec
because mass and rent constants are family-specific.

- `GaussianKernel` (`"gaussian"`): support is the whole domain at every
  positive sigma, so a position gradient reaches every neuron and the
  represented matrix is never structurally sparse.
- `TriangularKernel` (`"triangular"`): `relu(1 - r/sigma)`, exactly zero
  outside the radius-sigma ball. The represented matrix is structurally
  sparse, and shrinking sigma below the neuron spacing reproduces the entry
  family's delta behaviour exactly -- it is the only continuous family that
  reaches the entry family continuously. The cost is that atoms outside every
  neuron's support receive no position gradient, so a compact kernel relies on
  structural birth/death for transport where the Gaussian relies on its tail.

## Storage behavior

Logical entity IDs and physical tensor rows are intentionally different:

- `IdAllocator` issues monotonically increasing IDs and never reuses them;
- `SlotPool` reuses existing free holes and rows killed in the same atomic
  event;
- capacity grows geometrically only when the required live count no longer
  fits;
- store versions invalidate stale views and candidate snapshots;
- `OptimizerStateFollower` expands/resets optimizer state when store tensors
  grow or rows are replaced.

There is therefore no whole-tensor reallocation on every birth/prune event.
Alternative layouts such as CSR, paged, or block-sparse storage should replace
the store's physical allocator, not add fields to `StructuralQuota` or
`BudgetDistributor`.

## Implemented operation matrix

This table describes the current standard composed-policy route, not merely
the existence of operation dataclasses.

| Operation family | Composed `Policy` | Whole `StructuralPolicy` | Notes |
|---|---:|---:|---|
| Synapse birth | yes | yes | uniform and scored continuous birth supported |
| Synapse prune/death | yes | yes | court decision plus optional quota cap |
| Synapse merge | yes | yes | representation-dependent; not defined for every family |
| Neuron prune/retire | yes | yes | incident synapse deaths are expanded atomically |
| Neuron ungate + incident births | catalog compatibility path | yes | use an atomic bundle for new coupled algorithms |
| Generic neuron birth action | not yet routed | yes | `ActionSpec.neuron_birth` is reserved, but standard distribution/execution is incomplete |
| Kick operations | no | operation type only | deliberately rejected by current apply path |
| Distributed coordination | no | no | quota/distributor interfaces are local today |

The `ActionSpec.neuron_birth(...)` constructor alone must not be interpreted as
finished engine support. Until the composed route is implemented and covered
by mechanism tests, joint neuron+synapse experiments should use a whole policy
that emits `ProposalBundle` objects.

## Checklist for adding a policy

1. State the decision in terms of timing, evidence, quota, and operations.
2. Reuse a cadence/quota/action when its semantics match; do not create a
   catalog name merely to run one experiment.
3. Add a new observation request/instrument only when existing finalized state
   is insufficient.
4. Choose inline or deferred measurement in the request and implement only the
   capabilities that are valid for that statistic.
5. Use a whole policy and atomic bundles for genuinely coupled decisions.
6. Add mechanism tests for quota limits, timing, immunity, atomic failure,
   optimizer-state following, and deterministic replay as applicable.
7. Put only the reusable framework implementation and executable API example
   in `torchcst`; put the scientific protocol, runner, result, figure, and
   report in `cst`.

For code-level authoring details, continue with
[`policy-authoring.md`](policy-authoring.md). For exact hook and tensor lifetime
details, see [`capture-lifecycle.md`](capture-lifecycle.md).
