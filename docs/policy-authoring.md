# Authoring structural policies

This guide is for users who want a structural learning rule that is not in the
catalog. There are two equally public authoring paths:

- `Policy` composes reusable cadence, quota, action, distributor, and retention
  components. Use it when those boundaries fit the algorithm.
- `StructuralPolicy` implements the learning rule as one object. Use it when
  decomposition would distort the algorithm or merely add boilerplate.

A distributor and cadence are therefore conveniences, not framework
requirements. Implement a new observation instrument only when the policy
needs a new backward-derived statistic.

## Implement a complete policy as one object

A whole policy answers three engine lifecycle questions:

```python
class MyPolicy:
    requires = ()

    def capture(self, clock):
        # Should this update collect the declared backward observations?
        return False

    def plan(self, context):
        # None means no event. StructuralPlan() is an event with no operations.
        if context.clock.update_step % 100:
            return None

        view = context.synapses["conv1"]
        operations = decide_birth_prune_merge(view, context.ages["conv1"])
        return StructuralPlan(
            proposals=tuple(operations),
            synapse_immunity_events=3,
        )
```

Pass the object directly to `StructuralEngine`; it needs no `Cadence`,
`BudgetDistributor`, `OpProposer`, or `RetentionCourt`. `PolicyContext` exposes
read-only synapse and neuron views, aligned ages, declared instruments, the
retired-lineage registry, the policy RNG, and the candidate event clock.

Each plan item is either one operation or an atomic `ProposalBundle`. The
framework still owns operation validation, two-phase cross-store apply,
lineage retirement, optimizer-state reconciliation, event logging, and hook
lifetime. A policy can optionally implement
`on_applied(context, applied_operations)` to update state from the operations
that actually committed.

Backward-based whole policies declare and bind observations in the same way as
components:

```python
class MyPolicy:
    request = MyScoreRequest()
    requires = (request,)

    def bind_instruments(self, site, instruments):
        self.scores_by_site[site] = instruments[self.request.name]

    def capture(self, clock):
        return clock.update_step % 100 >= 90

    def plan(self, context):
        scores = context.instrument("conv1", self.request.name)
        ...
```

This is intentionally a lifecycle contract, not a hidden controller base
class. A policy may internally reuse `PeriodicCadence`, `ScoredBirth`, or a
selector, but it is not required to expose those choices to the engine.

## The five questions a composed policy answers

A composed policy normally decides five things:

1. **When is structure observed or changed?** The cadence owns update and
   event timing.
2. **How much structure may change?** The quota owns logical operation limits.
3. **What may be born?** A birth action owns candidate construction.
4. **How are candidates ranked?** A capture instrument owns backward-derived
   sufficient statistics; a selector turns scores into choices.
5. **What is removed?** A prune action owns retention decisions.

The framework owns hook lifetime, microbatch weighting, quota enforcement,
transactions, lineage IDs, optimizer-state reconciliation, and replay clocks.
A backward observation never mutates structure directly.

`ActionSpec` labels an independently authored rule; it does not become a large
planner object:

```python
actions = (
    ActionSpec.synapse_prune(RentCourt(...)),
    ActionSpec.synapse_birth(UniformBirth(...)),
    ActionSpec.synapse_merge(MergeProposer(...)),
)
```

The engine evaluates prune actions, distributes the matching quota, invokes
birth/merge actions, and constructs the `StructuralPlan`. A whole policy places
an all-or-nothing cross-store change in a `ProposalBundle`; no additional
planner object is required. The legacy `proposers=`, `retention=`, and
`allocator=` fields remain accepted only as a migration surface.

## Logical quotas are not storage allocation

`StructuralQuota` limits logical operations. `BudgetDistributor` divides that
quota over sites and actions; despite the old `BudgetAllocator` name, it never
selects tensor slots. Physical placement remains private to `SynapseStore` and
`SlotPool.prepare()`:

```text
Cadence -> StructuralQuota -> BudgetDistributor -> operations
                                                   |
                                                   v
                         Store.prepare -> free/dead slot reuse or capacity grow
```

The store reuses free holes and same-event death slots before growing. Capacity
grows geometrically only when the requested live count no longer fits. A
future CSR, paged, or block-sparse layout must replace the storage allocator,
not the policy budget API.

## Use an existing birth component

Policies that do not need backward observations require no instrument:

```python
policy = Policy(
    cadence=PeriodicCadence(event_interval=200),
    quota=ConstantQuota(StructuralQuota(synapse_birth=8)),
    actions=(
        ActionSpec.synapse_prune(
            RentCourt(immunity_events=3, rent_ratio=0.3, strikes=2),
        ),
        ActionSpec.synapse_birth(UniformBirth(initial_weight=0.0)),
    ),
    distributor=EvenBudgetDistributor(),
)
```

Catalog entries such as `LC` are convenience constructors for compositions of
this kind. Research code may compose `Policy` directly without adding a new
catalog entry.

## Compose a gradient-scored Gaussian birth

`ScoredBirth` connects any candidate-score instrument to a selector and emits
budgeted `SynapseBirth` operations. For continuous Gaussian coordinates:

```python
scores = ContinuousGradientRequest(
    pool_size=4096,
    decay=0.9,
    chunk_size=128,
)

birth = ScoredBirth(
    request=scores,
    selector=TopKSelector(),
    initial_weight=0.0,
)

policy = Policy(
    cadence=PeriodicCadence(
        event_interval=200,
        observe_window=20,
    ),
    quota=ConstantQuota(StructuralQuota(synapse_birth=32)),
    actions=(
        ActionSpec.synapse_prune(MagnitudeCourt(drop_fraction=0.1)),
        ActionSpec.synapse_birth(birth),
    ),
    distributor=EvenBudgetDistributor(),
)
```

The candidate coordinates are instrument state. They are sampled on first use
and sampled again only after the associated `SynapseStore.version` changes.
There is no separate candidate-freeze operation.

The complete executable example is
[`examples/gaussian_gradient_birth.py`](../examples/gaussian_gradient_birth.py).
The test suite runs that exact example with both observation timings.

## Inline versus deferred measurement

The observation request decides when its instrument computes. Different
requests at the same site may choose different stages:

```python
inline_scores = MyScoreRequest(timing="backward_inline")
deferred_residual = ResidualRequest(timing="after_backward")
```

An engine-level setting remains available as an experiment override:

```python
engine = StructuralEngine(
    stores,
    policy,
    modules=modules,
    capture_mode={
        "large_conv": "inline_reduced",
        "small_layer": "deferred",
    },
)
```

- `backward_inline` calls `reduce_backward` from the output tensor hook and
  retains only the returned sufficient statistics.
- `after_backward` retains detached `x` and `g_out`, then calls
  `measure_after_backward` in `finalize_backward()`.

In both modes, `finalize_update` receives weighted microbatch measurements at
the update boundary. Nonlinear aggregation such as absolute value therefore
occurs after signed microbatch summation.

## Add a new backward statistic

A new statistic consists of an observation request and an instrument. It does
not require an edit to `StructuralEngine`.

```python
from torchcst.instruments import (
    CandidateSnapshot,
    InstrumentBuildContext,
    weighted_sum,
)
```

The request is immutable configuration and builds one instrument per site:

```python
@dataclass(frozen=True)
class MyScoreRequest:
    scale: float = 1.0
    name: str = "my_scores"
    timing: str = "after_backward"

    def build(self, context: InstrumentBuildContext):
        return MyScoreInstrument(
            store=context.store,
            module=context.module,
            scale=self.scale,
        )
```

The instrument implements a timing-neutral lifecycle plus exactly the
capability requested. A deferred-only example is:

```python
class MyScoreInstrument:
    name = "my_scores"

    def prepare(self, view, module):
        # Reconcile state before the observed forward. Do nothing if the
        # relevant store version has not changed.
        ...

    def measure_after_backward(self, module, x, g_out):
        # Return signed tensors. Do not take abs here.
        return {"gradient": signed_candidate_gradient}

    def finalize_update(self, measurements, view):
        total = weighted_sum(measurements, "gradient")
        self.scores = total.abs()
```

An inline-only instrument implements `reduce_backward` instead. An instrument
that intentionally supports both stages may implement both methods and share a
private numerical kernel; the public methods remain separate so hook-time
cost and memory behavior are explicit.

For use with `ScoredBirth`, it additionally exposes:

```python
def candidate_snapshot(self):
    return CandidateSnapshot(source, target, scores, lineages=None)
```

Continuous candidates normally return `lineages=None`; `ScoredBirth` allocates
fresh persistent lineages only for candidates that are actually born.

`InstrumentSpec` remains supported for the built-in legacy catalog
instruments. New third-party extensions should use a request with
`build(context)` so they do not depend on the engine's compatibility registry.

An instrument may return multiple sufficient statistics:

```python
return {
    "gradient": g,
    "curvature": h,
}
```

This supports residual- or curvature-normalized ranking without changing the
capture engine. The instrument combines these components in
`finalize_update`, and `candidate_snapshot` exposes the resulting ranking
score.

## Contract checklist

Third-party policy tests should establish:

- request configuration is deterministic and validates invalid values;
- `prepare` changes candidate state only when its documented structural state
  changes;
- `reduce_backward` and/or `measure_after_backward` return detached, signed
  sufficient statistics and never mutate structure;
- microbatch accumulation applies nonlinear aggregation after signed summation;
- dual-stage instruments produce equal ranking scores and operations in both
  stages;
- selector output never exceeds its distributed structural quota;
- replay with the same RNG state produces identical candidates and operations;
- state serialization restores every policy-owned running statistic.

The framework tests in
[`tests/torchcst/test_scored_birth.py`](../tests/torchcst/test_scored_birth.py)
are a compact reference.

## When to add a catalog entry

Add a catalog constructor only after a complete policy composition has stable,
documented defaults. During research, construct `Policy` directly in the
experiment repository. This keeps the framework catalog small while leaving
all extension points public.
