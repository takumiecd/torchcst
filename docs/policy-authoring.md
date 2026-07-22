# Authoring structural policies

This guide is for users who want a structural learning rule that is not in the
catalog. The intended workflow is composition first: choose a schedule, a birth
rule, an allocator, and a retention court. Implement a new observation
instrument only when the policy needs a new backward-derived statistic.

## The four questions a policy answers

A policy author normally decides four things:

1. **When is structure observed or changed?** The schedule owns update and
   event timing.
2. **What may be born?** A proposer owns candidate construction.
3. **How are candidates ranked?** A capture instrument owns backward-derived
   sufficient statistics; a selector turns scores into choices.
4. **What is removed?** A retention court owns prune decisions.

The framework owns hook lifetime, microbatch weighting, birth budgets,
transactions, lineage IDs, optimizer-state reconciliation, and replay clocks.
A backward observation never mutates structure directly.

## Use an existing birth component

Policies that do not need backward observations require no instrument:

```python
policy = Policy(
    schedule=PeriodicSchedule(event_interval=200, birth_budget=8),
    proposers=(UniformBirth(initial_weight=0.0),),
    allocator=EvenBudgetAllocator(),
    retention=RentCourt(immunity_events=3, rent_ratio=0.3, strikes=2),
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
    schedule=PeriodicSchedule(
        event_interval=200,
        birth_budget=32,
        observe_window=20,
    ),
    proposers=(birth,),
    allocator=EvenBudgetAllocator(),
    retention=MagnitudeCourt(drop_fraction=0.1),
)
```

The candidate coordinates are instrument state. They are sampled on first use
and sampled again only after the associated `SynapseStore.version` changes.
There is no separate candidate-freeze operation.

The complete executable example is
[`examples/gaussian_gradient_birth.py`](../examples/gaussian_gradient_birth.py).
The test suite runs that exact example with both capture modes.

## Inline versus deferred measurement

The instrument implements one `measure(module, x, g_out)` method. The engine
decides when to call it:

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

- `inline_reduced` calls `measure` from the output tensor hook and retains only
  the returned sufficient statistics.
- `deferred` retains detached `x` and `g_out`, then calls the same `measure`
  method in `finalize_backward()`.

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

    def build(self, context: InstrumentBuildContext):
        return MyScoreInstrument(
            store=context.store,
            module=context.module,
            scale=self.scale,
        )
```

The instrument implements three capture methods:

```python
class MyScoreInstrument:
    name = "my_scores"

    def prepare(self, view, module):
        # Reconcile state before the observed forward. Do nothing if the
        # relevant store version has not changed.
        ...

    def measure(self, module, x, g_out):
        # Return signed tensors. Do not take abs here.
        return {"gradient": signed_candidate_gradient}

    def finalize_update(self, measurements, view):
        total = weighted_sum(measurements, "gradient")
        self.scores = total.abs()
```

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
- `measure` returns detached, signed sufficient statistics and never mutates
  structure;
- microbatch accumulation applies nonlinear aggregation after signed summation;
- deferred and inline-reduced modes produce equal ranking scores and operations;
- selector output never exceeds the schedule-issued budget;
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
