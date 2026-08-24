# CSTOptimizer: one owner per parameter

`PullbackAdam` is the single-site primitive. It owns one site's `s`/`t` — and,
under `rent=`/`decay=`, that site's `w` — and deliberately nothing else. A model
with many CST sites (six transformer layers × FFN up/down = twelve sites) then
left the *caller* to do two things by hand at every call site:

```python
# the old shape, per site, every time
coordinate_ids = {id(layer.synapses.s), id(layer.synapses.t)}
ordinary = torch.optim.AdamW(
    [p for p in model.parameters() if id(p) not in coordinate_ids], lr=1e-3
)
coordinates = [PullbackAdam(site, ...) for site in sites]
...
ordinary.step()
for optimizer in coordinates:
    optimizer.step()
```

Every parameter that fell through that seam was silently stepped twice or not at
all. `CSTOptimizer` moves the partition into one place and makes both failures
loud.

## The two objects

`PullbackConfig` is the immutable coordinate **policy**: exactly
`PullbackAdam`'s keywords minus the module it binds to. One record, reused
across layers, varied per site with `dataclasses.replace`. Validation stays
`PullbackAdam`'s and happens at `build()` time — a config is a record of intent,
not a second copy of the rules.

`CSTOptimizer` is the model-level **coordinator**. It discovers the continuous
sites through `model.named_modules()`, builds one `PullbackAdam` per site,
decides who owns what, hands the remainder to the base optimizer, and steps
everything once.

## A multi-layer model

```python
from dataclasses import replace

import torch
from torchcst import CSTOptimizer, PullbackConfig
from torchcst.optim import ChartPullbackAdam, continuous_sites

coordinates = PullbackConfig(
    moment_space="tangent",
    metric="diag",
    cap_sigma=0.1,
    target_step=0.01,
)

optimizer = CSTOptimizer(
    model,
    coordinates=coordinates,
    overrides={
        # the first matching glob wins, in insertion order
        "layers.*.ffn_down": replace(coordinates, target_step=0.005),
        "embed": None,            # leave this site's coordinates to the base
    },
    charts=[ChartPullbackAdam(model.residual, [s for _, s in continuous_sites(model)])],
    base=lambda params: torch.optim.AdamW(params, lr=1e-3),
)

print(optimizer.summary())
```

The update order is unchanged; the coordinator *is* the `optimizer` in it:

```python
optimizer.zero_grad(set_to_none=True)
engine.begin_update()
loss(model(x)).backward()
engine.observe_microbatch(weight=1.0)
engine.finalize_backward()
optimizer.step()      # one call: base, then every site, then every chart
engine.step()         # the structural transaction stays last
```

A pattern in `overrides` that matches no discovered site raises, so a renamed
layer surfaces as an error instead of a silently unapplied override.

## The partition

| Parameter | Owner |
|---|---|
| a site's `s`, `t` | that site's `PullbackAdam` |
| a site's `w` | the same `PullbackAdam` **only** when its config sets `rent=`/`decay=` (with `lr_w=`); otherwise the base optimizer |
| a learnable chart's `mu` | the `ChartPullbackAdam` passed in `charts=`, else the base optimizer |
| dense weights, norms, kernel bandwidths, per-atom family columns | the base optimizer |
| `requires_grad=False` | nobody, reported as `frozen` |

Amplitude ownership is read off the built optimizer's `owns_amplitudes`, so the
rule has exactly one definition. A parameter claimed twice raises at
construction; a trainable parameter claimed by nobody raises unless
`allow_unowned=True`. Every claim must also be a parameter of the model handed
in: a site whose store is not registered under it, or a `ChartPullbackAdam`
over a chart that lives outside it, is rejected by name rather than silently
stepped.

`base=` accepts either a factory called with the parameters the coordinator
determined it owns, or an already-constructed `torch.optim.Optimizer` whose
parameter list is then *validated* against the same partition — that is the
path for a caller who wants their own param groups and still wants the
double-step check. With a factory and nothing left to own, `optimizer.base`
stays `None`.

Diagnostics are named, not positional:

```python
optimizer.site_names                    # ('layers.0.ffn_up', 'layers.0.ffn_down', ...)
optimizer.ownership()                   # {'...synapses.s': 'site:layers.0.ffn_up', ...}
optimizer.expected_base_parameters()    # what the partition says base should hold
optimizer.summary()
```

The two are deliberately different questions.
`expected_base_parameters()` is the *expectation* computed from the partition —
what to hand a factory, or to build an explicit optimizer from. `ownership()`
is the *report*: it labels a parameter `base` only when the base optimizer
really holds it, so an explicit optimizer accepted under `allow_unowned=True`
that omits some of them shows those as `unowned` (and `summary()` counts them).
An `unowned` parameter is neither stepped nor cleared by `zero_grad()` — the
coordinator touches only what it owns.

## Shared charts

A chart is shared by every incident site-side, so its `mu` has one gradient and
must have one owner. Pass the chart optimizers explicitly — the coordinator
does not invent them, because the incident-site list is a modelling decision —
and it rejects two chart optimizers over the same `NeuronStore`, two charts
sharing a name (chart names key the state dict), and a chart whose `mu` is not
a parameter of the model. A learnable chart with no chart optimizer is an
ordinary base-optimizer parameter, and `ownership()` says so.

## State

`state_dict()` carries the site optimizers (keyed by module name), the chart
optimizers (keyed by chart name) and the base optimizer under one schema;
`load_state_dict()` rejects a state whose site or chart set differs from the
coordinator's, and one that disagrees about whether a base optimizer exists.

The site optimizers clone their tensors; the base optimizer keeps
`torch.optim`'s convention of handing out live references. Loading into a
second coordinator *in the same process* therefore needs a `copy.deepcopy`
first (a `torch.save`/`torch.load` round trip already does this) — otherwise
the two base optimizers share one set of moments.

## Why it is not a `torch.optim.Optimizer`

It has no `param_groups`, on purpose. A coordinate optimizer has no `lr` to
schedule: its step size is the `target_step`-calibrated `eta` and the
`cap_sigma` trust region. A `torch.optim.lr_scheduler` writing `lr` into group
dicts would therefore move only the dense half — silently. Attach schedulers to
`optimizer.base` and pass the same multiplicative factor to `step()`:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base, T)
...
optimizer.step(lr_scale=scheduler.get_last_lr()[0] / initial_lr)
scheduler.step()
```

## Step order is a contract, not a detail

`step()` runs the base optimizer, then the sites in discovery order, then the
charts in the order given.

Disjoint ownership does **not** make that order numerically neutral. A pullback
metric is a function of the amplitudes `w`, the chart points `mu`, the incident
atom coordinates and the kernel bandwidth `sigma` — and the base optimizer
steps `w`, `sigma` and any chart it owns *before* the site optimizers whiten
with `G`. Each site's step likewise moves coordinates a later chart optimizer's
metric sums over. Running the same three groups in another order gives
different numbers.

The order is fixed rather than configurable because it reproduces what the
hand-wired runners already do — `ordinary.step()` then `coordinates.step()`,
as in [`pullback-adam.md`](pullback-adam.md) — so moving a model onto
`CSTOptimizer` is behaviour-preserving. Treat a change to it as a change to
results.

## Clocks and geometry stay independent

The config forwards, and does not entangle, the four knobs that are easy to
confuse:

- `betas` is the **step clock** (ordinary Adam EMA over steps);
- `moment_distance=(tau_m, tau_v)` is a separate **per-row geometric clock** in
  sigma units, applied on top. `None` is the framework default and preserves
  ordinary step-clock Adam exactly; the framework ships no distance-clock
  default because there is no framework-level evidence for one;
- `metric` selects the geometry `G` is measured in (`diag`/`block`/`full`);
- `moment_space` selects where the moments live (`parameter`/`tangent`).

Likewise `repulsion=` enters the coordinate moments and `decoupled_repulsion=`
is applied after the adaptive step and never enters them — the config keeps
both fields separate and mutually exclusive, exactly as `PullbackAdam` does.

## Scope

`CSTOptimizer` is orchestration only. It never computes an update, never reads
a loss, and never touches structure — births, deaths and remaps still reach the
site optimizers through the store's follower hub, which
`PullbackConfig.build(..., subscribe=True)` wires by default. `PullbackAdam`
remains the low-level primitive and the compatibility API; nothing here changes
its behaviour or its `state_dict` schema.

`ChartPullbackAdam` and `PullbackAdam` still duplicate their whitening and
calibration internals. Unifying them is a separate, behaviour-preserving phase;
the coordinator was built on top of both as they are.
