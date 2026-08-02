# torchcst — Continuous Sparse Training in PyTorch

> [!WARNING]
> **Research Preview.** `torchcst` is an experimental research codebase, not a
> production-ready training library. APIs, numerical behavior, policy
> semantics, and checkpoint compatibility may change without notice. The
> current release is intended for inspection, method development, and early
> experiments; it should not yet be treated as a stable dependency or as
> evidence of an established scientific result.

`torchcst` explores continuous sparse training: a linear map is represented by
a finite collection of learnable synapse atoms in continuous coordinate
domains, rather than by one independently stored parameter for every entry of
a dense weight matrix. Its primary compute abstraction is `CSTLinear`.

## Mathematical model

Let the input and output neuron charts contain coordinates
$\mu_i^{\mathrm{in}}\in\mathbb{R}^{d_{\mathrm{in}}}$ and
$\mu_j^{\mathrm{out}}\in\mathbb{R}^{d_{\mathrm{out}}}$. A live synapse atom

```math
\theta_a = (s_a, t_a, w_a)
```

has a source coordinate $s_a$, a target coordinate $t_a$, and a scalar
amplitude $w_a$. Equivalently, $M$ live atoms define the signed atomic
measure

```math
\nu = \sum_{a=1}^{M} w_a\,\delta_{(s_a,t_a)}.
```

The continuous kernel operator induced by this measure is

```math
\begin{aligned}
W_{\mathrm{core}}(\mu_j^{\mathrm{out}},\mu_i^{\mathrm{in}})
&=
\int
\kappa_{\mathrm{out}}(\mu_j^{\mathrm{out}},t)\,
\kappa_{\mathrm{in}}(\mu_i^{\mathrm{in}},s)\,
\mathrm{d}\nu(s,t) \\
&=
\sum_{a=1}^{M}
w_a\,
\kappa_{\mathrm{out}}(\mu_j^{\mathrm{out}},t_a)\,
\kappa_{\mathrm{in}}(\mu_i^{\mathrm{in}},s_a).
\end{aligned}
```

The functions $\kappa_{\mathrm{in}}$ and $\kappa_{\mathrm{out}}$ are the
kernels: they specify how strongly an atom at one continuous coordinate
couples to a neuron at another coordinate. Evaluating them over every neuron
and live atom produces the rectangular kernel feature matrices

```math
(\Phi_{\mathrm{in}})_{ia}
= \kappa_{\mathrm{in}}(\mu_i^{\mathrm{in}},s_a),
\qquad
(\Phi_{\mathrm{out}})_{ja}
= \kappa_{\mathrm{out}}(\mu_j^{\mathrm{out}},t_a).
```

Then the conceptual dense weight is

```math
W_{\mathrm{core}}
=
\Phi_{\mathrm{out}}\,
\mathrm{diag}(w)\,
\Phi_{\mathrm{in}}^{\mathsf T}
\in \mathbb{R}^{n_{\mathrm{out}}\times n_{\mathrm{in}}}.
```

With input and output neuron gates $g_{\mathrm{in}}$ and
$g_{\mathrm{out}}$, the map actually represented by `CSTLinear` is

```math
W
=
\mathrm{diag}(g_{\mathrm{out}})\,
W_{\mathrm{core}}\,
\mathrm{diag}(g_{\mathrm{in}}).
```

For a row-major input batch $X$, the implementation computes the equivalent
factorized expression

```math
Y
=
\left(
\left(
(X\odot g_{\mathrm{in}})\,\Phi_{\mathrm{in}}
\right)
\odot w
\right)
\Phi_{\mathrm{out}}^{\mathsf T}
\odot g_{\mathrm{out}},
```

without materializing $W$ during the forward pass. Here $\odot$ denotes
broadcast elementwise multiplication.

The available continuous profiles are

```math
\kappa_{\mathrm{Gaussian}}(u,v)
=
\exp\left(-\frac{\lVert u-v\rVert_2^2}{2\sigma^2}\right),
\qquad
\kappa_{\mathrm{triangular}}(u,v)
=
\max\left(0, 1-\frac{\lVert u-v\rVert_2}{\sigma}\right).
```

Gaussian atoms have global support, so the represented $W$ is generally
dense even though it is parameterized by only $M$ atoms. Triangular atoms
have compact support and can produce exact zeros. Thus “sparse” primarily
describes the finite atomic representation and its structural lifecycle; it
does not imply that every supported kernel produces a sparse materialized
matrix.

Ordinary PyTorch autograd optimizes the live amplitudes $w_a$, continuous
endpoints $s_a,t_a$, neuron gates, and optionally the bandwidth $\sigma$.
Neuron chart coordinates $\mu$ are fixed buffers in the current
implementation. Separately, a clock-driven structural policy decides when
atoms or neurons are born, retired, merged, or ungated. This separates
continuous parameter optimization from discrete changes to model structure.

### Choosing a compute module

> [!IMPORTANT]
> Use `CSTLinear` for new continuous sparse training experiments
> (`CSTConv2d` for convolutional layers). `EntryLinear` and `RankOneLinear`
> are deprecated as primary modeling APIs: they remain public only as
> comparison baselines and compatibility controls, not as implementations of
> the central CST representation.

CST investigates parameter-efficient training by evolving a finite set of
continuous synapse atoms instead of assigning an independent parameter to
every dense matrix entry. In that sense, it is a continuous-coordinate member
of the broader dynamic sparse training (DST) family. The effective Gaussian
weight can still be dense; the sparsity is in the number of stored, trainable,
and structurally managed atoms.

`EntryLinear` is the discrete delta-kernel control. Conceptually, it is an
ordinary unstructured DST or masked sparse-entry model: each live atom selects
one matrix entry, although the implementation stores live entries as atoms
rather than as a dense parameter plus a literal mask.

`RankOneLinear` is the factorized control. Each atom contributes a rank-one
outer product, making it a low-rank/LoRA-like parameterization rather than
continuous sparse training. It is useful for comparison but is not identical
to LoRA, which commonly parameterizes an additive update to a separate base
weight.

Both controls share CST's storage and policy lifecycle so experiments can
compare representation families without changing the surrounding machinery.
Neither control owns neuron state; experiments that require neuron gates or
endpoint-aware response compose an explicit `NeuronGatedLinear` wrapper.

### Stacking layers, and who owns a neuron's gate

A neuron's gate belongs to the neuron and must be applied exactly once. A raw
`CSTLinear` applies both of its endpoints' gates itself, which is right for a
single map but wrong for a stack: wiring two together gates the shared hidden
store twice, so a neuron enters the composed function as `gamma^2`, its
derivative at a dormant `gamma=0` is identically zero, and **no dormant neuron
in a deep network can ever be woken**. Compose with `CSTBlock` instead — a
gate-free map, an activation, then the producing store's gate, once:

```python
block1 = CSTBlock(
    CSTLinear(inputs, hidden, syn1, kernel,
              gate_input=False, gate_output=False),
    activation=F.gelu,
)
block2 = CSTBlock(
    CSTLinear(hidden, outputs, syn2, kernel,
              gate_input=False, gate_output=False)
)
logits = block2(block1(x))
```

`gate_input=False, gate_output=False` does not remove gating; it moves it to
the block, which is what keeps the composition linear in `gamma`. A terminal
map that no CST layer consumes keeps the default `True` and gates itself.

> [!IMPORTANT]
> `NeuronStore.gate` is an `nn.Parameter`, but nothing puts it in your
> optimizer for you. A neuron woken by a growth policy is solved once and then
> never moves again unless you train it — its gate can sit near zero forever
> while the chart still reports it live. Pass the neuron stores' parameters to
> your optimizer, at a **smaller step than the synapse rate** (about a tenth in
> exploratory MNIST runs): a gate is a low-curvature direction, and the rate
> that suits synapse coordinates drives it to run away.

Dormant rows are masked out of `gate_vector()`, so they receive exactly zero
gradient — ordinary training can never wake a neuron by itself, which is the
policy tree's job. Training only ever moves gates the policy already opened.

When diagnosing a growth run, count *effective* width (live rows whose gate is
not near zero), not live rows. The two can differ by a factor of ten, and a
comparison against a fixed-width control is meaningless when they do.

> [!IMPORTANT]
> **Center the pre-activation of every stacked block.** A `LayerNorm` before the
> block's activation is enough, and a learnable bias is **not** — once the layer
> is dead the bias receives no gradient either and cannot climb out. A CST layer
> has no bias and no normalization of its own, so nothing keeps the sum of its
> atoms centered, and a stack fails in both directions without one: narrow
> coordinate domains saturate the activation and freeze training at chance,
> wide ones diverge outright. In exploratory MNIST runs exactly one domain
> width trained at all without centering; with it, every width did.

A single CST layer feeding a dense head does not need this — its head absorbs
what a CST consumer does not — which is why the failure only appears once you
stack.

### Choosing the coordinate domain

`bounds` is the coordinate range every atom and neuron chart lives on, and the
kernel resolves two points only if they are more than about one `sigma` apart.
So what a layer can represent is set by the *ratio* of the domain's extent to
`sigma`, not by how many atoms or chart rows you allocate:

```python
RepresentationSpec.continuous(2, 1, bounds=(0.0, 3.0))   # 30 sigma-widths
mu = torch.linspace(0.0, 3.0, H_MAX)[:, None]            # chart spans the same range
```

> [!IMPORTANT]
> Size the domain at roughly **20–30 `sigma` widths**, and give the neuron
> charts the same span. The default `bounds=(0.0, 1.0)` with `sigma=0.1` leaves
> only about ten distinguishable positions, which caps the layer no matter how
> many atoms it grows — in exploratory MNIST runs, widening the domain alone
> moved accuracy from 0.874 to 0.93 with `sigma` untouched. Too wide fails the
> other way: atoms can no longer cover the space, births stop being accepted,
> and the live count collapses.
>
> This is a choice you make once, when you build the stores and charts. The
> engine also projects live coordinates back inside the domain on every
> structural event, but that is worth little: letting coordinates leave scored
> slightly *better* on 10 of 12 seeds. Get the initial span right; do not count
> on the projection to rescue a badly sized one.

Both failure modes are visible without a validation set. Too narrow shows up as
accuracy that stops responding to more atoms; too wide shows up as a live-atom
count that stalls far below its birth budget.

The current implementation is built around five explicit responsibilities:

```text
compute  -> PyTorch forward and optional backward observation capture
policy   -> cadence, quota, observations, actions, distribution, and profit
engine   -> update lifecycle and ordered structural-event orchestration
storage  -> versioned two-phase mutation of synapse and neuron stores
audit    -> read-only event records and experiment accounting
```

Structural mutation is ID-based. Physical slots may be reused, while entity
IDs and lineages remain stable enough for replay, retirement, and optimizer
state reconciliation.

## Installation

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

Python 3.10 or newer and PyTorch 2.0 or newer are required.

## Minimal lifecycle

```python
import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import LC
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseStore

inputs = NeuronStore(
    "inputs",
    4,
    mu=torch.linspace(0.0, 1.0, 4)[:, None],
    initial_live=4,
)
outputs = NeuronStore(
    "outputs",
    3,
    mu=torch.linspace(0.0, 1.0, 3)[:, None],
    initial_live=3,
)
synapses = SynapseStore(
    "layer",
    d_in=1,
    d_out=1,
    capacity=16,
    spec=RepresentationSpec.continuous(1, 1),
)
layer = CSTLinear(inputs, outputs, synapses, GaussianKernel(0.2, learnable=True))
policy = LC(
    event_interval=1,
    birth_end_event=4,
    birth_budget=2,
    freeze_event=8,
    initial_weight=1e-2,
)
optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
engine = StructuralEngine(
    {"layer": synapses, "inputs": inputs, "outputs": outputs},
    policy,
    modules={"layer": layer},
    optimizer=optimizer,
    seed=0,
)

x = torch.randn(8, 4)
target = torch.randn(8, 3)

engine.begin_update()
optimizer.zero_grad()
loss = (layer(x) - target).square().mean()
loss.backward()
engine.observe_microbatch()
engine.finalize_backward()
optimizer.step()
applied_ops = engine.step()
```

`LC` means “lifecycle champion.” It is a convenience preset retained from the
internal 5c experiments, not a gradient optimizer and not a required part of
`CSTLinear`. Adam updates the differentiable values; `LC` controls discrete
structure at `engine.step()` by composing a cadence, an operation quota, birth
and prune rules, and a budget distributor.

In the configuration above, a structural event occurs after every optimizer
update. Events 1–4 may add up to two uniformly sampled continuous atoms per
event. Events 5–7 add no atoms but continue the rent-based cleanup sweep, and
event 8 onward is structurally frozen. The default rent rule protects a
newborn atom for three events and then removes it after two consecutive events
below 30% of the live population's median functional mass.

These `LC` constants are a historical catalog preset, not universal CST
hyperparameters. Continuous-kernel experiments should calibrate retention
thresholds and immunity for their kernel profile, bandwidth, coordinate
domain, and training timescale. Users can instead compose `Policy` directly;
see the [policy authoring guide](docs/policy-authoring.md).

Each observation request declares its execution timing. `after_backward`
keeps detached module-boundary `x`/`g_out` tensors and measures them in
`finalize_backward()`; `backward_inline` computes sufficient statistics in the
tensor hook and retains only the smaller payload. Different instruments at the
same site may use different timings.

`capture_mode` is an optional global/per-site experiment override, not the
policy's primary timing declaration:

```python
engine = StructuralEngine(
    stores,
    policy,
    modules=modules,
    capture_mode={
        "large_conv": "inline_reduced",
        "small_head": "deferred",
    },
)
```

The ordering is part of the API contract:

```text
begin_update
  -> forward/backward
  -> observe_microbatch (once per accumulated microbatch)
  -> finalize_backward
  -> optimizer.step
  -> engine.step
```

`finalize_backward()` releases captured tensors before the optimizer and
structural mutation boundaries. Policies that do not request observations add
no tensor hooks, but they use the same lifecycle.

## Authoring a policy

Policies compose cadence, logical quota, observations, action rules, and a
budget distributor. The engine—not a user-supplied planner—assembles their
operations into a validated structural plan. A
gradient-scored Gaussian birth requires no `StructuralEngine` modification:

```python
from torchcst.instruments import ContinuousGradientRequest
from torchcst.policy import ScoredBirth, TopKSelector

scores = ContinuousGradientRequest(
    pool_size=4096,
    decay=0.9,
    chunk_size=128,
    timing="backward_inline",
)
birth = ScoredBirth(scores, selector=TopKSelector(), initial_weight=0.0)
```

The request builds one instrument per site. Inline instruments implement
`reduce_backward`; deferred instruments implement `measure_after_backward`.
Both feed the common `finalize_update` boundary. `ScoredBirth` reads finalized
scores, accepts its distributed structural quota, selects top candidates,
allocates continuous lineages, and emits `SynapseBirth` operations.

Third-party observation requests implement `build(context)`. Their instruments
implement `prepare`, one timing-specific measurement method, and
`finalize_update`; the engine has no instrument-class registration step. See the
[`policy authoring guide`](docs/policy-authoring.md), the
[`capture lifecycle`](docs/capture-lifecycle.md), and the executable
[`Gaussian scored-birth example`](examples/gaussian_gradient_birth.py).
The [`framework design map`](docs/framework-design.md) records the current
responsibility boundaries, storage behavior, operation support matrix, and the
recommended extension checklist.

## Policy trees (v2 API, Phase 1 preview)

The policy tree is the successor authoring surface described in
[`docs/policy-tree-design.md`](docs/policy-tree-design.md): one root node owns
the coordination mechanism (and the knobs that go with it — the cadence, and
either a shared rent `lam` or an operation-count `budget`), while named method
constructors describe what happens at each site. Phase 1 ships it as a
compile-down adapter: `root.compile(stores)` folds the tree into the exact
composed `Policy` that `StructuralEngine` already runs, so nothing about the
engine, the update ordering, or existing composed policies changes.

```python
from torchcst.policy import (
    QuotaRegime,
    RentEconomy,
    RENT,
    cRES,
    cSET,
    thinned,
    PeriodicCadence,
)

# Central quota distribution, one method broadcast to every site.
root = QuotaRegime(
    budget=2,
    method=cSET(drop_fraction=0.3),
    cadence=PeriodicCadence(event_interval=1),
)

# Rent economy: the root alone owns lam; one site family is overridden.
root = RentEconomy(
    lam=1e-6,
    method=RENT(radius=0.01),
    cadence=PeriodicCadence(event_interval=1, observe_window=1),
    overrides={"stage2.*": thinned(cRES(), every=2)},
)

policy = root.compile(stores)  # a current-contract composed Policy
engine = StructuralEngine(stores, policy, modules=modules, seed=0)
```

The vocabulary is layered so most users never leave level 3:

1. **Named methods** — `cSET()`, `cRigL()`, `cRES()`, `RENT()` are plain
   functions returning a `SynapseLifecycle`; the difference between two
   methods reads as a code diff. They accept only their own selection-rule
   internals (pool size, drop fraction, thresholds) — never `lam`, never a
   cadence.
2. **Composition** — `SynapseLifecycle(birth_factory=..., ...)`, for authors
   of new methods.
3. **Op data** — engine-only; humans do not write it.

Rules the tree enforces at construction time rather than by convention:

- **Cadence is root-only.** A child may thin (`thinned(lifecycle, every=2)`
  fires on every second root-issued event) but can never carry a competing
  schedule.
- **Family compatibility is checked eagerly.** `RentEconomy(method=cSET())`
  raises immediately: `cSET`'s uniform birth has no per-candidate price tag to
  gate on, so it cannot join a rent economy. Conversely `RENT()` refuses to
  build under `QuotaRegime`/`Independent`, whose roots carry no `lam`.
- **`requires` is aggregated for you.** Each lifecycle's rules declare their
  own instrument requirements; the compiled `Policy` carries their union.
- **Override globs must match.** Passing `stores` to `compile` turns a typo'd
  pattern (`overrides={"stage9.*": ...}`) into an error instead of a silent
  no-op.

`cSET`, `cRigL`, `cRES`, `RENT`, `SynapseLifecycle`, `thinned`, and the three
root types are all exported directly from `torchcst.policy` — the vocabulary
is symmetric, with no separate catalog spelling to disambiguate.

## Implemented surface

- continuous Gaussian or compact-support triangular `CSTLinear` with learnable
  atom coordinates, amplitudes, and kernel bandwidth
- independent `CSTConv2d`, which owns a continuous CST filter shared across
  image locations and supports stride, zero padding, and dilation
- discrete-entry and rank-one/LoRA-like control families using the same engine
- opt-in `NeuronGatedLinear` composition for gated control-family experiments
- `SynapseStore` and `NeuronStore` with versioned prepare/commit mutation
- slot reuse, capacity growth, age/lineage columns, and follower notifications
- optimizer-state growth, reset, and coordinate-domain projection
- lifecycle (`LC`), anti-subspace, response, and merge catalog policies, plus
  the `cSET`/`cRigL`/`cRES`/`RENT` policy-tree method vocabulary
- public observation-instrument factories and high-level scored-birth policy
  composition for third-party policies
- atomic cross-store `ProposalBundle` application
- opt-in realized-profit trials with complete rollback and finite polish
- event audit, accounting, deterministic RNG streams, batch tapes, and replay

Deliberately unsupported paths raise explicit errors. These currently include
synapse/neuron kick operations, entry-family merge, generic composed-policy
neuron birth, and distributed structural coordination. Coupled neuron and
synapse birth can be authored today as a whole `StructuralPolicy` emitting an
atomic `ProposalBundle`.

## Repository layout

```text
examples/               # small executable public-API examples
docs/                   # user guides, lifecycle reference, and design history
src/torchcst/
├── audit/           # immutable event records and aggregate accounting
├── compute/         # CSTLinear/CSTConv2d plus controls and capture
├── instruments/     # gradient fields and certificate subspaces
├── lab/             # deterministic experiment/replay helpers
├── policy/          # cadence, quotas, actions, distributors, courts, catalog
├── representation/  # coordinate domains, kernels, family specification
├── storage/         # slot mechanics and mutable entity stores
├── engine.py        # StructuralEngine lifecycle and event orchestration
└── optim.py         # parameter groups and optimizer-state follower
```

The tests under `tests/torchcst/` are the executable contract. Design documents
under `docs/` record both the current v4 design and older decision history; see
[`docs/README.md`](docs/README.md) before treating a design note as current API
documentation.

This repository contains framework code and executable framework contracts
only. Research runners, preregistrations, raw results, figures, and result
reports live in the sibling `cst` experiment repository.

## Development principles

- Schedule-issued budgets are the only source of structural growth.
- Ordinary policies remain loss-blind; only an opt-in profit trial can evaluate
  an objective.
- Forward/backward capture cannot directly mutate structure.
- Cross-store atomic units prepare completely before any store commits.
- Audit subscribers are one-way sinks and cannot affect policy decisions.
- A failed policy event may consume its clock tick, but capture state is always
  closed before the next update.
