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

## Design principle

**CST gives coordinates to discrete, non-differentiable things — which
synapse exists, between which neurons — so that the discrete structure
becomes differentiable.** Sparse training's hard core is a combinatorial
support-search problem; masks cannot be relaxed (a relaxed mask is just a
dense matrix), so CST relaxes the *positions* instead: an entry's identity
becomes a continuous coordinate, and gradient descent can move it.

That relaxation is only real under four operating conditions, each measured
the hard way (2026-08 conv arc; detailed reports live in the sibling `cst`
experiment repository):

1. **Grounding** — coordinates must parameterize a *role in the data*, not a
   weight index. Moving a coordinate must continuously change what the atom
   reads; a coordinate moving through data-free vacuum (e.g. between the 9
   taps of a 3×3 conv kernel) carries only noise, and its atoms diffuse,
   flee, or freeze instead of localizing.
2. **Operating envelope** — the capacity law *capacity = extent/σ* has a
   measured sweet spot (≈10σ per axis for multi-dimensional hidden charts).
   Outside it, coordinate learning, scored birth, and growth all silently
   degrade to noise; inside it, all of them work at once.
3. **Factorization** — the envelope is only affordable on factorized charts:
   widening a joint chart multiplies its resolvable cells per axis until no
   atom budget can cover the volume. Factorization turns that product into
   a sum.
4. **Gradient clipping** — coordinate gradients scale as 1/σ; clipping is a
   family default, not an option.

### The casting rule

Factorize a layer, then cast each factor by its arity:

> **Many-to-many factors (product spaces) go to CST; one-to-many and
> many-to-one factors stay dense.**

Parameter explosion happens only in product spaces (N×M), and that volume
is what CST's compression pays for. A one-to-many / many-to-one factor is
already near-minimal — its parameter count is linear in the larger side —
so charting it buys nothing: an atom's price (coordinates plus amplitude)
exceeds the values it would replace. The measured embodiment is
`DepthwiseCSTConv2d`: the spatial `{h,w} → 1` factor (9 values per channel,
many-to-one) is a plain depthwise conv, the channel `{in} → {out}` factor
(the product space) is CST on lawful charts — a composition that beat every
anchor of the conv arc, including the standalone factorized winner, at ~10k
parameters. (Boundary note: the underlying variable is really the factor's
*volume and structure* — a hypothetical many-to-one with a huge, smooth
"many" side could still pay — but in practice a network's many-to-one
factors are always small, so the arity rule and the volume arithmetic
agree.)

The rule casts *chart* placement, not coordinates as such. The FC-9 arc's
`OffsetCSTConv2d` removes even the factorization: one atom lives on the
product domain `input-channel chart × displacement box`, its spatial
displacement Δ acting on the **data side** as a bilinear read position
(dense position gradients by construction — no tap-lattice vacuum) while
only the channel axes carry kernels. Measured at parity it matches
`DepthwiseCSTConv2d` at every width, scales monotonically on the atom
ladder where the separable form's budget law stalls, and its forward
materializes the equivalent dense kernel per call so the conv itself runs
at `nn.Conv2d` cost, independent of the atom count. Because Δ is two extra
axes of the source coordinate, birth candidates, scored birth, and box
retraction apply unchanged.

The conditions a machine can check are shipped as executable instruments.
`torchcst.representation.propose_chart` returns a chart that is lawful *by
construction* — box, dimension, and atom budget — whose `Box.sample` is the
measured winning initialization; build hidden sites through it and no
further check is needed. `survey_chart` is the optional diagnostic for
charts you did **not** propose: hand-built ones, data-pinned ones (pixels,
conv taps), or charts inherited from an experiment you are debugging. Every
empirical threshold in both instruments is a keyword argument with the
measured value as its default.

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

Here $\odot$ denotes broadcast elementwise multiplication. Whether this
factorized expression is evaluated directly, or $W$ is built once per
forward and applied as one GEMM, is an execution-backend choice — see
[Execution backends](#execution-backends) below; the represented map is
identical either way.

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

### Execution backends

One measure, several ways to apply it. `torchcst.compute.backends` is the
pure-function world (tensors in, tensors out; it never sees stores, capture,
or the engine), and every compute module takes a `backend=` argument:

| Backend | W | Costs and role |
|---|---|---|
| `Factored()` | never built | `((X Φ_in) ⊙ w) Φ_out^T`: `K(d_in+d_out)` FLOPs/row and a `[rows, K]` autograd intermediate — right only below the crossover `K < d_in d_out/(d_in+d_out)` |
| `Materialized(lean=…, compute_dtype=…)` | built per forward | one cuBLAS GEMM / cuDNN conv; ceiling = dense speed, activation memory = dense. `lean=True` replaces the build's autograd with a closed-form chunked backward: peak O(chunk) at any K. The short-term workhorse |
| `NativeTruncated(radius=…)` | never built | kernels truncated at `radius·σ`, rows routed through per-atom neighbor tables: `K(m_in+m_out)` FLOPs/row — **below the dense GEMM itself** in the lawful wide-domain regime, which neither other backend can reach. The long-term mainline; the PyTorch implementation is the semantics oracle a fused CUDA/Triton kernel must match |

`backend="auto"` (the default) switches Factored/Materialized per forward on
the live atom count. `OffsetCSTConv2d` accepts only `Materialized`: its
measure is applied through `F.conv2d`, so the kernel is materialized by
construction — a gather-style native conv (deformable-convolution-shaped) is
future work.

Under every backend, W is a compute intermediate and never state: parameters,
gradients, and optimizer moments live on the atoms. `dense_weight()` stays
public regardless of backend — it is the analysis surface for inspecting the
weight the atoms currently represent.

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
`CSTLinear` is purely synaptic and never applies a gate itself: wiring two maps
together by gating each map's own endpoints would gate the shared hidden store
twice, so a neuron enters the composed function as `gamma^2`, its derivative
at a dormant `gamma=0` is identically zero, and **no dormant neuron in a deep
network can ever be woken**. Compose with `CSTBoundary` instead — it takes a
map's output and applies normalize/activation, then the producing store's
gate, exactly once:

```python
m1 = CSTLinear(inputs, hidden, syn1, kernel)
m2 = CSTLinear(hidden, outputs, syn2, kernel)
b1 = CSTBoundary(m1, activation=F.gelu)
b2 = CSTBoundary(m2, activation=F.gelu)
logits = b2(m2(b1(m1(x))))
```

Composition is explicit at the call site — `boundary(map(x))` — which is what
keeps the composition linear in `gamma`. A terminal map that no further CST
layer consumes is wrapped in a boundary with no activation:
`CSTBoundary(m)`, whose forward is simply `pre * gate`.

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
> **Clip the gradient norm.** A coordinate gradient carries a factor of
> `1/sigma` that an amplitude gradient does not, so the two never share a
> magnitude: measured on MNIST at `sigma=0.1`, coordinates reached a norm of
> 4.5e4 against 823 for amplitudes, spiking after every structural event.
> Unclipped runs diverge seed-dependently. `clip_grad_norm_(params, 5.0)` over
> the whole parameter set is what the FASTCON experiments use. Giving
> coordinates a smaller learning rate is not a substitute — a fifth of the rate
> does not answer a fiftyfold gap.
>
> **Center the pre-activation of every stacked block** as well. A `LayerNorm`
> before the block's activation is enough, and a learnable bias is **not** —
> once the layer is dead the bias receives no gradient either and cannot climb
> out. A CST layer has no bias and no normalization of its own, so nothing keeps
> the sum of its atoms centered, and a narrow domain will saturate the
> activation and freeze training at chance. With clipping in place this is worth
> one to two points at a well-sized domain, and much more at a badly sized one.

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

## From proposal to training: the full flow

The design principle above is executable end to end. For a hidden site —
one whose populations have no data-pinned geometry — you never hand-place
chart coordinates or size boxes yourself: `CSTLinear.propose` runs the
propose → sample → wire flow in one call. This example runs as shown:

```python
import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import (
    ConstantQuota,
    EvenBudgetDistributor,
    PeriodicCadence,
    QuotaRegime,
    StructuralQuota,
    cSET,
)
from torchcst.representation import GaussianKernel
from torchcst.storage import NeuronStore, SynapseStore

# A CST layer is a composition, not a primitive.  Neurons come first: for a
# hidden population, propose() builds a lawful chart (a 2-D box spanning
# 10 sigma per axis) and samples the coordinates uniformly from it -- the
# measured winning placement.  Every envelope constant is an argument
# (e.g. ``axis_extent=``) with the measured value as its default.
gen = torch.Generator().manual_seed(0)
inputs = NeuronStore.propose("layer.in", 64, sigma=0.1, generator=gen)
outputs = NeuronStore.propose("layer.out", 32, sigma=0.1, generator=gen)

# Synapses depend on the neurons they connect: between() reads its domains
# back from the actual per-axis extent of each chart and sizes capacity at
# one atom per resolvable cell, so every future birth lands inside the
# envelope -- whether the charts were proposed or data-pinned.
synapses = SynapseStore.between("layer", inputs, outputs, sigma=0.1)

# The layer merely applies the composed site.
layer = CSTLinear(inputs, outputs, synapses, GaussianKernel(0.1))

# The policy tree is the current authoring surface: a named method under a
# root that owns cadence, quota, and distribution.  Note the distributor:
# the QuotaRegime default is replacement-only (SET-style rewiring at
# constant density), which never grows an initially empty store — pass a
# growth distributor when starting from zero atoms.
root = QuotaRegime(
    budget=16,
    method=cSET(initial_weight=1e-2),
    cadence=PeriodicCadence(event_interval=1),
    quota=ConstantQuota(StructuralQuota(synapse_birth=16)),
    distributor=EvenBudgetDistributor(),
)

optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
engine = StructuralEngine(
    layer.stores(),                       # {"layer", "layer.in", "layer.out"}
    root,
    modules={layer.capture_site: layer},
    optimizer=optimizer,
    seed=0,
)

rng = torch.Generator().manual_seed(1)
teacher = torch.randn(64, 32, generator=rng) * 0.1
for step in range(20):
    x = torch.randn(128, 64, generator=rng)
    engine.begin_update()
    optimizer.zero_grad()
    loss = (layer(x) - x @ teacher).square().mean()
    loss.backward()
    engine.observe_microbatch()
    engine.finalize_backward()
    optimizer.step()
    engine.step()
```

A proposed-and-sampled chart is lawful by construction and needs no
further check. `survey_chart` is the *optional* diagnostic for every other
chart: for a data-pinned site (image pixels, conv taps) the data supplies
the coordinates, and the survey tells you which axes are genuine continua
and which are quasi-discrete and must be handled by policy rather than SGD.

`torchcst.policy.recipes` holds only assemblies in active research use —
currently `FastConstruction` (cSFW-grow into solve-driven operation). The
historical 5c presets (`LC`, `LC_response`, `GrowthByProfit`) were removed
from the public surface; the mechanism test suite keeps a private copy of
`LC` as scaffolding. New work composes the tree directly, as above.

## Observation timing

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

## Adding a backward statistic

The policy tree (next section) is the authoring surface for structural
behavior; extension of the *observation* side is independent of it. A new
backward statistic is a frozen request plus an instrument — no
`StructuralEngine` modification and no registration step. A gradient-scored
Gaussian birth:

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
`finalize_update`; the engine has no instrument-class registration step. See
the [`capture lifecycle`](docs/capture-lifecycle.md) and the executable
[`Gaussian scored-birth example`](examples/gaussian_gradient_birth.py).
The [`framework design map`](docs/framework-design.md) records the current
responsibility boundaries, storage behavior, operation support matrix, and the
recommended extension checklist.

## The policy tree

The policy tree is the sole authoring surface
([`docs/policy-tree-design.md`](docs/policy-tree-design.md),
[`docs/policy-tree-phase2.md`](docs/policy-tree-phase2.md)): one root node
owns the coordination mechanism (and the knobs that go with it — the cadence,
and either a shared rent `lam` or an operation-count `budget`), while named
method constructors describe what happens at each site. `StructuralEngine`
binds the root directly; the older composed-`Policy` surface was removed with
the Phase 2 migration.

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

engine = StructuralEngine(stores, root, modules=modules, seed=0)
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
- the `cSET`/`cRigL`/`cRES`/`RENT` policy-tree method vocabulary and the
  `FastConstruction` recipe (cSFW-grow into solve-driven operation)
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
