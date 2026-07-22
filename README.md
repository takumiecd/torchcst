# torchcst — Continuous Sparse Training in PyTorch

`torchcst` is an experimental PyTorch library for continuous sparse training.
Its primary compute abstraction is `CSTLinear`: a linear map represented by
learnable synapse atoms in continuous coordinate domains and composed through
kernel matrices instead of a materialized dense weight matrix.

```text
W = K_out(mu_out, t) diag(w) K_in(mu_in, s)^T
```

The atom coordinates `s` and `t`, amplitudes `w`, and Gaussian bandwidth can be
optimized by ordinary PyTorch autograd. A clock-driven policy separately
decides when atoms or neurons are born, retired, merged, or ungated.

`EntryLinear` and `RankOneLinear` are useful special cases and control families,
not the intended center of the library. `EntryLinear` is the discrete
delta-kernel sparse-entry path. `RankOneLinear` is the low-rank/LoRA-like path.
They share the same storage and policy lifecycle so CST can be compared against
them without changing the experiment machinery. Neither control owns neuron
state; experiments that need neuron gates or endpoint-aware response compose an
explicit `NeuronGatedLinear` wrapper.

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

## Implemented surface

- continuous Gaussian `CSTLinear` with learnable atom coordinates, amplitudes,
  and kernel bandwidth
- independent `CSTConv2d`, which owns a continuous CST filter shared across
  image locations and supports stride, zero padding, and dilation
- discrete-entry and rank-one/LoRA-like control families using the same engine
- opt-in `NeuronGatedLinear` composition for gated control-family experiments
- `SynapseStore` and `NeuronStore` with versioned prepare/commit mutation
- slot reuse, capacity growth, age/lineage columns, and follower notifications
- optimizer-state growth, reset, and coordinate-domain projection
- lifecycle (`LC`), anti-subspace, response, merge, cSET, and cRigL policies
- public observation-instrument factories and high-level scored-birth policy
  composition for third-party policies
- atomic cross-store `ProposalBundle` application
- opt-in realized-profit trials with complete rollback and finite polish
- event audit, accounting, deterministic RNG streams, batch tapes, and replay

Deliberately unsupported paths raise explicit errors. These currently include
synapse/neuron kick operations, entry-family merge, and distributed structural
coordination.

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
