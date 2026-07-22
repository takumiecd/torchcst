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
policy   -> schedule, proposal, allocation, retention, and profit decisions
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

## Implemented surface

- continuous Gaussian `CSTLinear` with learnable atom coordinates, amplitudes,
  and kernel bandwidth
- `CSTConv2d`, which shares a `CSTLinear` patch map across image locations and
  supports stride, zero padding, and dilation
- discrete-entry and rank-one/LoRA-like control families using the same engine
- opt-in `NeuronGatedLinear` composition for gated control-family experiments
- `SynapseStore` and `NeuronStore` with versioned prepare/commit mutation
- slot reuse, capacity growth, age/lineage columns, and follower notifications
- optimizer-state growth, reset, and coordinate-domain projection
- lifecycle (`LC`), anti-subspace, response, merge, cSET, and cRigL policies
- atomic cross-store `ProposalBundle` application
- opt-in realized-profit trials with complete rollback and finite polish
- event audit, accounting, deterministic RNG streams, batch tapes, and replay

Deliberately unsupported paths raise explicit errors. These currently include
synapse/neuron kick operations, entry-family merge, and distributed structural
coordination.

## Repository layout

```text
src/torchcst/
├── audit/           # immutable event records and aggregate accounting
├── compute/         # CSTLinear/CSTConv2d plus controls and capture
├── instruments/     # gradient fields and certificate subspaces
├── lab/             # deterministic experiment/replay helpers
├── policy/          # contracts, schedules, proposers, courts, catalog
├── representation/  # coordinate domains, kernels, family specification
├── storage/         # slot mechanics and mutable entity stores
├── engine.py        # StructuralEngine lifecycle and event orchestration
└── optim.py         # parameter groups and optimizer-state follower
```

The tests under `tests/torchcst/` are the executable contract. Design documents
under `docs/` record both the current v4 design and older decision history; see
[`docs/README.md`](docs/README.md) before treating a design note as current API
documentation.

The `CONV-GROW-4` protocol reproduction using the continuous family is runnable
with local MNIST IDX files:

```bash
python experiments/e2e_mnist_conv.py --smoke --data-dir ../cst/data/MNIST/raw
python experiments/e2e_mnist_conv.py --device mps --data-dir ../cst/data/MNIST/raw
```

This keeps the original topology, tape family, K ladder, and birth timing, but
replaces the original per-offset rank-one filter with `CSTConv2d`/`CSTLinear`.
Its JSON labels that distinction explicitly; it is a protocol reproduction,
not a same-family numerical replication.

## Development principles

- Schedule-issued budgets are the only source of structural growth.
- Ordinary policies remain loss-blind; only an opt-in profit trial can evaluate
  an objective.
- Forward/backward capture cannot directly mutate structure.
- Cross-store atomic units prepare completely before any store commits.
- Audit subscribers are one-way sinks and cannot affect policy decisions.
- A failed policy event may consume its clock tick, but capture state is always
  closed before the next update.
