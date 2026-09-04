# torchcst — fixed-shape continuous operators for PyTorch

> [!WARNING]
> **Ground-up research rewrite.** This branch defines the target contract and
> implements it in separately tested milestones. Imports may be unavailable
> until their milestone lands. There is intentionally no compatibility promise for earlier
> `SynapseStore`, structural-policy, or Pullback Adam APIs.

`torchcst` represents an operator as a weighted sum of a fixed number of atoms.
Each atom owns an amplitude and an opaque kernel coordinate that move
continuously. Training never changes which atoms exist. There is no birth,
death, merge, absorb, slot reuse, or structural optimizer-state remapping.

The initial scope is deliberately narrow:

- fixed atom count and tensor shapes for the lifetime of a module;
- continuous atom-local coordinates and amplitudes;
- fixed-cardinality input and output charts whose coordinates are frozen by
  default;
- an implicit projected Adam optimizer;
- a full second-order CST displacement and unapproximated quartic objective;
- exact-loss trust-region acceptance;
- no persistent dense represented-weight moments.

The selected mathematical and experimental decisions are recorded in
[`docs/implicit-projected-adam-decisions.ja.md`](docs/implicit-projected-adam-decisions.ja.md).

## Target API

The README example is the public contract for the rewrite.

```python
import torch
import torch.nn.functional as F

from torchcst import (
    AdamWConfig,
    Chart,
    CSTLinear,
    CSTOptimizer,
    FullQuartic,
    Gaussian,
    ImplicitAdamConfig,
    Separable,
)

input_chart = Chart.grid((28, 28), trainable=False)
output_chart = Chart.linspace(10, trainable=False)

model = CSTLinear(
    input_chart,
    output_chart,
    atoms=64,
    kernel=Separable(
        input_profile=Gaussian(sigma=0.25),
        output_profile=Gaussian(sigma=0.10),
    ),
    atom_init="balanced",
    backend="auto",
)

optimizer = CSTOptimizer(
    model,
    cst=ImplicitAdamConfig(
        lr=0.05,
        betas=(0.9, 0.99),
        eps=1e-8,
        second_moment="separable",
        trust_radius=0.25,
        quartic=FullQuartic(starts=4, max_iter=80),
    ),
    dense=None,
)

for images, labels in loader:
    images = images.flatten(1)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        return loss

    loss = optimizer.step(closure)
```

This API makes seven ownership decisions explicit:

1. a `Chart` owns fixed-cardinality observation coordinates;
2. `Atoms` owns one fixed-shape amplitude vector and one opaque coordinate table;
3. a `Kernel` interprets atom coordinates but owns no trainable state;
4. a `CSTLinear` composes charts, atoms, and one kernel;
5. a `CSTOptimizer` partitions and exclusively owns every trainable parameter;
6. its implicit CST engine owns compressed moment and accepted-frame state;
7. a closure owns loss evaluation, backward, and candidate reevaluation.

There is no engine or structural policy between the module and optimizer.

## `Chart`

A chart is a fixed-cardinality set of observation points. It has no
live/dormant state, gate, lineage, allocator, or structural version. Its
coordinates are frozen by default and should normally remain frozen.

```python
pixels = Chart.grid((28, 28), trainable=False)
classes = Chart.linspace(10, trainable=False)
custom = Chart.points(
    torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
    trainable=False,
)
```

Target constructors:

```python
Chart.points(coordinates, *, trainable=False)
Chart.linspace(size, *, low=-1.0, high=1.0, trainable=False)
Chart.grid(shape, *, low=-1.0, high=1.0, trainable=False)
```

The coordinate tensor has shape `[features, dimensions]`. With
`trainable=False` it is a buffer; with `trainable=True` it is a parameter.
Neither mode permits resizing during training.

Trainable charts are retained as an experimental extension point, not as the
recommended path. Moving an external input chart weakens its grounding in the
data, and atom-local kernel coordinates can provide much of the continuous
support adaptation that previously motivated chart motion. Hidden
charts may eventually benefit from learned geometry, but that question is
separate from the initial optimizer result.

The first implicit CST engine milestone formally supports only frozen
charts. A trainable chart enlarges the local variable to

$$
d=(d_{\mathrm{atom}},d_\mu)
$$

and introduces atom/chart and chart/chart curvature blocks. A future
implementation must evaluate those blocks through JVPs, VJPs, and HVPs; it must
not materialize the full Hessian. Reserving `trainable=True` in the chart API
keeps that route open without imposing its cost on the recommended mode.

## `Atoms`

For `K` atoms and an opaque kernel-coordinate width `P`, `Atoms` owns exactly

```text
weight  [K]
p       [K, P]
```

`Atoms` knows the shapes and the correspondence between `weight[a]` and `p[a]`.
It does not know whether entries of `p` represent a source, target, bandwidth,
orientation, scale, or another kernel-specific quantity. Only the selected
`Kernel` interprets `p`.

Both tensors are parameters with fixed shapes and atom ordering. If an atom
property is trainable, it belongs in that atom's row of `p`; a kernel object does
not own shared trainable parameters.

## `CSTLinear`

The target constructor is

```python
CSTLinear(
    input_chart,
    output_chart,
    *,
    atoms,
    kernel,
    atom_init="balanced",
    backend="auto",
    device=None,
    dtype=None,
)
```

For `K = atoms`, the module asks `kernel` for its opaque coordinate width and
initial coordinates, then constructs one fixed-shape `Atoms` child. Tensor
shapes and atom ordering never change after construction.

The primary inspection surface is intentionally small:

```python
model.atoms.weight
model.atoms.p
model.kernel
model.dense_weight()       # diagnostic materialization only
model.extra_repr()
```

`forward(x)` accepts `[..., in_features]` and returns
`[..., out_features]`, matching `torch.nn.Linear` except for the missing bias
in the first implementation.

### Represented operator

Let the fixed charts be collected as $\mathcal C$, and let atom $a$ have opaque
kernel coordinate $p_a$ and amplitude $w_a$. The canonical represented operator
is

$$
\boxed{
W=\sum_{a=1}^{K}w_a\mathcal K(p_a;\mathcal C)
}.
$$

`CSTLinear` does not inspect $p_a$. It asks its single `Kernel` to evaluate
$\mathcal K(p_a;\mathcal C)$ and performs the weighted sum.

A separable kernel may additionally expose factors

$$
(\Phi_{\mathrm{in}})_{ia}
=\kappa_{\mathrm{in}}(\mu_i^{\mathrm{in}},s_a),
\qquad
(\Phi_{\mathrm{out}})_{ja}
=\kappa_{\mathrm{out}}(\mu_j^{\mathrm{out}},t_a)
$$

so the same canonical sum can be evaluated as

$$
W=\Phi_{\mathrm{out}}\operatorname{Diag}(w)\Phi_{\mathrm{in}}^\top,
\qquad
Y=((X\Phi_{\mathrm{in}})\odot w)\Phi_{\mathrm{out}}^\top.
$$

This factorization is an optional execution capability, not the semantic
definition of a CST operator.

No neuron gate is present. Capacity is selected by `atoms` at construction,
not by a discrete operation during training.

## Kernels

A `Kernel` is the stateless interpretation of one atom coordinate. Its contract
provides:

```python
kernel.parameter_dim(input_chart, output_chart)
kernel.initialize(input_chart, output_chart, atoms, mode=...)
kernel.materialize_atoms(input_chart, output_chart, p)
kernel.factors(input_chart, output_chart, p)  # optional capability
```

`materialize_atoms` returns one represented operator per atom. Production
backends may use a more structured kernel capability instead of materializing
those operators.

The first implementation composes scalar Gaussian profiles through one
separable operator kernel:

```python
Separable(
    input_profile=Gaussian(sigma=0.25),
    output_profile=Gaussian(sigma=0.10),
)
```

with

$$
\kappa_\sigma(u,v)
=\exp\left(-\frac{\lVert u-v\rVert_2^2}{2\sigma^2}\right).
$$

For this kernel, $p_a=(s_a,t_a)$ and the separable kernel alone knows that split.
The Gaussian bandwidths are fixed kernel configuration. A future trainable
bandwidth belongs in each atom's opaque $p_a$, preserving atom-locality.

Additional kernels must provide values and the derivative contractions needed
by the second-order displacement API. Merely implementing a forward value is
not sufficient for implicit optimization.

## Initialization

Initialization is a one-time continuous-model construction concern, not a
training-time mutation.

The initial modes are:

- `"balanced"`: ask the kernel to balance its output-facing coordinates and
  sample its remaining coordinates;
- `"uniform"`: ask the kernel to sample all of its coordinates;
- explicit tensors through a later `CSTLinear.from_atoms(...)` constructor.

Initialization fixes `K` and the parameter shapes. Changing capacity means
constructing a new module and optimizer.

## Execution backends

Execution backend and model semantics are independent:

| backend | behavior | role |
| --- | --- | --- |
| `"factored"` | uses an optional factorization supplied by the single kernel | low atom count |
| `"materialized"` | evaluates $\sum_a w_a\mathcal K(p_a)$ and calls a dense GEMM | oracle checks and kernels without a factorization |
| `"auto"` | selects between the two without changing the represented map | default |

Backends receive ordinary fixed-shape tensors. They know nothing about stores,
entity IDs, policies, or optimizer state.

`dense_weight()` remains available for diagnostics and small correctness
oracles. A dense represented matrix is never persistent model or optimizer
state.

## Atom-structured derivatives

For opaque coordinate width `P`, define each atom-local parameter as

$$
z_a=(w_a,p_a)\in\mathbb R^{P+1}.
$$

The internal derivative API consumes parameter points and directions with shape
`[K, P + 1]`. It evaluates displacement, JVP, VJP, and Hessian contractions
without interpreting columns of $p$. With frozen charts, the contracted
representation Hessian stores only its nonzero atom blocks:

```text
pullback             [K, P + 1]
contracted Hessian   [K, P + 1, P + 1]
```

The dense correctness oracle may construct the mathematical full Hessian with
shape `[K, P + 1, K, P + 1]` and verifies that distinct-atom blocks are zero.
This representation sparsity does not remove cross-atom terms created later by
the full quartic objective.

## Represented gradients

Implicit optimization needs the cotangent of the represented operator itself,
not only the ordinary gradients already pulled back to `weight` and `p`. For a
linear site with $y=xW^\top$, each backward call contributes

$$
g_W=\sum_{\text{leading indices}} g_y^\top x.
$$

`CSTLinear` captures this value independently of whether its forward backend is
factorized or materialized. Capture is explicitly enabled only around the base
loss evaluation, accumulates repeated module calls and microbatches, and can be
disabled during candidate reevaluation. The captured tensor is transient: it is
cleared through the site lifecycle and is never part of `state_dict()`.

## `CSTOptimizer`

`CSTOptimizer` is the public model-level optimizer. It accepts the complete
model rather than an arbitrary iterable of tensors, discovers every CST site,
and partitions every remaining trainable parameter into its dense block.

```python
CSTOptimizer(
    model,
    *,
    cst=ImplicitAdamConfig(...),
    dense=AdamWConfig(...) | None,
    strict=True,
)
```

Users do not construct a separate dense optimizer. Internally, the coordinator
uses a strict CST-only implicit engine and a functional dense AdamW engine.
The latter is a proposal/state implementation owned by `CSTOptimizer`, not an
independently stepping `torch.optim.AdamW` instance.

For mixed models:

```python
optimizer = CSTOptimizer(
    model,
    cst=ImplicitAdamConfig(
        lr=0.05,
        second_moment="separable",
        trust_radius=0.25,
        quartic=FullQuartic(starts=4, max_iter=80),
    ),
    dense=AdamWConfig(lr=3e-4, weight_decay=0.01),
)
```

If the model contains dense trainable parameters while `dense=None`,
construction fails. A dense configuration with an empty dense partition is
allowed so model variants can share one experiment configuration.

### Parameter ownership

Each CST site explicitly declares the parameters it owns. Let their union be
$P_{\mathrm{cst}}$ and let all remaining trainable model parameters be
$P_{\mathrm{dense}}$. Construction verifies

$$
P_{\mathrm{cst}}\cap P_{\mathrm{dense}}=\varnothing
$$

and

$$
P_{\mathrm{cst}}\cup P_{\mathrm{dense}}
=\{p\mid p.\mathrm{requires\_grad}\}.
$$

The default `strict=True` also rejects duplicate ownership and trainable
parameters shared by multiple CST sites. Parameter names, shapes, and owners
are included in the optimizer state manifest and validated while loading a
checkpoint.

The internal implicit engine accepts CST sites only. It never accepts a raw
parameter iterable, so a dense tensor cannot accidentally enter the CST
solver through the supported API.

### Closure contract

`step(closure)` may evaluate the closure multiple times. The closure must:

1. clear gradients;
2. evaluate the current minibatch loss;
3. call `backward()`;
4. return the scalar loss.

The optimizer forms the CST quartic proposal and dense AdamW proposal from the
same base-point backward pass. It applies both provisionally, evaluates the
actual loss, and accepts or rejects them as one transaction:

$$
(\theta_{\mathrm{cst}},\theta_{\mathrm{dense}})
\longmapsto
(\theta_{\mathrm{cst}}+\lambda d_{\mathrm{cst}},
 \theta_{\mathrm{dense}}+\lambda d_{\mathrm{dense}}).
$$

The optimizer is responsible for restoring both blocks after a rejected
candidate. Parameters, dense AdamW moments, compact CST moments, and
accepted-frame metadata must not advance inconsistently.

### Local objective

For a parameter candidate $d$, the represented displacement is

$$
\Delta W_t(d)=J_td+\frac12H_t[d,d],
$$

with derivative

$$
V_t(d)=J_t+H_t[d,\cdot].
$$

The solved objective is

$$
\boxed{
\Phi_t(d)
=m_t^\top\Delta W_t(d)
+\frac1{2\eta}
\Delta W_t(d)^\top
\operatorname{Diag}(\sqrt{v_t}+\varepsilon)
\Delta W_t(d)
}.
$$

The initial correctness implementation retains every same-atom and cross-atom
term in this quartic. `FullQuartic` configures the numerical inner solve; its
name means that the objective is not block-truncated, not that a global
algebraic root is guaranteed.

### CST persistent state

The selected first implementation stores, per site,

$$
\boxed{
(\alpha_t,r_t,c_t,\theta_t,d_t^\star,\text{step})
}.
$$

- $\alpha_t$ is the accepted-frame compressed first moment;
- $r_t,c_t$ are row/column second-moment EMAs;
- $\theta_t,d_t^\star$ reconstruct the old visible frame;
- no dense $W$, $m$, or $v$ EMA persists between steps.

Here "no dense moments" refers to the potentially huge represented CST weight
$W$. Ordinary dense parameters retain their ordinary per-parameter AdamW
moments inside `CSTOptimizer`.

The separable second moment reconstructs

$$
\widetilde v_{oi}
=\frac{\widehat r_o\widehat c_i}
{\operatorname{mean}_o\widehat r_o},
\qquad
D=\operatorname{Diag}(\sqrt{\widetilde v}+\varepsilon).
$$

The second-moment implementation sits behind an internal operator interface so
that a future diagonal product-space state can replace the separable backend
without changing the optimizer's public API.

## Continuous-only invariant

During the lifetime of a module and optimizer:

```text
atom count       constant
parameter shapes constant
atom ordering    constant
chart count      constant
chart coordinates frozen by default
structural ops   forbidden
```

Continuous retraction, gauge normalization, trust-region scaling, and line
search are allowed. They do not create or remove model entities. Experimental
trainable charts may change coordinate values, but never chart cardinality or
tensor shape.

If future work needs a different atom count, it constructs a new model. A
separate offline conversion tool may eventually transfer values, but it is not
part of optimizer semantics.

## Evidence behind the selected baseline

The A100 MNIST K=64 experiment used 256 CST parameters, 128 optimizer steps,
and seeds 17/29/43:

| optimizer | mean test accuracy |
| --- | ---: |
| Adam-target quartic | 81.50% |
| dense moments, exact diagonal implicit | 81.48% |
| **compact $\alpha$ + separable diagonal $v$** | **81.12%** |

The compact state used 1,050 moment scalars instead of dense Adam's 15,680:
a 93.3% reduction. The full report is in the sibling `cst` experiment
repository at
`docs/experiments/mnist_wgate_compact_diag_ablation_a100.md` and was registered
in Arctx lane `wgate-compact-adam-diagonal`.

The experiment supports the selected starting point. It does not yet prove:

- a fully ambient-free production contraction kernel;
- an exact compact construction of
  $\operatorname{Diag}(\sqrt{\operatorname{EMA}(g^{\odot2})})$;
- a fast solver with the same solution quality as the full quartic oracle;
- behavior beyond the tested fixed-atom MNIST setting.

## Rewrite milestones

1. **API contract** — this README and focused interface tests.
2. **Legacy reset and fixed representation** — remove the dynamic architecture,
   then implement `Chart`, opaque `Atoms`, the single-`Kernel` contract, and
   fixed-shape `CSTLinear` on the new package boundaries.
3. **Derivative operators** — displacement, JVP/VJP, and second-order
   contractions checked against dense autograd.
4. **Correctness optimizer** — model-wide ownership, functional dense AdamW,
   compact $\alpha$, separable $v$, full quartic, and joint exact-loss
   acceptance.
5. **Dense-free backend** — fuse the required contractions without persistent
   or transient represented-weight tables in the production path.
6. **Solver work** — reduce quartic cost while measuring solution and training
   degradation against the correctness optimizer.

Each milestone must land as a separately testable commit. Numerical shortcuts
are evaluated only after the full objective is working.

## Explicit non-goals

The initial rewrite does not provide:

- dynamic sparsity or changing atom count;
- birth, death, merge, absorb, pruning, or growth schedules;
- neuron gates or dormant/live state;
- compatibility with old checkpoints or policy definitions;
- convolutional CST modules;
- trainable-chart support in the first implicit optimizer milestone;
- distributed training;
- a promise of global quartic optimality.

These omissions define the experiment, rather than a temporary compatibility
layer around the previous architecture.

## Development

Install the project in editable mode and run the focused tests:

```bash
python -m pip install -e '.[dev]'
pytest -q
```

Until the optimizer milestone lands, the complete target example is expected to
fail at its optimizer imports. Each API segment becomes a required test when its
milestone lands and must not drift from the implementation.

## License

See [`LICENSE`](LICENSE).
