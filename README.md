# torchcst — fixed-shape continuous operators for PyTorch

> [!WARNING]
> **Research API.** `CSTLocalAdam` is the primary optimizer; the full
> second-order algorithm remains an explicit option. There is intentionally no compatibility promise for earlier
> `SynapseStore`, structural-policy, or Pullback Adam APIs.

`torchcst` represents an operator as a sum of a fixed number of kernel-defined
atoms. Each atom owns one opaque parameter row that moves continuously, and
only its kernel interprets that row. Training never changes which atoms exist.
There is no birth, death, merge, absorb, slot reuse, or structural
optimizer-state remapping.

The initial scope is deliberately narrow:

- fixed atom count and tensor shapes for the lifetime of a module;
- continuous opaque atom-local parameters;
- fixed-cardinality input and output charts whose coordinates are frozen by
  default;
- an implicit projected Adam optimizer;
- a composable normalized optimizer family with independent numerator,
  denominator, and solver components;
- a primary first-order optimizer with current-tangent moment compression and transport;
- an optional second-order optimizer retaining the full quartic objective;
- regularized direct atom-local updates, with trust-region alternatives;
- compact optimizers without persistent dense represented-weight moments, plus
  an explicit dense-moment accuracy baseline.

The current optimizer design, equations and memory limits are recorded in
[the local Adam specification](docs/local-adam-math.ja.md). Earlier second-order
decisions remain in [the historical decision record](docs/implicit-projected-adam-decisions.ja.md).

## Basic use

The README example is the public API contract.

```python
import torch
import torch.nn.functional as F

from torchcst import (
    AdamWConfig,
    AmplitudeBandwidthSeparable,
    Chart,
    CSTLinear,
    CSTLocalAdam,
    Gaussian,
)

input_chart = Chart.grid((28, 28), trainable=False)
output_chart = Chart.linspace(10, trainable=False)

model = CSTLinear(
    input_chart,
    output_chart,
    atoms=64,
    kernel=AmplitudeBandwidthSeparable(
        sigma_min=0.10,
        sigma_max=1.0,
        tau=0.005,
        temperature=0.25,
    ),
    atom_init="uniform",
    backend="auto",
)

optimizer = CSTLocalAdam(model)

for images, labels in loader:
    images = images.flatten(1)
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
```

This API makes seven ownership decisions explicit:

1. a `Chart` owns fixed-cardinality observation coordinates;
2. `Atoms` owns one fixed-shape opaque parameter table;
3. a `Kernel` interprets complete atom rows but owns no trainable state;
4. a `CSTLinear` composes charts, atoms, and one kernel;
5. a CST optimizer partitions and exclusively owns every trainable parameter;
6. the CST engine attaches its concrete transient `AtomGrad` to each atom table;
7. its implicit CST engine owns compressed moment and accepted-frame state;

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
p  [K, P]
```

`Atoms` knows only the fixed shape and atom-row correspondence. It does not know
whether entries of `p` represent an amplitude, source, target, bandwidth,
orientation, scale, or another kernel-specific quantity. Only the selected
`Kernel` interprets `p` and turns each row into one complete operator atom.

The table is one parameter with fixed shape and atom ordering. Every trainable
atom property belongs in that atom's row of `p`; a kernel object does not own
shared trainable parameters.

`Atoms.grad` is a transient optimizer extension point, analogous in purpose to
`Parameter.grad` but not restricted to one tensor. The optimizer attaches a
concrete `AtomGrad` program before the base-point backward pass. `Atoms` neither
defines nor interprets its contents, and the object is not serialized in the
model `state_dict()`.

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
model.atoms.p
model.kernel
model.dense_weight()       # diagnostic materialization only
model.extra_repr()
```

`forward(x)` accepts `[..., in_features]` and returns
`[..., out_features]`, matching `torch.nn.Linear` except for the missing bias
in the first implementation.

### Represented operator

Let the fixed charts be collected as $\mathcal C$, and let atom $a$ have one
opaque parameter row $p_a$. The canonical represented operator is

$$
\boxed{
W=\sum_{a=1}^{K}\mathcal K(p_a;\mathcal C)
}.
$$

`CSTLinear` does not inspect $p_a$. It asks its single `Kernel` to evaluate
$\mathcal K(p_a;\mathcal C)$ as the complete contribution of atom $a$, then
sums those contributions. In particular, an amplitude is an ordinary opaque
coordinate interpreted inside the kernel rather than a distinguished factor
owned by `Atoms` or `CSTLinear`.

A separable kernel may additionally expose factors satisfying

$$
\mathcal K(p_a;\mathcal C)
=f_{\mathrm{out}}(p_a)f_{\mathrm{in}}(p_a)^\top
$$

with every coordinate effect, including amplitude, already embedded in those
factors. If their columns are collected into $\Phi_{\mathrm{in}}(p)$ and
$\Phi_{\mathrm{out}}(p)$, the same canonical sum can be evaluated as

$$
W=\Phi_{\mathrm{out}}\Phi_{\mathrm{in}}^\top,
\qquad
Y=(X\Phi_{\mathrm{in}})\Phi_{\mathrm{out}}^\top.
$$

This factorization is an optional execution capability, not the semantic
definition of a CST operator.

No neuron gate is present. Capacity is selected by `atoms` at construction,
not by a discrete operation during training.

## Kernels

A `Kernel` is the stateless interpretation of one complete atom row. Its
contract provides:

```python
kernel.parameter_dim(input_chart, output_chart)
kernel.initialize(input_chart, output_chart, atoms, mode=...)
kernel.materialize_atoms(input_chart, output_chart, p)
kernel.factors(input_chart, output_chart, p)  # optional capability
```

`materialize_atoms` returns the complete represented contribution of every
atom; `CSTLinear` adds no separate amplitude. Production backends may use a
more structured kernel capability instead of materializing those operators.

The base separable kernel composes scalar profiles without adding an amplitude:

```python
Separable(
    input_profile=Gaussian(sigma=0.25),
    output_profile=Gaussian(sigma=0.10),
)
```

An independent signed amplitude is an explicit kernel composition:

```python
Amplitude(
    Separable(
        input_profile=Gaussian(sigma=0.25),
        output_profile=Gaussian(sigma=0.10),
    )
)
```

The Gaussian profile uses the L2-normalized gauge

$$
\kappa_\sigma(u,v)
=\frac{
\exp\left(-\frac{\lVert u-v\rVert_2^2}{2\sigma^2}\right)
}{
\left\lVert\exp\left(-\frac{\lVert \cdot-v\rVert_2^2}{2\sigma^2}\right)\right\rVert_2
}.
$$

For `Separable`, $p_a=(s_a,t_a)$. The `Amplitude` wrapper changes this to
$p_a=(w_a,s_a,t_a)$ and evaluates $w_a\mathcal K(s_a,t_a)$. The wrapped kernel
still owns the meaning of $(s_a,t_a)$; `Atoms` sees only one opaque row.

The amplitude-dependent shared-bandwidth variant is:

```python
AmplitudeBandwidthSeparable(
    sigma_min=0.10,  # strong-atom width
    sigma_max=1.0,   # weak-atom width
    tau=0.005,
    temperature=0.25,
)
```

It keeps the same opaque layout $(w_a,s_a,t_a)$, but smoothly interpolates one
shared input/output precision between `sigma_max` and `sigma_min` using the even
gate

$$
g(w)=\operatorname{sigmoid}\left(
\frac{\log(w^2+\epsilon_g^2)-2\log\tau}{T}
\right).
$$

Thus weak atoms are broad on both sides and sufficiently strong atoms approach
the same narrow Gaussian on both sides. Both bounds are finite. Fixed positive
`gate_eps` makes the map twice differentiable at zero amplitude, which is
required by the CST Hessian contractions. These bandwidth controls are fixed
kernel configuration; a future learned threshold or bandwidth must live in
each opaque atom row.

Both fixed- and variable-width Gaussian columns are L2-normalized. Therefore
an `Amplitude`-bearing separable atom has Frobenius norm exactly `abs(w)`, so
bandwidth changes do not silently rescale the meaning of its amplitude.
Amplitude-bearing kernels initialize
$w_a\sim\mathcal N(0,(0.1/\sqrt K)^2)$, matching the successful fixed-$K$
MNIST protocol.

Kernel values must remain differentiable with respect to `p`; the internal
derivative layer supplies the second-order displacement contractions. A later
optimized kernel capability may provide those contractions directly without
changing this canonical atom contract.

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
| `"materialized"` | evaluates $\sum_a \mathcal K(p_a)$ and calls a dense GEMM | oracle checks and kernels without a factorization |
| `"auto"` | selects between the two without changing the represented map | default |

Backends receive ordinary fixed-shape tensors. They know nothing about stores,
entity IDs, policies, or optimizer state.

`dense_weight()` remains available for diagnostics and small correctness
oracles. A dense represented matrix is never persistent model or optimizer
state.

## Atom-structured derivatives

For opaque atom-row width `P`, the internal derivative API consumes parameter
points and directions with shape `[K, P]`. It evaluates displacement, JVP, VJP,
and Hessian contractions
without interpreting columns of $p$. With frozen charts, the contracted
representation Hessian stores only its nonzero atom blocks:

```text
pullback             [K, P]
contracted Hessian   [K, P, P]
```

The dense correctness oracle may construct the mathematical full Hessian with
shape `[K, P, K, P]` and verifies that distinct-atom blocks are zero.
This representation sparsity does not remove cross-atom terms created later by
the full quartic objective.

## Atom gradients and autograd

The optimizer decides which represented derivative information it needs by
attaching a concrete `AtomGrad` to `model.atoms`. Its explicit lifecycle is

```text
attach -> begin -> forward/backward (possibly repeated) -> complete -> read
```

The object is mutable during backward: the selected Linear or future Conv
integration sets and accumulates its fields directly. PyTorch's backward return
values remain the ordinary gradients for the corresponding forward inputs;
optimizer-specific values travel through the attached `AtomGrad` object.

Operation contracts are separate. `LinearAtomGrad` describes what a
`CSTLinear` backward may invoke, while a future `ConvAtomGrad` can retain the
spatial structure needed by convolution without forcing both through one
matrix-specific interface. Concrete implementations belong to the optimizer
layer.

Each concrete program selects one execution route:

| mode | behavior |
| --- | --- |
| `"auto"` | use custom autograd when implemented, otherwise use hooks |
| `"custom"` | require custom autograd and fail if it is unavailable |
| `"hooks"` | force the reference hook route |

The routes never run simultaneously. The custom backward computes and returns
the ordinary input and atom-parameter gradients while setting optimizer fields
inside the same backward processing. Hooks provide a correctness and debugging
route with the same resulting contract.

The first implicit Linear program produces the raw observations

```text
J.T g              [K, P]
g contracted H     [K, P, P]
r                  [out_features]
c                  [in_features]
```

where $r$ and $c$ are row and column means of the squared, aggregate represented
gradient. It does not create the full represented $g_W$ tensor. For a Linear
call it retains the implicit factors $(x,g_y)$; at `complete()` it reconstructs
only chunks of $g_W=g_y^\top x$. Repeated calls are summed before squaring, so
their cross terms are exact. Persistent EMA updates remain an optimizer
responsibility and are not performed by autograd.

## Replaceable moment components

The implicit optimizer assembles its moment behavior from separate first- and
second-moment components. A component owns four decisions:

```text
observation request -> initial state -> step-local expansion -> committed compression
```

The installed components union their `AtomGradRequest` values before backward.
Consequently, replacing a moment also changes which autograd observations are
computed: the accepted-frame first moment requests `J.T g` and `g contracted H`,
while the separable second moment requests only row and column square
statistics.

`MomentSystem.expand(...)` combines the previous persistent state and the
current immutable observation into raw pending state and bias-corrected solver
views. Expansion never mutates persistent state. After the solver returns,
`MomentSystem.compress(...)` creates both new component states before returning
one new aggregate state.

The selected first implementation composes:

```python
MomentSystem(
    first=AcceptedFrameFirstMoment(beta=beta1),
    second=SeparableDiagonalSecondMoment(beta=beta2, eps=eps),
)
```

The first component transports the old visible representative through its old
accepted frame and recompresses the raw EMA only at the actual accepted
displacement. The second component provisionally updates raw row/column EMAs
and exposes a bias-corrected diagonal metric; its accepted compression simply
commits those compact statistics.

Moment components use an optimizer-independent `FrameGeometry` contract for
cross-frame pullbacks and Gram solves. The initial `AutogradFrameGeometry` is a
correctness implementation. Future Linear, Conv, or kernel-specific
implementations may fuse those contractions without changing moment or solver
interfaces.

### Composable normalized optimizers

The new normalized optimizer family works directly with the candidate-dependent
atom-coordinate update

The design details, tensor shapes, moment state, and solver geometry are in
[the normalized optimizer design document](docs/normalized-optimizer.ja.md).

$$
d=-\eta\frac{N(d)}{D(d)}.
$$

The numerator and denominator are separate components, and their persistent
states remain in fixed atom coordinates. Expansion prepares the step-local
values; compression only commits the pending states and does not recenter or
transform them using the accepted displacement.

The four public wrappers select only the two components:

| optimizer | numerator `N(d)` | denominator `D(d)` |
| --- | --- | --- |
| `CSTSGD` | current `g + H d` | `1` |
| `CSTMomentum` | EMA of `g` and `H` | `1` |
| `CSTRMSProp` | current `g + H d` | component-wise EMA of `(g + H d)^2` |
| `CSTNormalizedAdam` | EMA of `g` and `H` | component-wise EMA of `(g + H d)^2` |

`CSTImplicitAdam` is an alias for `CSTNormalizedAdam`. The historical
`CSTAdam` remains available with its existing tangent-moment behavior while the
new family is being evaluated.

Use the same whole-model interface as the other model-level optimizers:

```python
from torchcst import CSTNormalizedAdam

optimizer = CSTNormalizedAdam(
    model,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    trust_radius=0.25,
)

for inputs, targets in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    optimizer.step()
```

The common configuration is `NormalizedOptimizerConfig`. `CSTSGD`,
`CSTMomentum`, `CSTRMSProp`, and `CSTNormalizedAdam` accept its fields directly
or through `cst=NormalizedOptimizerConfig(...)`.

The update solver is injected independently of the optimizer wrapper. The
default is `NormalizedFixedPointSolver`; another solver only needs to implement
the `NormalizedSolver` contract and return a `NormalizedSolveResult`:

```python
from torchcst import CSTNormalizedAdam
from torchcst.optim import NormalizedFixedPointSolver

solver = NormalizedFixedPointSolver(max_iter=64, tolerance=1e-7, damping=0.8)
optimizer = CSTNormalizedAdam(model, solver=solver)
```

The atom-gradient program used by this family is `LinearJGHAtomGrad`, which
collects `jg:[K, P]` and the local contracted Hessian `gh:[K, P, P]`. These
observations are reused by both the numerator and denominator components.

### Exact structured quartic evaluation

`SecondOrderAdamConfig(quartic_evaluation="auto")` selects an exact quadratic-feature
Gram evaluation when the kernel supplies an exact factorization and the visible
metric is separable. `"visible"` forces the reference evaluation; `"gram"` forces
the structured path and raises an error for unsupported geometry, metric, or
precision. The evaluation backends work with all quartic solvers. Newton and
subspace model construction use the exact geometric derivatives independently
of the selected value/gradient backend.

The structured path writes the second-order displacement as `R(d) = T z(d)`,
where `z` contains each atom's linear and upper-triangular quadratic monomials.
It constructs `G = T.T D T` from input/output factor derivatives, without
materializing visible Jacobian or Hessian tensors. The solver then evaluates
`z.T G z` and its analytic gradient. All cross-atom terms, second derivatives
(including amplitude-dependent bandwidth), and the metric's additive epsilon
are retained. This is an algebraic rewrite, not a truncation or low-rank
approximation; floating-point summation order can still change solver trajectories.

Automatic selection currently requires CPU float32/float64, at most 1,024 monomial
features, at least four visible entries per feature, and at most 16 million
factor derivative elements. These are conservative size heuristics, not a
hardware-specific speed guarantee. CUDA currently keeps the visible backend
under `"auto"`; explicit `"gram"` supports CUDA and bypasses the automatic device
and size restrictions.
Each problem builds a new step-local Gram matrix; it is never reused after a
parameter or moment update. The solver and its stopping tolerances are unchanged.

The setup-inclusive CPU benchmark and its limitations are described in
[the structured quartic report](https://github.com/takumiecd/cst-experiments/blob/main/docs/torchcst-experiments/quartic-gram.md).


## `CSTParameterAdam`: parameter-sized state for short training runs

```python
from torchcst import CSTParameterAdam, ParameterAdamConfig

# Construct every CSTLinear with backend="factored".
optimizer = CSTParameterAdam(model, cst=ParameterAdamConfig(decay_steps=128))
```

This opt-in optimizer applies ordinary Adam directly to the atom coordinates,
with a cosine learning rate from 0.03 to 0.003 over 128 updates and
`betas=(0.5, 0.99)`. Its only tensor state is two parameter-shaped moment
buffers and a step scalar: **2,052 bytes for 64 four-coordinate FP32 atoms**.
No visible weights/moments, Jacobians, Hessians, or Gram matrices are built by
the optimizer. Forward/backward still use the existing factor tables and batch
activations. Frozen charts and the explicit factored backend are required.

Pass `dense=AdamWConfig(...)` for ordinary trainable parameters in mixed models;
that group uses its own unscheduled AdamW settings. The internal schedule and
moments resume from `state_dict()`. Set `decay_steps=None` to use an external
learning-rate schedule. See [the equations and memory contract](docs/parameter-adam.ja.md).

In the frozen A100 MNIST protocol (64 atoms, batch 128, 128 updates), six
seeds reached **80.04% mean accuracy on the existing 2,000-example test prefix**
and **83.96% on all 10,000 test examples**. Settings were selected using a
separate validation subset; the three previously unused seeds averaged
79.48% / 83.42%, respectively. This is a tuned short-run configuration, not
a guarantee that every seed exceeds 80%.


## `CSTLocalAdam`: default atom-local optimizer

The [mathematical specification (Japanese)](docs/local-adam-math.ja.md) records
the transport equations, coordinate systems, state timing, and approximations.

`CSTLocalAdam` accepts the same whole-model interface as `CSTAdam` and
`CSTSecondOrderAdam`. It discovers each `CSTLinear` site and updates ordinary
trainable parameters, including `nn.Linear`, through the optional dense AdamW
block in the same coordinated step. Pass the model, not `model.parameters()`.

```python
from torchcst import CSTLocalAdam, LocalAdamConfig, AdamWConfig

optimizer = CSTLocalAdam(
    model,
    cst=LocalAdamConfig(
        lr=0.05,
        betas=(0.9, 0.99),
        first_moment_damping=0.01,
        update_damping=0.01,
    ),
    dense=AdamWConfig(lr=1e-3),
)

optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model(inputs), targets)
loss.backward()
optimizer.step()
```

For a model with only CST-owned trainable parameters, omit `dense`. A model
without any `CSTLinear` should use an ordinary PyTorch optimizer. Existing
frozen-chart, fixed-layout, and strict parameter-ownership constraints apply.
The selected defaults are `lr=0.05`, `betas=(0.9, 0.99)`,
`whitening="cholesky"`, `whitening_damping=1e-4`, and first/update damping `0.01`.
They match the 77.00% median / 76.25% mean at 128 updates in the three-seed
MNIST experiment, not a guarantee for other models or data. `CSTAdam` retains
its existing algorithm as an explicit alternative.

For each atom, let `R = J_t.T @ J_t`, `S = J_t.T @ J_previous` and let `g` be
its accumulated parameter gradient. First-moment transport and recompression are

```text
b       = beta1 * S @ alpha_previous + (1 - beta1) * g
alpha   = solve(R + first_moment_damping * I, b)
b_hat   = b / (1 - beta1**step)
```

By default, factor `R + whitening_damping * I = L @ L.T` and construct
`B = L^{-T}` using a triangular solve. With `T = B.T @ S @ B_previous` and
`h = B.T @ g`, the second moment is

```text
C       = beta2 * T @ C_previous @ T.T + (1 - beta2) * outer(h, h)
C_hat   = C / (1 - beta2**step)
A       = B.T @ R
M       = A.T @ (sqrt_psd(C_hat) + eps * I) @ A
delta   = solve(M + update_damping * I, -lr * b_hat)
```

All matrices above are per-atom blocks, with storage proportional to `K*q*q`
for `K` atoms and `q` coordinates per atom. First-moment recompression and the
final update use one batched `solve_ex` inverse-vector product without forming
an inverse or expanding the blocks to `K*K*q*q`. There is no iterative linear
solver, global Gram matrix, or trust-radius clipping. All damping values must be
positive. Default whitening still uses batched Cholesky because it needs the
factor `B = L^{-T}`; the C square root still requires a small eigendecomposition.
Cross-atom history and covariance are
omitted: information lost to one atom is not handed to another atom. `C` uses
the parameter dtype; basis and small solve/metric calculations use FP64.
Parameter gradients and `C` are formed from the accumulated batch gradient;
this differs from pulling back a dense elementwise squared-gradient EMA.

`whitening_damping=1e-4` sets the default whitening regularization independently
of first-moment recompression. Explicit `None` reuses `first_moment_damping`,
matching the initial Cholesky implementation. Positive damping makes `J @ B`
contractive rather than orthonormal. `tangent_rtol` does not select directions
in Cholesky mode.

Set `whitening="eigen"` for the original active-eigenspace method, which excludes
small directions using `tangent_rtol`. To resume a checkpoint from the original
public defaults, explicitly set `lr=1e-3`, `betas=(0.9, 0.999)`, and
`whitening="eigen"`. For an old Cholesky checkpoint with shared damping, also
set `whitening_damping=None` and restore its original hyperparameters.
Checkpoint loading rejects mismatched moment settings rather than reinterpreting
stored histories. The implementation does not change existing `CSTAdam` or
`CSTSecondOrderAdam` defaults.

`state_dict()` / `load_state_dict()` support mixed-model checkpoint continuation,
including the FP64 transport basis. Save/restore the model weights as well.
`device_execution=True` defers validity checks and requires factored geometry;
call `optimizer.check_errors()` at a chosen host boundary. It does not promise
that every PyTorch eigendecomposition or backend operation avoids synchronization.
The factorized backend avoids a dense Jacobian; the reference backend is an oracle
and may materialize one. There are no trust-region options on `LocalAdamConfig`.

This is a public research optimizer. The current evidence is a small MNIST sweep,
not broad convergence or speed superiority; see [the no-trust experiment](https://github.com/takumiecd/cst-experiments/blob/main/docs/torchcst-experiments/no-trust.ja.md).


## `CSTLocalVisibleAdam`: projected dense-diagonal alternative

`CSTLocalVisibleAdam` shares `CSTLocalAdam`'s atom-local first-moment transport
and unconstrained direct update, but gives the second state a different meaning.
It requests the exact local blocks

```text
E[a] = J[a].T @ Diag(visible_gradient**2) @ J[a]
```

from `AtomGrad`, transports the previous visible operator representative as
`S @ Gamma @ S.T`, and stores `Gamma` with shape `[K, q, q]`. It omits
cross-atom second-moment blocks and never stores a whitening basis or dense
visible second moment. The update metric approximates
`J.T @ Diag(sqrt(v)) @ J` by the matrix geometric mean of `J.T @ J` and the
transported raw second-moment block. See the
[mathematical specification](docs/local-visible-adam-math.ja.md) for the exact
state equations and approximation boundary.

```python
from torchcst import (
    AdamWConfig,
    CSTLocalVisibleAdam,
    LocalVisibleAdamConfig,
)

optimizer = CSTLocalVisibleAdam(
    model,
    cst=LocalVisibleAdamConfig(
        lr=0.05,
        first_moment_damping=0.01,
        second_moment_damping=1e-4,
        update_damping=0.01,
    ),
    dense=AdamWConfig(),
)
```

This optimizer has a separate checkpoint contract from `CSTLocalAdam`; their
states cannot be loaded into one another.


## `CSTDenseVisibleAdam`: dense-moment accuracy alternative

`CSTDenseVisibleAdam` deliberately keeps Adam's first and second EMA in the
represented weight space. For each `CSTLinear` site it stores two tensors with
shape `[out_features, in_features]`:

```text
m = beta1 * m + (1 - beta1) * visible_gradient
v = beta2 * v + (1 - beta2) * visible_gradient**2
```

The atom update still remains local. It streams Jacobian rows and atom tiles to
form the exact blocks

```text
b[a] = J[a].T @ corrected_m
M[a] = J[a].T @ Diag(sqrt(corrected_v) + eps) @ J[a]
(M[a] + update_damping * I) @ delta[a] = -lr * b[a]
```

It never materializes a dense Jacobian, a `[K, K, q, q]` matrix, or an inverse.
Unlike `CSTLocalVisibleAdam`, it performs no recursive projection transport and
uses no geometric-mean reconstruction. The cost is persistent state of
`2 * out_features * in_features` scalars, so this is an accuracy-oriented
reference rather than the memory-saving CST default.

```python
from torchcst import CSTDenseVisibleAdam, DenseVisibleAdamConfig

optimizer = CSTDenseVisibleAdam(
    model,
    cst=DenseVisibleAdamConfig(lr=0.05, update_damping=0.01),
)
```

See the [mathematical specification](docs/dense-visible-adam-math.ja.md).


## `CSTAdam`: full-tangent first-order alternative

`CSTAdam` uses only first derivatives of the represented weight map. It stores
first-moment coefficients in the pre-update tangent, transports that history
into the next current tangent, and solves a convex quadratic inside the
parameter-space Euclidean trust ball. It preserves cross-atom terms with the
default separable metric. It never requests Hessian observations.

```python
from torchcst import CSTAdam, CSTSecondOrderAdam, FirstOrderAdamConfig, FullQuartic

optimizer = CSTAdam(model, lr=0.05, trust_radius=0.25)
# Equivalent configuration object; do not mix cst= with direct options.
optimizer = CSTAdam(model, cst=FirstOrderAdamConfig(lr=0.05))

# Experimental visible RMS, transported separately for each atom.
optimizer = CSTAdam(model, lr=0.05, second_moment="atom_block")
optimizer = CSTAdam(model, lr=0.05, second_moment="atom_diag")

# Select the optional second-order algorithm explicitly.
optimizer = CSTSecondOrderAdam(model, lr=0.05, quartic=FullQuartic())
```

These examples are alternatives; one model's atoms have exactly one optimizer
owner. `dense=AdamWConfig(...)` includes ordinary dense parameters in the same
coordinated step. The training/ownership contract below applies to both classes.

For the default metric, the first-order problem is

$$
\min_{\|d\|\le r}\quad
(J^\top\widehat m)^\top d+\frac{1}{2\eta}d^\top J^\top D Jd.
$$

`separable` stores row/column EMAs of squared represented gradients, then
reconstructs the diagonal RMS action. `atom_block` and `atom_diag` are different,
experimental metrics: they transport squared-gradient operators in each atom's
tangent space before taking matrix square roots. They omit cross-atom metric
couplings; `atom_diag` additionally drops within-atom off-diagonals each step.
Neither is claimed to reproduce diagonal Adam or preserve learning accuracy.

`FirstOrderAdamConfig` contains only first-order controls: learning rate, betas,
epsilon, trust radius, second-moment backend, first-moment damping, observation
mode/chunk size, factored geometry, tangent rank cutoff, prepared backend/tile,
and recompression/update solver budgets and tolerances. There is no
`approximation_order` or `first_moment_frame` switch and no quartic setting.
The old `CSTOptimizer` and `ImplicitAdamConfig` names have been replaced.

This first-order implementation supports CPU/CUDA float32/64, with optional
deferred GPU execution. It does not support MPS. The spectral update solver retains a full
parameter-space Gram for the separable metric. First-moment compression and
transport use the prepared tangent operator described below; compression is
matrix-free when `recompression="pcg"` is explicitly selected.
Atom block/diagonal metric state scales linearly with atom count at fixed
parameters per atom. Selecting PCG for both recompression and updates avoids
full parameter-space Grams with the separable metric.

### Prepared tangent actions and recompression

```python
ops = layer.cst_derivatives().tangent_ops(backend="auto", atom_tile=32)
current = ops.prepare(layer.atoms.p)  # ordinary Tensor / nn.Parameter
previous = ops.prepare(saved_parameters)
pulled = current.cross_gram_matvec(previous, alpha)  # J_current.T @ J_old @ alpha
gram_x = current.gram_matvec(x)
weighted_x = current.weighted_gram_matvec(metric, x)
blocks = current.gram_blocks()  # exact same-atom blocks, for preconditioning

optimizer = CSTAdam(
    model,
    recompression="pcg",
    first_moment_damping=1e-3,  # explicit algorithm choice, not a tuned default
    recompression_max_iter=64,
    recompression_rtol=1e-5,
    tangent_atom_tile=32,
)
# After optimizer.step():
diagnostics = optimizer.last_step.compression_results
```

`auto` selects kernel-owned analytic first factor derivatives when available
(Gaussian Separable, Amplitude, AmplitudeBandwidthSeparable), then exact
`factor_autograd`, then `reference`. The selected name is `ops.backend`.
`specialized` fails explicitly if unsupported. `factored_geometry=False` forces
the reference path. Prepared objects own detached parameter/factor snapshots;
their `jvp(x)` and `vjp(force)` are also available as dense-visible oracles.

Fast Gram actions preserve cross-atom coupling, with bounded atom-pair tiles
and no full Gram or visible weight matrix. `weighted_gram_matvec` accepts the
separable row/column metric and includes epsilon exactly. Metric construction
does not materialize its visible diagonal until a dense operation requests it.
Cached factors cost O(K q (inputs + outputs)); pairwise arithmetic still scales
quadratically in K. The default uses eager PyTorch contractions; the optional
CUDA path below fuses these actions.

PCG solves **(J.T J + damping I) alpha = b**, using exact same-atom blocks only
as a preconditioner. It does not introduce D into first-moment compression or
discard cross-atom terms. Positive damping is required; zero-damping minimum-norm
compression keeps the existing `direct` default. PCG uses native parameter dtype
and FP64 scalar reductions, checks the true residual of the returned alpha, and
raises before any parameter/moment commit on failure. CUDA convergence checks
in eager execution synchronize with the host. Diagnostics report iterations and residual;
direct compression reports `None`.

### Optional GPU execution for first-order Adam

```python
optimizer = CSTAdam(
    model,
    device_execution=True,
    recompression="pcg",
    first_moment_damping=1e-3,
    recompression_max_iter=128,
)
# Run ordinary zero_grad / backward / step calls.
optimizer.check_errors()  # explicit synchronization, e.g. at a logging boundary
```

This opt-in path requires factorized tangents and `second_moment="separable"`.
CUDA float32/float64 execution uses Triton factor contractions, a captured PCG
loop, and the selected update solver. Same-atom blocks precondition the
full cross-atom system; the approximation and damping are unchanged. Convergence
and true-residual checks stay on the GPU. Inactive iterations skip Gram arithmetic,
but graph replay still launches the fixed iteration budget. CPU execution provides
a tensor-controlled correctness fallback, not a performance optimization.

Install `pip install -e '.[cuda]'` on Linux with a matching CUDA-enabled PyTorch,
CUDA development headers/libraries, and a C++ compiler. The native cuSOLVER wrapper
is built with Ninja on first use. Triton/Inductor compilation, graph capture, and
native initialization are cold-path costs. Graph buffers persist per shape,
dtype, device and solver settings; changing shapes/settings retains cached graphs.
There is no silent fallback if these CUDA dependencies are unavailable.

Device-mode compression diagnostics contain scalar tensors. Calling `.item()`,
printing them, or `check_errors()` intentionally synchronizes. Numerical failures
latch a device error and suppress subsequent parameter/moment tensor commits;
inspect `check_errors()` and restore a valid checkpoint into a fresh optimizer
before resuming. Python step metadata still advances while the latch is set.
The default spectral update solver still allocates a full weighted Gram and uses
cuSOLVER, whose internals can synchronize. Select the PCG update below to avoid
that full Gram and eigensolve.

For standalone prepared actions, pass `execution="triton"` to `tangent_ops`.
Standalone `prepare` validates immediately, and standalone PCG reads its final
success flag once. The optimizer defers these checks to its device error latch.

See the companion repository's
[CUDA timing, transfer-count, and persistent-memory measurements](https://github.com/takumiecd/cst-experiments/blob/main/docs/torchcst-experiments/tangent-device.ja.md).

### Matrix-free trust-region updates

```python
optimizer = CSTAdam(
    model,
    device_execution=True,
    recompression="pcg",
    first_moment_damping=1e-3,
    update_solver="pcg",      # default: "spectral", retained as a reference
    update_max_iter=512,      # CG iterations per search round
    update_shift_steps=32,    # budget for half-interval shift search
    update_rtol=1e-5,         # final KKT residual / complementarity tolerance
)
```

The objective and Euclidean trust ball are unchanged. For `H = J.T D J / lr`,
the solver applies H from prepared factors and solves `(H + shift I) d = -b`.
The shift enforces the existing radius; it is separate from first-moment damping.
Cross-atom terms are preserved. Only the positive-shift PCG preconditioner uses
same-atom blocks of H; the objective itself is never diagonalized or approximated
by atom blocks. This mode requires factorized tangents and separable moments.

Zero-shift CG starts from zero without preconditioning to preserve the Euclidean
minimum-norm solution for consistent singular systems in exact arithmetic.
If it crosses the ball or does not converge, a bracketed positive-shift search
uses warm-started atom-block PCG. For positive shifts, the true residual bounds
the solution error by `norm(residual) / shift`; the bracket advances only when
that interval establishes the exact solution is inside or outside the radius.
Ambiguous rounds continue at the same shift, preserving unfinished CG residuals and
directions. If an inner solve has converged but its error bound is still too wide,
the inner tolerance is tightened and CG restarts from that solution. Candidates
are also checked after conversion to parameter dtype; rounding failures trigger
further refinement before a candidate is accepted. Factor contractions
and solver arithmetic use FP64, including for FP32 parameters.

Before commit, the returned parameter-dtype displacement is checked using a fresh
H action: `norm((H + shift I)d + b) / norm(b) <= update_rtol`,
`shift * abs(radius - norm(d)) / norm(b) <= update_rtol`, and the radius constraint.
Zero b returns zero. Numerical failure or insufficient budgets raise in eager
mode and latch/suppress updates in deferred mode; there is no dense fallback.
Near-null directions can produce different displacements within this tolerance,
even with very similar objective values. The tolerance does not guarantee a
small displacement error for ill-conditioned H.

`optimizer.last_step.site_results` exposes `shift`, `relative_residual`,
`relative_complementarity`, `shift_iterations`, and total active CG `iterations`.
`evaluations` counts scheduled H calls (including device-masked calls). These
diagnostics are scalar tensors. Inspect them at explicit logging boundaries.

With both solvers set to PCG, this path stores factors, vectors and bounded graph
workspaces, without a full Gram, dense Jacobian, eigenbasis or Krylov basis.
Workspace scales linearly with atom count for fixed chart sizes, coordinates per
atom and iteration budgets. CUDA applies JVP, D and VJP in tiles of at most 16
visible output rows, with O(K*q*inputs*outputs) arithmetic and bounded visible
scratch, instead of explicitly contracting every atom pair. The CPU oracle uses
the existing atom-pair contraction. Two reusable
CG graphs handle zero/positive shifts, with fresh factors, metric and RHS supplied
on replay. GPU control stays on device, although the host schedules a fixed outer
budget and inactive graph nodes still launch. This can be slower than spectral
updates on small problems; it is an explicit memory-oriented option.
The PCG update does not build or call the native cuSOLVER extension.

See the companion repository's
[matrix-free solver timings, memory costs, and dense accuracy audit](https://github.com/takumiecd/cst-experiments/blob/main/docs/torchcst-experiments/trust-pcg.ja.md).

### Optional diagonal and atom-block updates

`update_approximation="diagonal"` or `"atom_block"` explicitly approximates only
the update quadratic. The default `"full"` retains all cross-atom terms.

```python
optimizer = CSTAdam(
    model,
    update_approximation="diagonal",  # or "atom_block"
    update_solver="spectral",
    second_moment="separable",
    recompression="pcg",
    first_moment_damping=0.01,
    device_execution=True,
)
```

For `H = J.T D J / lr`, the diagonal option uses `diag(diag(H))`; atom-block uses
`block_diag(H_aa)`. D's separable EMA, the linear term, cross-frame first-moment
transport, and full cross-atom recompression are unchanged. These options differ
from `second_moment="atom_diag"` / `"atom_block"`, which change the second-moment
model itself. They require separable moments, factorized tangents and the
`"spectral"` solver; combining them with `update_solver="pcg"` is rejected.

Both solve one global Euclidean trust ball. Diagonal updates are
`d_i = -b_i / (H_ii + shift)`; atom-block updates use small per-atom eigensystems.
All coordinates share the same shift, found by a scalar secular search. No CG
iterations, full parameter-space Gram, or global eigenbasis is needed for these
updates. Recompression may still iterate. CUDA atom-block eigensystems use
fixed-sweep batched Jacobi with residual/orthogonality checks, avoiding host
eigensolver status reads on warmed calls. Odd coordinate widths are padded
internally; the kernel still owns the atom layout.

`update_rtol` checks stationarity and complementarity of the **approximated**
quadratic after conversion to parameter dtype. It does not certify a solution
of the full quadratic. Non-finite data or failed certificates raise in eager
mode and latch/suppress updates with deferred execution. `site_results` exposes
`shift`, `relative_residual`, and `relative_complementarity`; its objective is
the approximate objective. The reported 80 `iterations` are the fixed scalar
search budget, not CG iterations. `update_max_iter` / `update_shift_steps` are
PCG-only controls and do not affect these options.

These approximations trade cross-atom coupling for lower update cost; their
learning accuracy is task-dependent. See the [paired accuracy and timing
experiment](https://github.com/takumiecd/cst-experiments/blob/main/docs/torchcst-experiments/local-tangent.ja.md).

Fixed kernel/profile configuration and chart buffers must remain unchanged for
the lifetime of an optimizer. Checkpoints now include their tangent descriptors;
first-order checkpoints without descriptors are rejected. Load model state before
constructing the optimizer. Custom kernels/profiles declare non-buffer settings
through `tangent_config()` and bump `tangent_layout_version` when layout semantics
change. Writes through tensor `.data` bypass version checks and are unsupported.
Prepared caches are step-local and never serialized. CUDA graph caches are also
runtime-only. The default spectral update constructs its full weighted Gram;
the PCG update uses the matrix-free path described above.

Checkpoints use schema version 2 and record the algorithm and moment contract.
Loading a different algorithm/second-moment definition or incompatible parameter ownership is an
error; there is no implicit conversion of old checkpoints.

See [the mathematical design and complete memory accounting](docs/first-order-rebuild.ja.md)
and [the paired pilot](https://github.com/takumiecd/cst-experiments/blob/main/docs/torchcst-experiments/tangent-rebuild.md).

## `CSTSecondOrderAdam`

`CSTSecondOrderAdam` is the optional second-order model-level optimizer. It accepts the complete
model rather than an arbitrary iterable of tensors, discovers every CST site,
and partitions every remaining trainable parameter into its dense block.

```python
CSTSecondOrderAdam(
    model,
    *,
    cst=SecondOrderAdamConfig(...),
    dense=AdamWConfig(...) | None,
    strict=True,
)
```

Users do not construct a separate dense optimizer. Internally, the coordinator
uses a strict CST-only implicit engine and a functional dense AdamW engine.
The latter is a proposal/state implementation owned by `CSTSecondOrderAdam`, not an
independently stepping `torch.optim.AdamW` instance.

For mixed models:

```python
optimizer = CSTSecondOrderAdam(
    model,
    cst=SecondOrderAdamConfig(
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

### Step contract

`CSTSecondOrderAdam` follows the ordinary PyTorch training order:

1. `optimizer.zero_grad(set_to_none=True)` clears parameter gradients and begins
   the CST observation scope;
2. the caller evaluates the minibatch loss and calls `backward()`;
3. `optimizer.step()` consumes those gradients and observations exactly once.

The optimizer forms the CST quartic proposal and dense AdamW proposal from the
same backward pass and commits both together:

$$
(\theta_{\mathrm{cst}},\theta_{\mathrm{dense}})
\longmapsto
(\theta_{\mathrm{cst}}+d_{\mathrm{cst}},
 \theta_{\mathrm{dense}}+d_{\mathrm{dense}}).
$$

Calling `step()` without first opening an observation scope with `zero_grad()`
is an error. The optimizer does not reevaluate the model loss, rescale the
proposal, or reject a completed update.

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

Internally, `QuarticProblem` receives only the current `MomentContext` and an
immutable `ExpandedMoments` proposal. It exposes the scalar objective and its
exact cubic gradient, dense Hessian, and restricted quartic models; it does not
read or mutate persistent optimizer state.
`FullQuartic` is the second-order default and deterministic comparison solver.
It evaluates cold zero, negative-gradient, and deterministic starts through a
smooth trust-ball parameterization. `ProjectedLBFGS` is an experimental,
strict-budget alternative that uses the exact analytic quartic gradient,
safeguarded zero and boundary-gradient candidates, and projected Armijo steps.
The optimizer applies the selected candidate without a second model-loss
evaluation.

Two additional experimental solvers use the structure of the full quartic:

```python
from torchcst import BallNewton, SubspaceQuartic

# Direct ball-constrained Newton models, including negative curvature.
cst = SecondOrderAdamConfig(
    quartic=BallNewton(starts=4, max_iter=30, max_evaluations=150),
)

# Adaptively solve exact quartics in small orthonormal subspaces.
cst = SecondOrderAdamConfig(
    quartic=SubspaceQuartic(max_dimension=8, max_models=8),
)
```

`BallNewton` forms the exact Hessian as a weighted frame Gram plus atom-local
blocks and solves regularized quadratic models on the original displacement
ball, without a saturating change of coordinates. Eigenvectors remain on the
problem device; by default the scalar secular equation uses small host double arrays.
Candidates are checked against the original quartic along feasible chords.
`starts` defaults to one; multiple starts reuse the same deterministic initial
points as `FullQuartic`. Iteration/evaluation budgets apply **per start**;
returned counts aggregate all starts (plus the initial-gradient evaluation).
First-order convergence alone is not a global-optimality certificate.

For CUDA, an experimental compiled path keeps the secular search on the GPU:

```python
quartic = BallNewton(
    starts=1, max_iter=30, max_evaluations=150,
    execution="compiled", secular_solver="device",
)
```

This compiles the exact value/gradient, Hessian, spectral search, and decision
arithmetic with `torch.compile(fullgraph=True, mode="reduce-overhead")`.
It requires materialized local quadratic derivatives and a separable diagonal
metric. The spectral scalar arithmetic remains float64 on the problem device;
no lower-precision approximation is enabled. Adaptive Python decisions and
`torch.linalg.eigh` still synchronize with the CPU. First-use compilation adds
latency; floating-point fusion can change the optimization trajectory.
See [the compiled Newton measurements](https://github.com/takumiecd/cst-experiments/blob/main/docs/torchcst-experiments/compiled-newton.md).

`DeviceBFGS(max_iter=30, max_evaluations=150)` is an experimental alternative
for `SecondOrderAdamConfig(quartic=..., device_execution=True)`. It retains the
complete quartic but changes the search to projected BFGS. Adaptive decisions,
validation flags, and moment compression remain on the device. Warmed CUDA
updates avoid host synchronization in the tested configuration; compilation,
initialization, and explicit `optimizer.check_errors()` are outside that scope.
Its iteration/convergence/boundary diagnostics are device tensors. Check errors
at an explicit reporting boundary; a failed check latches and disables updates.
See [device execution](https://github.com/takumiecd/cst-experiments/blob/main/docs/torchcst-experiments/device-execution.md) for restrictions,
accuracy comparisons, and a training loop without per-step diagnostic reads.


For factor-capable kernels, `DeviceRay` can use an experimental compact
geometry that avoids visible Jacobians/Hessians and their graph-input copies:

```python
from torchcst import DeviceRay

cst = SecondOrderAdamConfig(
    lr=0.05,
    quartic=DeviceRay(corrections=1),
    device_execution=True,
    factored_geometry=True,
)
```

This contracts exact factor-local derivatives for the ray objective, moment
transport, Gram construction, and backward observations. It retains the
original quartic and cross-atom terms; only the fixed-budget ray search is
inexact. It requires a factor-capable kernel and uses the existing separable
metric. Floating-point contraction order changes, so learning trajectories can
differ even when derivative-oracle tests pass. The default remains unchanged.
See [compact derivative measurements](https://github.com/takumiecd/cst-experiments/blob/main/docs/torchcst-experiments/factored-contractions.md).

With explicit positive `first_moment_damping`, `gram_solver="cholesky"` opts
into a faster device solve for the damped compression system. For example,
add `gram_solver="cholesky", first_moment_damping=1e-4` to the configuration
above. This changes the previous undamped, rank-truncated pseudoinverse to a
full-rank damped solve; it can change learning behavior. Cholesky status and
residual checks feed the same device failure latch. See the
[damped Gram measurements](https://github.com/takumiecd/cst-experiments/blob/main/docs/torchcst-experiments/damped-gram.md) for the precision,
accuracy, and timing tradeoff. The default remains the Jacobi pseudoinverse.

Experimental `gram_solver="pcg"` requires `factored_geometry=True` and
explicit positive `first_moment_damping`. It uses a blocked separated Gram
operator and diagonal-preconditioned conjugate gradients, controlled by
`gram_iterations` (default 64), `gram_rtol` (default 1e-5), and
`gram_block_size` (default 32). These are a fixed iteration budget, a relative
residual acceptance threshold, and a column tile size, respectively. The
returned solution is checked against the actual operator; failure freezes
updates through the device latch when `device_execution=True`. No direct-solve
fallback runs. This eager experimental path avoids the full Gram but is not
a faster replacement for Cholesky. Its FP64 factor contractions and iterative
error can change learning trajectories. See [PCG validation](https://github.com/takumiecd/cst-experiments/blob/main/docs/torchcst-experiments/pcg-gram.md).




`SubspaceQuartic` starts from gradient and atom-block-preconditioned directions,
constructs the exact restricted polynomial on the problem device, and solves
its small coefficients in CPU float64. It checks the original full-space
objective/residual, expands the basis, and restarts while retaining the best
candidate when the dimension cap is reached. Its `iterations` count reduced
models; `evaluations` include both reduced-polynomial and full-objective calls.
Both new solvers currently require float32/float64 and retain the best finite
objective reached, with zero among the candidates. Large dense Hessians can be
expensive; these implementations target the current few-hundred-variable sites.

See [the external solver comparison](https://github.com/takumiecd/cst-experiments/blob/main/docs/torchcst-experiments/quartic-solvers.md) for setup
costs, stationarity residuals, and limitations. The production default remains
`FullQuartic` pending broader learning-quality validation.

Within one optimizer step, `AutogradFrameGeometry` may materialize and cache the
atom-local Jacobian and Hessian blocks. Quartic displacement, pullback, and
accepted-frame Gram calculations share that cache. Shapes above the internal
cache budget retain the matrix-free derivative path.

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
moments inside `CSTSecondOrderAdam`.

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

Hardware-specific protocols, measurements, raw results, and figures live in the
companion `cst` experiment repository. The relevant
[experiment index](https://github.com/takumiecd/cst-experiments/blob/main/docs/INDEX.md)
supports the selected starting point. It does not yet prove:

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
   compact $\alpha$, separable $v$, full quartic, and coordinated state commit.
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

The complete target example is covered by the public optimizer integration
tests. Each API segment must remain synchronized with the implementation.

## License

See [`LICENSE`](LICENSE).

### Retained-basis update solver

`CSTAdam(..., update_solver="krylov")` restricts the full first-order trust
problem to a reorthogonalized symmetric Krylov basis. It retains all cross-atom
terms and the Euclidean trust radius. Shift searches reuse the projected matrix.
`update_basis_size=128`, `update_check_interval=32` and
`update_basis_memory_mb=64.0` control the search. The memory limit covers only
the two persistent FP64 basis/image arrays, not factors, reduced matrices,
transient tensors or CUDA graph caches. A fresh full-operator KKT check on the
returned parameter-dtype displacement must pass `update_rtol`; exhausted bases
fail explicitly and suppress the optimizer commit. There is no dense fallback.
CUDA uses a fixed schedule with masked actions after convergence; reported
iterations/evaluations count active work, not scheduled kernel launches.
The default remains `spectral`, which is faster for small tested problems.

`recompression="pcg", recompression_action="jvp_vjp"` selects streamed
JVP→VJP products for both recompression and cross-frame momentum transport.
It computes the complete `J_current.T @ J_previous @ x`, with at most 16
visible rows in scratch, without atom-pair Gram construction. Factor directions
still need O(K*(I+O)) storage. The existing `"pair"` action remains the default;
which contraction is faster depends on atom count and visible dimensions.
Damping, the full residual criterion and moment definitions are unchanged.
The bound derivative API exposes the same selection as
`site.cst_derivatives().tangent_ops(gram_action="jvp_vjp")`.

[Measured solver, recompression and learning tradeoffs](https://github.com/takumiecd/cst-experiments/blob/main/docs/torchcst-experiments/krylov-recompression.ja.md)
include successful large-atom trials, explicit basis-exhaustion cases and paired
MNIST results. A passing linear-system residual does not establish equal training
accuracy; the combined Krylov/streamed path has not established that equivalence.

[Time to accuracy and peak memory](https://github.com/takumiecd/cst-experiments/blob/main/docs/torchcst-experiments/time-to-accuracy.ja.md)
compares three seeds over 512 updates, recording both first observed targets and
three consecutive confirmations. Unreached targets remain explicitly censored.

[Dense versus CST: shared timing protocol](https://github.com/takumiecd/cst-experiments/blob/main/docs/torchcst-experiments/dense-comparison.ja.md)
