# torchcst — fixed-shape continuous operators for PyTorch

> [!WARNING]
> **Ground-up research rewrite.** This branch defines the target contract and
> implements it in separately tested milestones. Imports may be unavailable
> until their milestone lands. There is intentionally no compatibility promise for earlier
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
- a full second-order CST displacement and unapproximated quartic objective;
- direct commit of the trust-bounded quartic proposal;
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
    Amplitude,
    AmplitudeBandwidthSeparable,
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
    kernel=Amplitude(
        Separable(
            input_profile=Gaussian(sigma=0.25),
            output_profile=Gaussian(sigma=0.10),
        )
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
5. a `CSTOptimizer` partitions and exclusively owns every trainable parameter;
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

The amplitude-dependent output-bandwidth variant is:

```python
AmplitudeBandwidthSeparable(
    input_profile=Gaussian(sigma=0.25),
    output_profile=Gaussian(sigma=0.10),  # narrow/committed width
    sigma_explore=float("inf"),           # weak-atom width
    tau=0.005,
    temperature=0.25,
)
```

It keeps the same opaque layout $(w_a,s_a,t_a)$, but smoothly interpolates
output precision between `sigma_explore` and `output_profile.sigma` using the
even gate

$$
g(w)=\operatorname{sigmoid}\left(
\frac{\log(w^2+\epsilon_g^2)-2\log\tau}{T}
\right).
$$

Thus weak atoms are broad explorers and sufficiently strong atoms approach the
narrow Gaussian. Fixed positive `gate_eps` makes the map twice differentiable
at zero amplitude, which is required by the CST Hessian contractions. These
bandwidth controls are fixed kernel configuration; a future learned threshold
or bandwidth must live in each opaque atom row.

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

### Exact structured quartic evaluation

`ImplicitAdamConfig(quartic_evaluation="auto")` selects an exact quadratic-feature
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
hardware-specific speed guarantee. A100 measurements improved individual
evaluations but did not improve complete `FullQuartic` solves, so CUDA currently
keeps the visible backend under `"auto"`. Explicit `"gram"` supports CUDA and
bypasses the automatic device and size restrictions.
Each problem builds a new step-local Gram matrix; it is never reused after a
parameter or moment update. The solver and its stopping tolerances are unchanged.

The setup-inclusive CPU benchmark and its limitations are described in
[the structured quartic report](docs/experiments/quartic-gram.md).

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

### Step contract

`CSTOptimizer` follows the ordinary PyTorch training order:

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
`FullQuartic` is the production default and deterministic comparison solver.
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
cst = ImplicitAdamConfig(
    quartic=BallNewton(starts=4, max_iter=30, max_evaluations=150),
)

# Adaptively solve exact quartics in small orthonormal subspaces.
cst = ImplicitAdamConfig(
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
See [the compiled Newton measurements](docs/experiments/compiled-newton.md).

`DeviceBFGS(max_iter=30, max_evaluations=150)` is an experimental alternative
for `ImplicitAdamConfig(quartic=..., device_execution=True)`. It retains the
complete quartic but changes the search to projected BFGS. Adaptive decisions,
validation flags, and moment compression remain on the device. Warmed CUDA
updates avoid host synchronization in the tested configuration; compilation,
initialization, and explicit `optimizer.check_errors()` are outside that scope.
Its iteration/convergence/boundary diagnostics are device tensors. Check errors
at an explicit reporting boundary; a failed check latches and disables updates.
See [device execution](docs/experiments/device-execution.md) for restrictions,
accuracy comparisons, and a training loop without per-step diagnostic reads.


For factor-capable kernels, `DeviceRay` can use an experimental compact
geometry that avoids visible Jacobians/Hessians and their graph-input copies:

```python
from torchcst import DeviceRay

cst = ImplicitAdamConfig(
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
See [compact derivative measurements](docs/experiments/factored-contractions.md).

With explicit positive `first_moment_damping`, `gram_solver="cholesky"` opts
into a faster device solve for the damped compression system. For example,
add `gram_solver="cholesky", first_moment_damping=1e-4` to the configuration
above. This changes the previous undamped, rank-truncated pseudoinverse to a
full-rank damped solve; it can change learning behavior. Cholesky status and
residual checks feed the same device failure latch. See the
[damped Gram measurements](docs/experiments/damped-gram.md) for the precision,
accuracy, and timing tradeoff. The default remains the Jacobi pseudoinverse.



`SubspaceQuartic` starts from gradient and atom-block-preconditioned directions,
constructs the exact restricted polynomial on the problem device, and solves
its small coefficients in CPU float64. It checks the original full-space
objective/residual, expands the basis, and restarts while retaining the best
candidate when the dimension cap is reached. Its `iterations` count reduced
models; `evaluations` include both reduced-polynomial and full-objective calls.
Both new solvers currently require float32/float64 and retain the best finite
objective reached, with zero among the candidates. Large dense Hessians can be
expensive; these implementations target the current few-hundred-variable sites.

See [the A100 solver comparison](docs/experiments/quartic-solvers.md) for setup
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
