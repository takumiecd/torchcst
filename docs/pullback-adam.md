# PullbackAdam

`PullbackAdam` owns the learnable source and target coordinates of one
continuous CST site whose columns use `L2NormalizedColumns`. It keeps all
persistent moments in coordinate-shaped tensors. The practical `"diag"` and
`"block"` forms use block-Jacobi approximations of the represented map's
pullback metric; `"full"` is an expensive exact-coordinate oracle.

Let

$$
\delta W = J_t\delta\theta,
\qquad
G_t = J_t^\top J_t,
\qquad
Q_t = J_tG_t^{-1/2}.
$$

With the full metric and a full-column-rank Jacobian,
$Q_t^\top Q_t=I$. The implementation uses
$D_t=\operatorname{diag}(J_t^\top J_t)$ in its default `"diag"` form, so it
normalizes each Jacobian column while neglecting cross-atom and
cross-coordinate coupling. Damping replaces each selected metric form by an
effective positive metric before inversion.

## Metric form

`metric=` selects how much of the within-atom metric is used.

`"diag"` (default) is the historical per-axis diagonal above.

`"block"` keeps, per atom and per side, the full axis Gram

$$
R_{ab}(c)=\langle d_a(c),\,d_b(c)\rangle,
\qquad
d_a(c)=\frac{\partial u(c)}{\partial c_a},
$$

and applies its batched symmetric inverse square root $R^{-1/2}$
($d\times d$ `eigh`, $d$ the chart dimension) wherever `"diag"` divides by
$\sqrt{D_t}$. Under `L2NormalizedColumns` the amplitude row and the
source-target cross block of the within-atom metric vanish exactly
($\langle u,\partial u\rangle=0$ identically for a normalized family), so
these two small blocks are the *complete* within-atom metric: what remains
neglected is exactly the cross-atom coupling. On an irregular neuron cloud
the axis derivatives of one atom are not orthogonal — the M5 coherence
measurement found ~99% of the off-diagonal energy of $J_t^\top J_t$ inside
these blocks — and `"block"` removes that part structurally instead of
regularizing against it.

`"full"` is the deliberately expensive coordinate oracle. It constructs the
exact source/source, source/target and target/target Gram across every live
atom in one site and applies one dense symmetric inverse square root. The
separable CST factors are used directly, so no dense represented-map Jacobian
is materialised. Amplitudes are held fixed in this first oracle; their Adam
clock and lifecycle price remain independent. This mode is intended to test
whether cross-atom coordinate coupling matters before introducing sparse,
neighbour-block or matrix-free approximations.

Moments stay coordinate-shaped in all forms and the follower contract is
unchanged. `state_dict()` records the metric form and rejects restoration
into the other one. The fixed-metric equivalence of the two moment spaces
below is a property of `"diag"` only: Adam's elementwise nonlinearity does
not commute with the non-diagonal $R^{-1/2}$.

## Moment space

The public option is named `moment_space` because the two algorithms differ
only in where Adam's exponential moving averages are accumulated.

For the parameter-space gradient $g_t^\theta$:

```python
PullbackAdam(site, moment_space="parameter", cap_sigma=0.1)
```

accumulates

$$
m_t=\operatorname{EMA}_{\beta_1}(g_t^\theta),
\qquad
v_t=\operatorname{EMA}_{\beta_2}((g_t^\theta)^{\odot2}).
$$

The Adam direction $a_t$ is mapped to coordinates with the current metric:

$$
\delta\theta_t=-\eta D_t^{-1/2}a_t.
$$

The tangent alternative

```python
PullbackAdam(site, moment_space="tangent", cap_sigma=0.1)
```

first forms the gradient coefficient along the normalized tangent directions,

$$
r_t=D_t^{-1/2}g_t^\theta,
$$

and accumulates the moments of $r_t$. Its direction is mapped back with the
same current $D_t^{-1/2}$. This removes the coordinate sensitivity that
existed when each gradient arrived, but it is a moving-frame approximation:
moments are not parallel-transported when the tangent directions rotate.

When the diagonal metric is fixed, the two forms agree up to Adam's epsilon
arithmetic. They separate when the representation's local units move during
the moment horizon.

Neither form is elementwise Adam in the ambient dense weight coordinates.
Adam is basis-dependent: applying Adam after projection into tangent
coefficients is generally different from applying dense Adam first and then
projecting its update.

When `PullbackAdam` owns amplitudes, `betas` still names the coordinate clock
and `amplitude_betas=(beta1_w, beta2_w)` may select a separate scalar-amplitude
clock.  Leaving it as `None` shares `betas`, preserving the original API.  The
separation is important in lifecycle experiments: setting coordinate
`beta1=0` asks a mobile reserve to follow the current tangent without also
removing momentum from amplitude settlement.

## Step calibration, cap, and travel

The learning rate is not a free parameter. On the first step the joint
per-atom norm of the whitened Adam direction,

$$
\rho_k=\sqrt{\lVert(\delta s_k)\rVert^2+\lVert(\delta t_k)\rVert^2},
$$

is measured over live atoms and the scalar $\eta$ is frozen so that the
median displacement is `target_step` in kernel-sigma units:

$$
\eta=\frac{\texttt{target\_step}\cdot\sigma}{\operatorname{median}_k\rho_k}.
$$

Every subsequent step is clipped per atom at `cap_sigma`$\cdot\sigma$ on the
same joint $s\oplus t$ norm, and the applied displacement accumulates into
`travel` (a per-atom path length in sigma units — net displacement from init
is a different quantity and is deliberately not tracked here; an atom can
have large travel and zero net displacement).

### Distance-clock moment forgetting

Ordinary Adam forgets on the optimizer-step clock.  That is a poor clock for
a mobile atom: a reserve pinned at the per-step cap can cross a kernel
neighbourhood while its first and especially second moments still describe
the old location and old tangent frame.  The opt-in

```python
PullbackAdam(
    site,
    moment_space="tangent",
    cap_sigma=0.1,
    moment_distance=(0.25, 1.0),
)
```

adds a geometric clock.  If the joint source-target displacement applied on
one step is $d_k$ in kernel-sigma units, the retained histories are

$$
m_k \leftarrow e^{-d_k/\tau_m}m_k,
\qquad
v_k \leftarrow e^{-d_k/\tau_v}v_k,
$$

where `moment_distance=(tau_m, tau_v)`.  The first number is normally shorter:
direction should adapt within a neighbourhood, while the RMS estimate may
average noise over a longer path.  The optimizer carries per-row normalising
masses and decays them by the same factors, so this extra forgetting changes
the relative weight of old and new evidence without corrupting Adam's bias
correction.  Amplitude moments remain on ordinary step-clock Adam even when
this optimizer owns them: the option is specifically a correction for the
moving coordinate frame and does not silently change the amplitude optimizer.

`None` is the default and is exactly the ordinary step-clock behaviour.  The
option does not reduce the current step or the cap; it prevents a fast-moving
atom from spending later capped steps following stale evidence.  A useful
screening range is $\tau_m=0.1\ldots0.5\,\sigma$ and
$\tau_v=0.5\ldots2\,\sigma$, measured against net arrival and score gain rather
than path length alone.

## Structural forces

Three optional forces ship with the optimizer. All of them act inside
`step()`; none of them require a loss term.

**Rent** (`rent=SmoothRent(...)`, amplitude ownership). The rent gradient
joins the amplitude gradient *before* the moments,

$$
g_w \leftarrow g_w+\lambda(t)\,\frac{w}{\sqrt{w^2+\varepsilon^2}},
$$

so a price larger than the loss-gradient scale is normalized away by
$\sqrt{v}$ — the CL1 measurement: the effective charge saturates at the
learning rate and $\lambda$ stops being the price. `decay=` is the
decoupled alternative in AdamW's sense, charged after the step,

$$
w \leftarrow w-\texttt{lr\_w}\cdot\texttt{decay}\cdot w,
$$

which $\sqrt{v}$ cannot eat. They compose; either one hands the amplitudes
to this optimizer.

**Pair repulsion** (`repulsion=PairRepulsion(mu)`). The potential over live
atoms,

$$
V=\mu_r\sum_{i<j}\exp\!\left(-\tfrac{1}{2}r_{ij}^2\right),
\qquad
r_{ij}^2=\frac{\lVert s_i-s_j\rVert^2}{\sigma_{\mathrm{in}}^2}
        +\frac{\lVert t_i-t_j\rVert^2}{\sigma_{\mathrm{out}}^2},
$$

contributes $-\partial V/\partial(s,t)$ to the coordinate gradient before
whitening and before the moments. Two atoms repel only when close on
*both* sides — exactly when their columns collide. Amplitudes do not
enter, so the force is mass-blind. Above the pair budget the sum is
replaced by uniformly sampled ordered pairs, an unbiased stochastic
gradient. Because the force enters before the moments, its magnitude is
partially normalized away like rent: $\mu_r$ decides which term wins the
*direction* of a coordinate's step, while the step *size* stays governed
by the target-step calibration.

**Wall** (`wall=True`). After the update, coordinates are clamped
per axis into the store's chart box. Note the wall lives in this
optimizer: coordinates trained by any other owner are unconfined.

## Chart coordinates (`ChartPullbackAdam`)

Status: designed and implemented 2026-08-23 (atom-mobility arc, AM3) —
`torchcst.optim.chart`, mechanism tests in
`tests/torchcst/test_chart_pullback.py`. The chart sample points
$\mu_i$ of a `NeuronStore` are shared: one chart is read by every incident
site-side (in lm1's FFN, the hidden chart is the `up` output side and the
`down` input side).

### The map and its Jacobian

In the amplitude gauge a site represents

$$
W_{ij}=\sum_k w_k\,\Phi^{\mathrm{out}}_{ik}\,\Phi^{\mathrm{in}}_{jk},
\qquad
\Phi^{\mathrm{out}}_{ik}=\varphi\!\left(\frac{\mu^{\mathrm{out}}_i-t_k}{\sigma}\right),
$$

with the Gaussian $\varphi(z)=\exp(-\lVert z\rVert^2/2)$. A chart point
$\mu_i$ on the output side of one site moves **row $i$** of that site's
$W$ and nothing else:

$$
\frac{\partial W_{ij}}{\partial\mu_{i,a}}
=\sum_k w_k\,\Phi^{\mathrm{in}}_{jk}\,\Phi^{\mathrm{out}}_{ik}\,
 \frac{t_{k,a}-\mu_{i,a}}{\sigma^2}
\;=\;\sum_k \beta^{(a)}_k\,\Phi^{\mathrm{in}}_{jk},
\qquad
\beta^{(a)}_k=w_k\Phi^{\mathrm{out}}_{ik}\frac{t_{k,a}-\mu_{i,a}}{\sigma^2}.
$$

An input-side incidence is the transpose statement (column $i$). Since
$\mu_i$ appears in several sites, its full Jacobian maps into the
*product* of the incident maps, $J(\mu_i):\mathbb{R}^d\to\bigoplus_m
\mathbb{R}^{n_m}$, and the pullback of the product (Frobenius) metric is
the **sum of the per-incidence pullbacks**:

$$
G(\mu_i)=\sum_{m\in\mathrm{inc}(i)} J_m(\mu_i)^\top J_m(\mu_i)
\qquad(d\times d\ \text{per neuron}).
$$

The loss gradient needs no such assembly: $\mu$ is one shared parameter,
so autograd already delivers $\partial L/\partial\mu_i$ summed over
incidences (the lean L2 backward ships this since MN1).

### The normalization spreads the perturbation

The single-row statement above holds for the *raw* columns. Under
`L2NormalizedColumns` it does not survive: the delivered side column is
$p=v/\lVert v\rVert$, and since $\lVert v\rVert$ contains entry $i$,

$$
\frac{\partial p}{\partial\mu_{i,a}}
= g_a\,(e_i - p\,p_i),
\qquad
g_a=\frac{\partial v_i/\partial\mu_{i,a}}{\lVert v\rVert},
\qquad
\left\lVert\frac{\partial p}{\partial\mu_{i,a}}\right\rVert^2
= g_a^2\,(1-p_i^2),
$$

so one chart point perturbs **every row** of its side's columns. (An
early draft claimed cross-neuron blocks vanish exactly by disjointness of
rows; the autograd oracle in the tests refuted it — the derivation above
is what the oracle confirms, to machine precision on a single atom.)
Consequences:

- cross-*incidence* coupling is still exactly zero (different components
  of the product), but within one side the cross-neuron block is
  $-\sum_k w_k^2\,g^{(i)}g^{(j)}p_ip_j$ per atom — second order in the
  column entries, small for peaked columns, not zero;
- the per-neuron $d\times d$ block is therefore the same deliberate
  block-Jacobi cut the atom metric makes, and its *within-neuron* content
  is exact per atom through the $(1-p_i^2)$ projection;
- chart–atom cross coupling stays neglected — the one-owner split between
  the chart optimizer and the site optimizers.

### Closed form of the per-neuron block

In the delivered (normalized) gauge the other side's column has unit norm
and the projection is the $(1-p_i^2)$ factor above, so dropping cross-atom
terms (the same neglect the atom metric makes) the implemented block is

$$
G_{ab}(\mu_i)\;\approx\;\sum_k w_k^2\,
f_{ik}^2\,\big(1-p_{ik}^2\big)\,
(\mu_{i,a}-c_{k,a})(\mu_{i,b}-c_{k,b}),
$$

where $(p, f)$ are exactly the ``(unit, factor)`` pair of
`metric.normalized_columns` for the chart-side kernel and $c$ is the
atom coordinate on that side — the same closed-form family the atom
metric already computes, evaluated per incidence and summed per the
product-metric section. Per atom the direction of
$\partial p/\partial\mu_i$ is axis-independent, so each atom contributes
a rank-one $d\times d$ term; the sum over atoms fills the block. Under
`L2NormalizedColumns` the normalized column obeys
$\langle u_k,\partial u_k/\partial\mu\rangle=0$ identically (differentiate
$\lVert u_k\rVert^2=1$), so the amplitude–chart cross block vanishes for
the same reason the amplitude–coordinate block does for atoms.

### Moments, calibration, forces

Everything downstream of the metric is unchanged in form, per chart
instead of per site: tangent or parameter moments of
$r_t=G^{-1/2}(\mu)\,g_t^\mu$; first-step $\eta$ from the median
per-neuron step against the chart's own $\sigma$; per-neuron cap and
travel (single-chart norm — a neuron has no second side). The repulsion
is the single-sided specialization

$$
V=\mu_r\sum_{i<j}\exp\!\left(-\frac{\lVert\mu_i-\mu_j\rVert^2}{2\sigma^2}\right),
$$

and the wall clamps $\mu$ into the chart box.

### Kernel genericity

The chart metric must be written against the kernel contract, not the
Gaussian: `metric.normalized_columns(kernel, mu, centers)` already
delivers `(unit, factor)` for any registered radial family through
`profile` / `profile_grad`, and the assembly above only uses the radial
identity $\partial\kappa/\partial c=\texttt{factor}\cdot(c-x)$. The fused
`_gaussian_*` functions are a compiled fast path, gated on
`isinstance(..., GaussianKernel)`, and stay optional. Known limit of the
contract (chart and atom metrics alike): the squared distance is computed
outside the kernel with a scalar $\sigma$, so an axis-wise-$\sigma$
family needs the kernel to own the distance — a separate contract
extension, not part of this design.

### Ownership and the rejected alternative

One chart optimizer per `NeuronStore`, holding references to the incident
sites (for the metric only — the gradient arrives pre-summed). The
ordinary optimizer must exclude $\mu$, exactly as it excludes
`synapses.s`/`synapses.t` today. Charts have no lifecycle, so the
follower contract is trivial; index charts stay buffers (only continuous
families are learnable).

The per-site-copy alternative — unsharing $\mu$ so each site owns a
private chart — was considered and rejected: the hidden activation's
$i$-th component would be *written* at one position and *read* at
another, which destroys the chart-as-position semantics that spacing
laws, neighbour overlap, ceiling analysis, and repulsion all stand on,
and doubles the chart parameters for the privilege.

## Optimizer ownership

The ordinary optimizer must exclude `synapses.s` and `synapses.t` so that each
coordinate has exactly one owner:

```python
from torchcst import PullbackAdam

coordinate_ids = {id(layer.synapses.s), id(layer.synapses.t)}
ordinary = torch.optim.AdamW(
    [p for p in layer.parameters() if id(p) not in coordinate_ids],
    lr=1e-3,
)
coordinates = PullbackAdam(
    layer,
    moment_space="tangent",
    cap_sigma=0.1,
    target_step=0.01,
)

ordinary.zero_grad(set_to_none=True)
coordinates.zero_grad(set_to_none=True)
loss.backward()
ordinary.step()
coordinates.step()
```

`lr_scale` on `coordinates.step(lr_scale)` accepts the same multiplicative
schedule applied to the ordinary optimizer. Birth, death, capacity growth,
and remapping automatically update the slot-indexed moments through the
store's follower contract. `state_dict()` contains the selected moment space
and rejects restoration into the other variant.
