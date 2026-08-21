# PullbackAdam

`PullbackAdam` owns the learnable source and target coordinates of one
continuous CST site whose columns use `L2NormalizedColumns`. It keeps all
persistent moments in coordinate-shaped tensors and uses the block-Jacobi
diagonal of the represented map's pullback metric.

Let

\[
\delta W = J_t\delta\theta,
\qquad
G_t = J_t^\top J_t,
\qquad
Q_t = J_tG_t^{-1/2}.
\]

With the full metric and a full-column-rank Jacobian,
\(Q_t^\top Q_t=I\). The implementation uses
\(D_t=\operatorname{diag}(J_t^\top J_t)\), so it normalizes each Jacobian
column while neglecting cross-atom and cross-coordinate coupling. Damping
replaces it by an effective positive diagonal before inversion.

## Moment space

The public option is named `moment_space` because the two algorithms differ
only in where Adam's exponential moving averages are accumulated.

For the parameter-space gradient \(g_t^\theta\):

```python
PullbackAdam(site, moment_space="parameter", cap_sigma=0.1)
```

accumulates

\[
m_t=\operatorname{EMA}_{\beta_1}(g_t^\theta),
\qquad
v_t=\operatorname{EMA}_{\beta_2}((g_t^\theta)^{\odot2}).
\]

The Adam direction \(a_t\) is mapped to coordinates with the current metric:

\[
\delta\theta_t=-\eta D_t^{-1/2}a_t.
\]

The tangent alternative

```python
PullbackAdam(site, moment_space="tangent", cap_sigma=0.1)
```

first forms the gradient coefficient along the normalized tangent directions,

\[
r_t=D_t^{-1/2}g_t^\theta,
\]

and accumulates the moments of \(r_t\). Its direction is mapped back with the
same current \(D_t^{-1/2}\). This removes the coordinate sensitivity that
existed when each gradient arrived, but it is a moving-frame approximation:
moments are not parallel-transported when the tangent directions rotate.

When the diagonal metric is fixed, the two forms agree up to Adam's epsilon
arithmetic. They separate when the representation's local units move during
the moment horizon.

Neither form is elementwise Adam in the ambient dense weight coordinates.
Adam is basis-dependent: applying Adam after projection into tangent
coefficients is generally different from applying dense Adam first and then
projecting its update.

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
