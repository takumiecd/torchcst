# Polar profile products on a single Chart

**English** | [日本語](profile-product.ja.md)

`CSTLinear(chart=..., kernel=presets.polar_profile_product(...))` declares an
operation multiplying coordinate-axis profiles on one Euclidean Product/Strip
Chart. The Chart's logical shape remains `(out_features, in_features)`.
`composition="profile_product"`, `parameterization=PolarAmpWidthSpec(...)` and
`update.id="polar_activity_width"` are separate declarations.

```python
import torch
from torchcst import (
    CSTLinear, CSTOptimizer, BandwidthBounds, TriweightSpec, presets,
    chart_presets as charts, pattern_presets as patterns,
)

chart = charts.product(
    shape=(16, 32),
    axes=(patterns.line(16, spacing=0.2), patterns.line(32, spacing=0.2)),
)
kernel = presets.polar_profile_product(
    profiles=(TriweightSpec(), TriweightSpec()),
    amplitude_max=1.0,
    bounds=BandwidthBounds(minimum=0.1, birth=1.0, maximum=2.0, upper_floor=0.1),
    w_c=0.1,
    radial_regularization=0.2,
)
layer = CSTLinear(chart=chart, atoms=12, kernel=kernel, backend="factored")
optimizer = CSTOptimizer(torch.optim.AdamW(layer.parameters(), lr=1e-3), model=layer)
optimizer.zero_grad()
layer(torch.randn(4, 32)).square().mean().backward()
optimizer.step()
```

## Coordinates and atom ownership

`profiles` is a tuple in Euclidean coordinate order: all coordinates of the
output Pattern, followed by all coordinates of the input Pattern. Even with two
logical axes, a 2D grid on the input side requires three profiles. The profile
count must match the Chart's coordinate dimension; profiles may have different
shapes on different axes. Each is assembled as a raw ProfileBinding without
bandwidth or normalization.

An atom Parameter is `[polar_x, polar_y, center_0, ..., center_(D-1)]`. The
existing activity-width law derives amplitude and shared sigma from the leading
Polar coordinates. Retain one Chart and one atom-center vector rather than
duplicating input/output Charts for computation. Existing Atoms own Parameters;
the base optimizer owns moments and step counters.

Initial support covers Euclidean geometry and Product/Strip Charts with Line/Grid
Patterns. That version rejects Explicit/Points Patterns and Sphere/Torus; do not
reinterpret their coordinates or distances as independent Euclidean axes. Strip
coordinates reflect tile_pitch and partial final tiles. Do not add
bandwidth-dependent restrictions on neighboring stations or per-tile normalization.

## Mathematical contract

### Torus centre-fibre product (revision 2)

`presets.polar_torus_profile_product` supports Torus Product/Strip Charts on
S¹×S². Its two profiles are ordered as circle, then S² cross-section, independently
of logical input/output axis order. Intrinsic-center Parameters are
`[polar_x, polar_y, arc, s1, s2]`; ambient-center Parameters contain the leading
two Polar coordinates and the existing four-dimensional center. Use the shared
Polar bandwidth and update law.

For center `(theta_a,q_a)`, major/minor radii `R,r`, circle sites `theta_i` and
cross-section sites `q_j`, define chord distances along fibres that hold the
other axis at the atom center:

$$
d_c(i;a)^2=4(R+r q_{a,0})^2\sin^2((\theta_i-\theta_a)/2),
\qquad d_s(j;a)^2=r^2\|q_j-q_a\|^2.
$$

For two raw profiles `u_a(i),v_a(j)`, normalization is

$$
W_{ij}=\sum_a A_a\frac{u_a(i)v_a(j)}
 {\max(\|u_a\|_2\|v_a\|_2,\epsilon)}.
$$

Do not detach the center dependence of circle radius `R+r*q_a0`. The cross-section
center receives gradients from both profiles and both norms. As in current
Polar, the task VJP holds the current bandwidth fixed, while the training-step
update law changes it. This composition differs from the existing joint radial
chord distance and does not replace the Euclidean revision 1 Chart checks or
semantics. Initialization, eager updates, gradient projection and moment
transport use the existing Torus Geometry. Fused CUDA and Graph-captured Torus
updates remain under development.

### Euclidean coordinate product (revision 1)

Let $a$ denote an atom, $d$ a coordinate axis, and $i_d$ all observed sites on
that axis. The raw Triweight axis profile is

$$
g_{a,d}(i_d)=\left[1-
\frac{(t_d(i_d)-c_{a,d})^2}{\sigma_a^2}\right]_+^3.
$$

Gaussian, Biweight and other declared axis profiles use their existing scalar
profiles. `QuadraticBSplineSpec()` is a [quadratic B-spline
(Japanese)](quadratic-bspline-profile.ja.md) with center-curvature bandwidth sigma
and support half-width `sqrt(6)*sigma`; its bandwidth meaning differs from
Triweight. Since all observed sites form a Cartesian product, the atom product
and norms are

$$
P_a(i_1,\ldots,i_D)=\prod_d g_{a,d}(i_d),\qquad
n_{a,d}=\sqrt{\sum_{i_d}g_{a,d}(i_d)^2},\qquad
\|P_a\|_2=\prod_d n_{a,d}.
$$

Here L2 is the norm of the vector containing every operator site: the Frobenius
norm for a matrix. Mathematical normalization **applies the floor once to the
complete product**:

$$
D_a=\max\!\left(\prod_d n_{a,d},\varepsilon\right),\qquad
W_a(j,i)=\mathrm{amp}_a\frac{P_a(j,i)}{D_a},\qquad
W=\sum_a W_a.
$$

The default is
`NormalizationSpec(kind="discrete_l2", domain="operator_sites", floor=1e-6)`.
A positive floor can be specified through `normalization=`. Do not apply separate
floors to each axis. For example, axis norms of 0.6 and a floor of 0.5 activate the
floor on their product, 0.36. Empty support contributes zero. Even a single
positive site must not be replaced with amplitude alone when the floor is active.
With the floor inactive, this agrees with multiplying the per-axis
L2-normalized profiles.

## Reference computation through H and differentiation

The backend can combine raw output/input axis products into $U_a(j),V_a(i)$ and
compute

$$
H_{b,a}=\sum_i X_{b,i}\frac{V_a(i)}{\sqrt{D_a}},\qquad
Y_{b,j}=\sum_a H_{b,a}\frac{\mathrm{amp}_a U_a(j)}{\sqrt{D_a}}
$$

Splitting the denominator between both factors avoids `amp / floor` overflowing
in FP16 and producing NaN when multiplied by zero for empty support. FP16/BF16
norms are computed in FP32, then factors are cast back to the original dtype.
Do not expand the full operator-site table or full raw product to compute norms.
Torch `factored` retains input/output factors and H as its reference method;
`materialized` constructs W. Evaluate every coefficient from current atoms and
Chart buffers on each call. `operator.factors()` also supports a single Chart.

As in existing Polar, task differentiation treats current sigma as constant while
retaining derivatives of amplitude, all centers and the normalization denominator.
Sigma is not a fixed setting: it changes after each Polar update. Use the same
update law for existing finite_chord/time_energy, the radial regularizer,
activity_gain and dormant_expansion_rate.

The old separable method applies a floor separately to input and output and
stores centers in input/output order. This method applies the floor to the
complete product and stores centers in the single Chart's output/input coordinate
order. Do not implicitly reinterpret old checkpoints or declarations. Record the
new normalization contract in checkpoints as well.

## Validation and implementation scope

`tests/test_profile_product.py` uses an independent FP64 oracle enumerating all
sites to check Y, dX, all source gradients, different axis profiles, 3D coordinates,
Strip pitch on both sides, partial final tiles, empty/single support and the
global floor. With the corresponding center order made explicit, compare AdamW
Parameters, moments, steps and changing bandwidths against existing separable
Polar, and check weights-only checkpoints.

This addition covers mathematical declarations and Torch reference execution.
Connecting it to CUDA local_product, an outer Strip schedule, and large-GPU
complete-step time and capture/replay peaks require separate validation. Preserve
the semantics and default recipes of existing radial, separable and strip_torus
methods.
