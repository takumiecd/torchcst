# Channel charts for radius cell-mass Conv2d

**English** | [日本語](radius-cell-mass-channel-charts-20261124.ja.md)

## Weight definition for the first candidate

Keep a unit-spaced 1D periodic Chart for input channels and one for output
channels. For a 3×3 convolution, associate a fixed subset of Atoms with each
of the nine spatial offsets. Each subset produces one channel matrix
$W_s[o,i]=\sum_{j\in A_s} a_j m^{\mathrm{out}}_j(o)m^{\mathrm{in}}_j(i)$.
Stack these matrices into the Conv2d weight and use one ordinary Conv2d for
stride, padding and dilation. The 1×1 case is the same construction with one
offset. This keeps a one-site channel displacement consistent across offsets.
The periodic channel ordering is a modeling assumption that should be tested
with the underlying dense architecture held fixed.

Fix the total number of Atoms over all offsets. Each rational Atom still has
four learned scalars. A later candidate can share an Atom's channel factors
across offsets with a separate spatial factor. That changes expressivity,
parameter counts and the meaning of amplitude in the radius law, so it needs
its own comparison. Channel-major flattening of a 3×3 patch makes one channel
step span nine Chart sites and joins unrelated spatial offsets at row and
channel boundaries. Treat the existing flattened 864-site configuration as a
separate weight definition in learning comparisons.

## Dyadic levels up to the radius cap

Use base tile 1, merge factor 2 and spacing 1 on both channel axes. For the
96→128 case choose $R_{\max}=32$ for the first fixture, matching the prior
layer experiment's cap. Every reachable tile width $T\in\{1,2,4,8,16,32\}$
divides both Chart periods. For spacing (h), the general condition is that
each reachable tile's site count divides the Chart size and that the largest
declared tile has physical width at least $R_{\max}$. Each selected local
period and window follow the existing cell-mass definition. In the fixture,
$P=(\lceil 2R/T\rceil+1)T\le96$. The 96-site period and its wrap are preserved.

The public declaration, Python AxisPlan and native parser currently require the
complete hierarchy to reach the Chart size with a uniform merge factor. An
explicit capped hierarchy needs consistent bank limits and cap validation in
all three places. For this fixture any representable $R>32$ requires a
different hierarchy or a rejected input. Test exact boundaries, both sides of
the transition, and window wrapping before admitting the route. A 128→128
fixture can use the existing reference's public dyadic declarations while the
96-site extension is developed.

For the earlier flattened 864-site input with base tile 27, $R=4$ selects
local period 54. A 96-site channel Chart with base tile 1 selects period 12
for the same radius. The earlier measurement is still useful for its specific
declaration. Clipping a padded 128-site Chart to 96 channels would have its own
boundary and mass contract and would need an independent learning comparison.

## Reuse from Linear

For both directions, expose a weight-only operation. Decode rational rows with
the existing `physical_parameters` path, retaining the amplitude-to-radius
and fraction-to-radius Jacobians. For each routed local window, $B_i,B_o$ map
the 17 real M8 coefficients to their wrapped Chart sites.
$B_o C B_i^\mathsf{T}$ writes each accumulated coefficient block into the
weight. For each tap, Conv2d backward supplies $G_s=\partial\ell/\partial W_s$.
Project it as $B_o^\mathsf{T}G_s B_i$ and contract with the Atom coefficient
derivatives on the selected forward branch. Ordinary convolution supplies
$dX$ and $dW$ with its spatial padding and overlap rules.

| Linear implementation | Applicability to this Conv2d design |
| --- | --- |
| `packed_make_geometry` and the cell Fourier coefficient/derivative routines | Reuse the per-axis computation and save each Atom's two memberships. Both axes choose the same level and bank for a shared physical radius under this fixture's spacing and hierarchy. Owner indices still wrap over different Chart sizes. |
| `packed_make_dense_weight` and `packed_dense_weight_grad` | These are the forward and VJP candidates. The VJP already consumes a dense $dW$. Extract a weight-only autograd entry point, parameterize the level range and owner counts, and carry the tap through grouping or batched contractions. Validate shallow levels, periodic edges, rectangular layouts and tensor strides. |
| `axis_parameter_gradient` and the direct row-projection Plan | Reuse the per-axis derivative formulas and the ordering idea. These launches consume row-domain signals and Atom responses. The weight-only Conv2d path obtains its cotangent from Conv2d and uses pair projections. |

Both dense native paths currently loop over a fixed level range and depend on
the admitted wide shapes. The existing native wrapper takes an activation
matrix and forms $dW=dY^\mathsf{T}X$. The proposed wrapper accepts the
per-tap weight cotangent, with its layout conversion included in timing.
Compare direct accumulation on short windows and Fourier owner-pair
contractions on broader windows at matched weights. The faster split for
small channel matrices requires measurement. Keep full optimizer-step timing
and allocated/reserved peaks for the dispatcher qualification.

## FP64 check

A CPU probe checked 96→128, stride 2, padding 0 or 1, with five rational M8
Atoms per offset. It tested unit mass and scalar Chebyshev quadrature over
both Chart seams and at the cap and hierarchy boundaries. Weight assembly
and the dense-weight adjoint were exercised separately. Across 1×1 and 3×3,
the maximum absolute discrepancies were $5.8\times10^{-15}$ for $W$
against quadrature, $2.1\times10^{-14}$ for $Y$ against quadrature,
$5.6\times10^{-16}$ for $dX$ against the direct mass reference, and
$2.5\times10^{-14}$ across all four Atom gradient fields against direct
autograd. An independent quadrature directional finite difference agreed with
the adjoint to $8.2\times10^{-9}$. This checks the mathematical design and
CPU reference. CUDA extraction, launch tuning and model-quality comparison
remain open.

The probe directly constructs the internal reference axis metadata for 96
sites with base tile 1 and checks that every selected tile divides the Chart.
The public capped declaration is future work. The ignored
`output/conv-radius-chart-20261124/` directory contains `check.py` and
`result.json`. From the worktree root run

```sh
OMP_NUM_THREADS=1 PYTHONPATH=src:tests python output/conv-radius-chart-20261124/check.py
```

Source `680742aa8a73b8fe7c4f85be5ce1e5be52c79bdf`. Probe SHA-256
`afd3dd2a0e07c7f460cb6eb74c4340f78d19eebdf2a27913e6cde9b138d0b14a`.
Result SHA-256 `6fdbb8453b960f0d0260a0d641916022910153ea7c6adc0032237c33edb5b8dd`.
