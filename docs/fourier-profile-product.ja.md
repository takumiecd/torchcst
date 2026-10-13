# 有限Fourier profile product

`presets.polar_fourier_profile_product` は有限級数そのものをprofileとして宣言する。
FlatTorusのPeriodicGrid、または一周を覆うRegularGridに対応し、各座標のprofileを
掛け合わせる。既存Triweight/Gaussianの近似ではなく、kernel revision 4の別の演算である。

```python
import torch
from torchcst import (
    BandwidthBounds, CSTLinear, ChebyshevSquareSpec, KaiserSquareSpec,
    DefaultRecipe, ExecutionPlan, FixedSelector, chart_presets,
    make_algorithm_registry, presets,
)

chart = chart_presets.periodic_grid(
    (700, 500), periods=(70., 50.), output_dims=1,
)
kernel = presets.polar_fourier_profile_product(
    profiles=(ChebyshevSquareSpec(modes=64, tail_bound=1e-6),
              KaiserSquareSpec(modes=64, concentration_floor=3.5)),
    tile_widths=(10., 10.), amplitude_max=1., w_c=.1,
    bounds=BandwidthBounds(minimum=.1, birth=.1, maximum=10., upper_floor=.1),
)
layer = CSTLinear(chart=chart, kernel=kernel, atoms=16,
                  dtype=torch.float64, backend="factored")
x = torch.randn(2, 500, dtype=torch.float64)
y = layer(x)  # native H² at selected sites, whole-operator L2

# Explicit shared-tile CPU FP64 reference, with the same mathematical weights.
layer.selector = FixedSelector(
    ExecutionPlan("torch_fourier_tiles", "v1", DefaultRecipe()),
    registry=make_algorithm_registry(),
)
y_shared = layer(x)
```

`FourierProductKernelSpec` extends the mathematical declaration with
`FourierTileSupportSpec`. Older `KernelSpec` serialization and checkpoint fields
are unchanged. M, concentration/tail settings and tile support are part of the
mathematical function, not Recipe tuning. Different axes may use different families/M.

## Period, ownership and exact zeros

For coordinate d, let tile width be T_d and local profile period P_d=3T_d.
The chart's global circumference L_d can be larger; L_d/T_d must be an integer
of at least three, with an integer number S_d of sites per tile. RegularGrid
spacing must agree with L_d/n_d. Maximum Polar width cannot exceed any T_d.
Origins and periods are live chart buffers; valid eager reconfiguration is observed.

For a centre c and chart origin o, canonicalize c-o modulo L, assign
owner=floor(((c-o) mod L)/T), and keep only owner-1, owner, owner+1, modulo
the global tile count. The boundary belongs to the tile on its right.
Unwrap these neighbours relative to the owner origin, using offsets -T,0,T.
The local profile must use P, not the global L.

Inside these three tiles K has small tails. Outside them g is exactly zero
by the declared support mask. A nonzero finite trigonometric polynomial cannot
be zero on a whole open interval. There is no per-site clipping of K.
For more than three global tiles, ownership changes can discard tiny tails;
task VJP follows the selected branch and does not differentiate integer routing.
When the global chart has exactly three tiles, all sites remain selected.

## Native families and width floor

M is the highest positive frequency of K; there are 2M+1 real Fourier features.
M is even, with the reference accepting 2..512 subject to concentration feasibility.
H has degree M/2 and K=H², so the mathematical profile is nonnegative.
For all three families, normalize real symmetric h by sum(h)=1:

\[
H(d;\sigma)=\sum_{k=-M/2}^{M/2}h_k(\sigma)e^{ik\omega d},\quad
K=H^2=\sum_{k=-M}^{M}c_k e^{ik\omega d},\quad
\omega=2\pi/P,\quad c=h*h.
\]

Coefficient convolution is linear, zero-padded before FFT. A centre phase is
applied afterward; coefficients depend on current width, not centre.
Width is controlled by an effective concentration radius:

\[
r(\sigma)=\left[r_0^4+(1-(r_0/T)^4)\sigma^4\right]^{1/4}.
\]

This approaches a nonzero r_0 for small sigma and reaches T at sigma=T.
Sigma is a live Polar width parameter, not the literal half-height radius or
strict support radius. The narrow-profile study gives M64/sigma .1 half-height
radii approximately .338 (Chebyshev), .388 (DPSS), .429 (Kaiser) for the defaults.
The Torch coefficient definitions were independently compared with that study.

### ChebyshevSquareSpec

With epsilon=`tail_bound`, default1e-6:

\[
r_0=\frac P\pi\arccos\!\left[\operatorname{sech}
 \left(\frac{\operatorname{acosh}(\epsilon^{-1/2})}{M}\right)\right],\quad
\beta=\sec(\pi r/P),\quad
H(d)=\frac{T_M(\beta\cos(\pi d/P))}{T_M(\beta)}.
\]

At circular distance at least r, K is bounded by
sech²(M acosh beta) <= epsilon. This is a continuum bound for the mathematical
profile, not a strict floating-point bound. The identity for T_M is documented
by [DLMF](https://dlmf.nist.gov/18.5.E1). A requested epsilon/M combination
requiring r_0>=T is rejected. Native evaluation uses the polynomial recurrence;
coefficient preparation extracts H's finite coefficients from M+1 samples.

### KaiserSquareSpec

With eta=`concentration_floor`, default3.5, use r_0=eta P/(M+1),
NW=(M+1)r/P and B=pi NW:

\[
h_k\propto I_0\!\left(B\sqrt{1-(2k/M)^2}\right)/I_0(B).
\]

The [Kaiser window formula](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.kaiser.html)
defines the coefficients. Torch uses scaled `i0e` to avoid overflow. No
eigendecomposition or SciPy dependency is needed. There is no declared universal
tail bound; the study measures its tail/width tradeoff.

### DPSSSquareSpec

With eta=`concentration_floor`, default3, use the same r_0 and NW definition.
Take the largest-eigenvalue eigenvector of the symmetric tridiagonal of length
n=M+1, with diagonal ((n-1-2j)/2)² cos(2pi NW/n), j=0..n-1, and off-diagonal
j(n-j)/2, j=1..n-1, then normalize its sum to1.
[DPSS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.dpss.html)
uses this eigenvector formulation. Its concentration property concerns H's
energy; it does not establish optimal L2 leakage for K=H². The initial Torch
reference uses a dense eigensolve and remains a comparison candidate for
coefficient-generation cost. Both NW families require 0<eta<(M+1)/3.

## Whole-operator L2 and shared contraction

For raw masked axis functions g_ad, the atom is P_a=product_d g_ad and

\[
n_{ad}=\sqrt{\sum_i g_{ad}(i)^2},\quad
D_a=\max\left(\prod_d n_{ad},\varepsilon\right),\quad
W_a=\mathrm{amp}_a P_a/D_a.
\]

Apply the floor once to the complete product. The direct factored/materialized
Torch path sums actual site values, including aliased/coarse grids. Norms use
FP32 accumulation for FP16/BF16. Native preparation/evaluation uses FP64 before
casting factors to the parameter dtype; no all-FP32 preparation is claimed.

The three selected tiles form one complete uniform local period N=3S. Only if
N>2M does discrete orthogonality give the centre-independent shortcut

\[
n^2=N\sum_{k=-M}^{M}c_k^2.
\]

The shared reference rejects N<=2M and requires FP64-accurate period/spacing
consistency for its diagonal norm shortcut. Approximately aligned grids use
the direct path, which sums their actual sites.
It supports exactly one input/output coordinate, CPU FP64, and eager execution.
Other coordinate dimensions use the Cartesian direct reference.

For input tile t, store live moments Z_btk=sum_j X_btj exp(ik omega local_j).
Reuse them for every atom. Owner moments combine the three neighbouring Z with
fixed exp(ik omega nT) rotations, n=-1,0,1. Contract with c_k exp(-ik omega delta_a)
to produce H_ba/sqrt(D_a). On output, aggregate amplitude*H*phase coefficients
by owner, rotate/distribute to neighbours, and reconstruct sites once per tile.
Dense frequency Gram matrices and per-atom site factors are absent in this path.
This clarity-first reference uses complex moments and constructs bases each call;
an optimized real sine/cosine representation is future implementation work.

Task VJP detaches current sigma as in existing Polar products. Amplitude,
centres and the normalization denominator retain their applicable derivatives.
Current coefficients are regenerated after width updates; no stale width cache
is used. Checkpoints reject mismatched family/M/support. Existing atom and
optimizer state ownership and periodic centre updates are reused.

## Qualification and remaining work

`tests/test_fourier_profiles.py` independently prepares native coefficients in
NumPy and enumerates complete atom matrices before normalization. It checks
Y/dX/all-Parameter VJPs, M16/32/64, active/inactive floors, mixed families,
centres between sites and at routing seams, exact outside zeros, coarse-grid
fallback, D3 Cartesian products, valid live period/origin changes, FP32 direct
execution, checkpoint rejection and public optimizer updates.

CPU suite:1691 passed /2702 GPU/database skips /18 existing warnings.
72 independent coefficient comparisons with the saved SciPy/native research
study have maxabs7.88e-16. Raw logs and the comparison JSON are ignored
`output/fourier-profile-reference/` evidence. Scientific definitions and tests
do not establish a speed or memory win.

Triton/CUDA, all-FP32 reconstruction accuracy, GPU Graph replay, complete-step
timing and allocated/reserved peak measurement remain unimplemented/unvalidated.
The shared Plan is explicitly selected and is not added to the default selector.
