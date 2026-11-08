# Dense-compatible second-moment initialization

**English** | [日本語](initialization.ja.md)

`torchcst.init` provides explicit He/Kaiming and Xavier/Glorot presets for
`CSTLinear`. Compatibility means the same average operator second-moment
target as the corresponding Dense initializer, not independent dense weights,
identical spectra, row energies, gradients for every input, or model quality.
Constructor initialization is unchanged; these functions are opt-in.

## API

Construct a supported layer on the CPU, initialize it, then move it to CUDA
and create its optimizer:

```python
import torch
from torchcst import init as cst_init

# layer is a CPU CSTLinear with a two-chart Polar radius cell-mass kernel.
report = cst_init.kaiming_normal_(
    layer,
    mode="fan_in",
    nonlinearity="relu",
    radius_fraction=0.0,
    generator=torch.Generator().manual_seed(17),
)
print(report.relative_error, report.clipped_atoms, report.atom_count)
layer = layer.cuda()
# Create the optimizer after initialization.
```

Alternative presets and the extension point:

```python
report = cst_init.xavier_normal_(layer, gain=1.0, radius_fraction=0.25)
# Example target for a new preset: Dense weights of standard deviation 0.02.
report = cst_init.variance_scaling_(layer, variance=0.02**2, radius_fraction=0.0)
```

These alternatives must also be called while the layer is on the CPU, before
optimizer creation. Each call initializes anew. `generator` is optional and
must generate CPU samples. Lowercase `q`, `s`, `t` are used throughout.
Unlike `torch.nn.init`, these functions return an immutable
`InitializationReport`, not the input Tensor/Module. The report contains scheme,
q, Dense variance, target/expected energy, relative error, nominal amplitude
scale, physical amplitude RMS, clipping count, Atom count and evaluation count.
Save it with experiment metadata; it is not added to the module state dict.

Currently supported: CPU FP32/FP64 `CSTLinear`, two-chart Polar radius cell-mass
products. Unsupported kernels and GPU initialization raise errors. No dense W,
Dense model, training features or labels are needed. Initialization preserves
centers, declarations, Atom count, parameter shape and Parameter identity.
A failed search leaves the complete module state unchanged; its random
generator may have advanced. Initialize before backward/optimizer creation;
do not reuse existing optimizer moments after reinitialization.

## Bridge from Dense to CST

Write W=sum_a a_a Phi_a. With fixed amplitude-free shapes Phi_a, independent
zero-mean amplitudes of variance sigma_a² and E_K=sum_a ||Phi_a||_F²:

```math
\mathbb E\|W\|_F^2=\sigma_a^2 E_K,\quad
f_{in}=E_K/d_{out},\quad f_{out}=E_K/d_{in}.
```

He uses sigma_a²=gain²/f_mode; Xavier uses
sigma_a²=2*gain²/(f_in+f_out). Coordinate-basis Atoms for all dense entries
give E_K=d_in*d_out and recover the usual Dense formulas exactly. For square
maps with A Atoms of mean unit-amplitude energy e, He/ReLU becomes
sigma_a=sqrt(2/e)/sqrt(A/N).

The common API accepts a **Dense weight variance** v and targets
E||W||_F²=v*d_in*d_out. He obtains v=gain²/d_in or gain²/d_out using
`torch.nn.init.calculate_gain`; ReLU gain is sqrt(2). Xavier obtains
v=2*gain²/(d_in+d_out). Future presets can compute their Dense variance and call
`variance_scaling_`; the operator-specific solver does not need another preset.
Other activations and residual-depth scaling need their appropriate target.

## Radius fraction and actual solver

For rho=sqrt(s²+t²), amplitude a=a_max*s/rho and
q=clip((rho²-1)/3,0,1), physical radius is R=L(a)^(1-q)*U(a)^q.
Initialization chooses q in [0,1], sets rho=sqrt(1+3q), then encodes
s=rho*a/a_max, t=rho*sqrt(1-(a/a_max)²). Thus q=0 selects the lower bound,
q=0.5 the geometric midpoint and q=1 the upper bound. q differs from Atom
density alpha=A/(d_in*d_out). It selects an initial condition, not a constraint
fixing radial activity during training.

For Polar kernels, radius and factor energy also depend on amplitude magnitude.
The implementation therefore solves the moment target with that dependency
included. It samples a standard normal proposal z, symmetrically clips scale*z
to a_max*(-1+1e-6,1-1e-6), encodes actual module-dtype Atoms, and evaluates their
factor energies in CPU FP64. Magnitudes and centers are held fixed when
defining the independent-sign expectation:

```math
\mathbb E_{sign}\|W\|_F^2=
\sum_a a_a^2\|\Phi_a(|a_a|,q)\|_F^2.
```

Sign invariance of the geometry cancels cross terms even when Atoms overlap.
One realized W need not equal this conditional expectation. Factor vector norm
products avoid allocating an Atom-by-output-by-input array or dense weight.

The search tries relative scales 0,.025,.05,.1,.2,.4,.8,1.6,3.2,6.4,12.8,32
times a_max, then at most32 bisections of a sign bracket. It accepts measured
relative target error <=0.5%, without assuming global monotonicity. Failure is
reported with its history; it does not prove global unreachability or silently
change q, Atom count, bounds, kernel normalization or add an output gain.

`normal` describes the amplitude proposal before clipping. Heavy clipping can
occur, so physical amplitude RMS can differ substantially from nominal scale;
inspect the report. Bounded amplitudes and kernel profiles can make a target
unreachable. For nonnegative unit-mass Atom matrices and amplitude bound a_max,
the expected energy is at most A*a_max², a necessary feasibility bound.

The exploratory HIGGS pilot in
[cst-research](https://github.com/takumiecd/cst-research/pull/26) tests this moment
principle. Its frozen consumer and Gaussian draws are separate from this public
API; moving the method here does not rerun or revalidate that history. Initial
signal scaling and downstream accuracy are separate claims.

For widths above 1024, calibration evaluates squared cell-mass norms over each
Atom's local period rather than materializing full site-by-Atom factors.
Repeated periodic images are added before squaring when a local period exceeds
the chart. This bounds setup by actual support while preserving the same
conditional-sign energy target; reduction order can change FP64 last bits.
Widths up to 1024 retain their existing factor-norm calculation.
