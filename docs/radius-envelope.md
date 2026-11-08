# Amplitude-dependent radius envelope

English (canonical) | [日本語](radius-envelope.ja.md)

`kernels.radius_envelope` declares cell-mass coordinates with an independently
learned interpolation ratio q. It is an opt-in coordinate family, with its own
`RadiusEnvelopeSpec` and kernel revision8. Existing Polar/activity coordinates,
updates, checkpoints and default selectors retain their original contracts.

## Contract

Each Atom stores `[a, q, output_center, input_center]`. The signed amplitude is a;
q lies in[0,1]. The physical radius shared by the two factors is

\[
U=r_{\max},\qquad
L(a)=r_{\min}+(r_{\max}-r_{\min})e^{-(a/\tau)^2},\qquad
r=L(a)^{1-q}U^q.
\]

At a=0 both bounds meet at the maximum radius. Away from zero, L approaches the
minimum while U stays at the maximum. The amplitude threshold tau has the same
units as a; it is not divided by amplitude_max. At |a|=2tau the interval spans
98.17% of the full **linear radius difference** r_max-r_min. This is not a claim
about logarithmic interval width, useful exploration or learned accuracy.

q is an ordinary loss-trained Atom coordinate, not a coordinate-motion/activity
clock. Its endpoint derivatives are the one-sided continuation of the interior
formula:

\[
\partial_qr=r\log(U/L),\qquad
\partial_ar=r(1-q)L'(a)/L(a).
\]

Both terms reach the Atom gradients; there is no radius detach. At exactly zero
amplitude both the bound slope and the Atom's geometric loss gradient vanish.
This design requirement does not assert a universal optimal radius law.
Hierarchy/window routing remains piecewise constant as in physical cell masses;
at a routing boundary, the selected branch derivative is used.

## CPU example and optimizer

```python
import torch
from torchcst import CSTAdamW, CSTLinear, ChebyshevSpec, chart_presets, init, kernels

chart = chart_presets.regular_grid(128, geometry="flat_torus", spacing=1.0)
model = CSTLinear(
    chart, chart, atoms=1024, dtype=torch.float32, backend="factored",
    kernel=kernels.product(
        kernels.cell_mass(ChebyshevSpec(modes=8)),
        coordinates=kernels.radius_envelope(
            radius_min=0.5, radius_max=50.0, tau=0.03, q_init=0.0,
            amplitude_max=1.0, amplitude_std=0.1,
        ),
    ),
)
# Optional existing Dense-moment bridge, performed on CPU before the optimizer.
report = init.xavier_normal_(model, q=0.0)
optimizer = CSTAdamW(
    model.parameters(), model=model, lr=1e-3, weight_decay=0,
    field_lrs={"radius_ratio": 1e-3},
)
x = torch.randn(8, 128)
model(x).square().mean().backward()
optimizer.step()
q = model.atoms.p[:, 1]  # View of the learned Atom Parameter, not separate state.
r_input, r_output = model.support_radii()
```

The default amplitude proposal is Normal(0, amplitude_std), clipped to the
amplitude cap. Initialization presets `variance_scaling_`, `kaiming_normal_` and
`xavier_normal_` include the new envelope in their bounded energy search.
Their unchanged tolerance/admission rules apply; a failed search leaves the
model unchanged. These presets preserve centers and initialize q to their
explicit q argument, which supersedes constructor q_init.

`CSTAdamW`/`CSTOptimizer` projects proposals to |a|<=amplitude_max and q in[0,1]
and uses the existing geometry update for centers. `radius_ratio` is the semantic
LR field for column1; `amplitude`, `output_center` and `input_center` remain
separate fields. Adam moments and counters retain their ordinary meanings;
projection does not reset them. Plain torch.optim.AdamW alone does not enforce
the coordinate bounds. Invalid rows are rejected by evaluation.

Tau and radius endpoints are fixed configuration, stored with checkpoints;
q is per-Atom trainable state. Fixed constants retain FP64 precision through
Module.float()/double(); Atom Parameters follow the requested dtype. The
reference envelope map uses FP64 arithmetic/autograd intermediates, then casts
the physical row to the Atom dtype. Encoded radii are bounded inward when an
endpoint is not representable. This map is not an all-FP32 performance kernel.

## Explicit CUDA FFN composition

The initial GPU route composes the differentiable map with the existing physical
LM Algorithm. It requires FP32, width512/expansion4 (512->2048 or reverse),
8192 Atoms, flattened row count1..256, M8/base1/merge2, origin0/spacing1,
radius bounds within[0.5,50], amplitude_max<=1 and eager/autocast-off execution.
The underlying physical Plan applies its remaining support/precision checks.
No CUDA arithmetic changed. Existing physical/Polar Plans reject revision8;
passing q directly to a native radius entrypoint is forbidden.

```python
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.radius_envelope import (
    RADIUS_ENVELOPE_LM, RadiusEnvelopeLMAlgorithm,
)

registry = make_algorithm_registry()
registry.register(RadiusEnvelopeLMAlgorithm())
model.selector = FixedSelector(RADIUS_ENVELOPE_LM, registry=registry)
# Move a model with the qualified FFN shape/count to CUDA after CPU initialization.
```

Registration and selector choice are explicit; this example does not qualify the
128-square CPU model above for the FFN-only Plan. Graph capture and other GPU
shapes are unsupported. Metadata-only Plan tests do not establish GPU correctness
or performance. See the accompanying validation record for what actually ran.
