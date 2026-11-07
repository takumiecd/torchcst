# Center-driven Polar activity (research opt-in)

English (canonical) | [日本語](center-polar-activity.ja.md)

Keep z=(s,t), S=||z||², a=amplitude_max*s/sqrt(S), p=(S-1)/3, 1<=S<=4,
and r=L(|a|)^(1-p)*U(|a|)^p. Existing diagnostics call p q/alpha; it is not
the Atom-budget density. This opt-in changes the activity source, not Atom
layout, Linear, initialization, angular amplitude proposal or center retraction.
Physical radius remains detached from task gradients. At fixed radius W=aK
gives dLoss/da=<G,K> and dLoss/dmu_j=a<G,dK/dmu_j>. Identical current values
have identical Y/dX/all-Atom VJPs. Subsequent trajectories may differ because
support and Polar norm change.

## Update

Use accepted shortest FlatTorus displacements (negative branch at half-period
ties), divided by the old physical support on each axis:

$$E_\mu=\tfrac12[(\Delta\mu_{out}/r_{out,old})^2+
(\Delta\mu_{in}/r_{in,old})^2].$$

For original tangential amplitude proposal u, let d=(z+u)/||z+u||. Replace its
activity energy with E_mu, retaining the optional dormant term and exact decay:

$$S_* = \operatorname{clip}(S+\gamma E_\mu+3\eta\delta\omega(a_*),1,4),$$
$$S_{new}=\left[1-\frac{S_*-1}{S_*}e^{-4\lambda\eta}\right]^{-1},\qquad
z_{new}=d\sqrt{S_{new}}.$$

Gamma is activity_gain; lambda is radial_regularization; eta is outer optimizer
time. Delta/omega are the unchanged dormant-expansion term; set its rate to zero
to isolate center activity. Amplitude after rescaling is amplitude_max*d[0].
No hidden EMA, direct sigmoid/clip target assignment or new coordinate embedding
is introduced. Small sustained activity need not saturate p at one.

This normalized finite-step energy is a control choice, not uniquely implied by
task derivatives. It is not divided by optimizer time. Learning rate, update
frequency, support and gain affect its scale. Net shortest movement does not
count complete torus windings. Algebra alone establishes no learning advantage.

## Explicit research execution

Use activity_mode="center_chord" with RadiusBounds in kernels.polar. Bandwidth
coordinates reject it. Existing finite_chord/time_energy arithmetic/settings are
retained. Fused CUDA/Graph routes reject the new mode; checkpoint contracts
distinguish modes/gains. The new eager Torch Algorithm accepts CPU/CUDA
float32/float64 radius Atoms and two scalar FlatTorus centers. It is not registered
by default. The existing generic CPU reference can also execute the new declared
law. Both control and candidate must use this same update implementation
when attributing a difference to the activity source:

```python
from torchcst import CSTAdamW, FixedSelector, make_algorithm_registry
from torchcst._backends.schema import DefaultRecipe, ExecutionPlan
from torchcst._backends.torch.algorithms.polar_update.radius_eager import (
    RadiusEagerPolarUpdateAlgorithm,
)

registry = make_algorithm_registry()
registry.register(RadiusEagerPolarUpdateAlgorithm())
plan = ExecutionPlan("torch_radius_eager_polar_update", "v1", DefaultRecipe())
optimizer = CSTAdamW(
    model.parameters(), model=model, lr=3e-4,
    update_selector=FixedSelector(plan, registry=registry),
)
```

This private backend dependency is experimental, not a stable public alias.
Scalar-oracle, amplitude-only/stationary, periodic seam, all-Atom VJP,
metadata/support, checkpoint and optimizer/resume checks accompany the code.
CPU validation does not establish CUDA/full-task admission, model quality,
Graph behavior or a new default; record those scopes separately.

## Bounded A100 evidence

Frozen operator `1cb4bec3ae2f1a859a077bf1c664888082ccc7ab` and consumer
`ac64f6db4fdf616870c41fed779e1119ecc5e441` passed14 actual CPU/CUDA update
tests on A10080GB PCIe MIG3g.40gb / Torch2.6.0+cu126. Four CST recipes passed
independent quadrature, native rectangular Y/dX/all-Atom VJPs, mixed optimizer
and actual four-block task/resume. All initial Parameters/eight calibration
reports match across the CST arms. Five fresh learners completed400 updates;
trained audits passed32 site VJPs and five exact DEV/next-update resume checks.
All48 compact raw files were retrieved/hash-verified; original source/checkpoints
remain intact and owned workers are finished.

DEV CE: Dense4.135901; amplitude gain1 4.248089; center gain1/64/4096
4.247539/4.250420/4.262914. Center activity changes p/radius but this one-seed
short pilot establishes no quality improvement or default adoption. All CST
arms use the same eager update implementation and fixed minimum-001 envelope.
GPU learning/update parity here is FP32; CUDA FP64, Graph capture and other
workloads are not qualified by this record. [Results and curves](https://github.com/takumiecd/cst-research/blob/8492b5e89aa50f5fcfd2a489ead90e7ec5e2a770/docs/ffn-center-activity-20261107-results.md)
([research PR33](https://github.com/takumiecd/cst-research/pull/33)).
Archive SHA256 `19ad6885fa0d2c5a989b03e2b322be8cfa32272e575c59dc56a345c3dff3769d`.
