"""Independent-maturity Gaussian: continuation and geometry contracts."""

from __future__ import annotations

import torch

from torchcst import MaturityGaussianKernel, PullbackAdam
from torchcst.compute import CSTLinear
from torchcst.representation import (
    GaussianKernel,
    L2NormalizedColumns,
    RepresentationSpec,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


def _site(*, maturity: float = -2.0):
    dtype = torch.float64
    spec = RepresentationSpec.continuous(
        1, 1, kernel="maturity_gaussian"
    )
    store = SynapseStore(
        "maturity",
        1,
        1,
        1,
        spec=spec,
        dtype=dtype,
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.31]], dtype=dtype),
                torch.tensor([[0.67]], dtype=dtype),
                torch.tensor([0.4], dtype=dtype),
                torch.tensor([0], dtype=torch.int64),
                extras={
                    "maturity": torch.tensor([[maturity]], dtype=dtype)
                },
            )
        ]
    )
    inputs = NeuronStore(
        "maturity-in",
        17,
        mu=torch.linspace(0.0, 1.0, 17, dtype=dtype).reshape(-1, 1),
        initial_live=17,
        dtype=dtype,
    )
    outputs = NeuronStore(
        "maturity-out",
        19,
        mu=torch.linspace(0.0, 1.0, 19, dtype=dtype).reshape(-1, 1),
        initial_live=19,
        dtype=dtype,
    )
    module = CSTLinear(
        inputs,
        outputs,
        store,
        MaturityGaussianKernel(0.12, learnable=False).double(),
        gauge=L2NormalizedColumns(),
    )
    return module, store


def test_maturity_is_a_declared_atom_column_with_a_broad_birth() -> None:
    spec = RepresentationSpec.continuous(
        2, 3, kernel="maturity_gaussian"
    )
    assert spec.atom_cost == 2 + 3 + 1 + 1
    store = SynapseStore("birth", 2, 3, 1, spec=spec)
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.zeros(1, 2),
                torch.zeros(1, 3),
                torch.ones(1),
                torch.tensor([0]),
            )
        ]
    )
    torch.testing.assert_close(store.maturity[0], torch.tensor([-2.0]))


def test_maturity_interpolates_from_broad_to_the_ordinary_gaussian() -> None:
    query = torch.linspace(0.0, 1.0, 101, dtype=torch.float64).reshape(-1, 1)
    center = torch.tensor([[0.5]], dtype=torch.float64)
    adaptive = MaturityGaussianKernel(0.1, learnable=False).double()
    ordinary = GaussianKernel(0.1, learnable=False).double()
    broad = adaptive.scaled_columns(
        query, center, {"maturity": torch.tensor([[-30.0]])}
    )
    mature = adaptive.scaled_columns(
        query, center, {"maturity": torch.tensor([[30.0]])}
    )
    expected = ordinary.scaled_columns(query, center)

    # The broad reserve retains much more mass five ordinary sigmas away.
    assert broad[0, 0] > 1.0e5 * mature[0, 0]
    torch.testing.assert_close(mature, expected, rtol=2e-12, atol=2e-14)


def test_l2_gauge_keeps_w_as_mass_and_maturity_receives_gradient() -> None:
    module, store = _site()
    dense = module.dense_weight()
    torch.testing.assert_close(dense.norm(), store.w[0].abs())
    dense.square().sum().backward()
    assert store.maturity.grad is not None
    # Norm alone is maturity-invariant under the gauge, up to roundoff.
    assert store.maturity.grad.abs().max() < 1.0e-12

    module.zero_grad(set_to_none=True)
    probe = torch.linspace(-0.4, 0.7, dense.numel(), dtype=dense.dtype).reshape_as(dense)
    (module.dense_weight() * probe).sum().backward()
    assert store.maturity.grad is not None
    assert store.maturity.grad.abs().sum() > 1.0e-6


def test_independent_maturity_preserves_amplitude_coordinate_orthogonality() -> None:
    module, store = _site()
    dense = module.dense_weight()
    tangent_w = torch.autograd.grad(
        dense, store.w, grad_outputs=torch.ones_like(dense), retain_graph=True
    )[0]
    # Form the actual Jacobian columns explicitly for this tiny map.
    jac_w = torch.autograd.functional.jacobian(
        lambda weight: (
            module._kernel_matrices(store.s, store.t)[1] * weight
        ) @ module._kernel_matrices(store.s, store.t)[0].transpose(0, 1),
        store.w,
    ).reshape(-1)
    jac_s = torch.autograd.functional.jacobian(
        lambda source: (
            module._kernel_matrices(source, store.t)[1] * store.w
        ) @ module._kernel_matrices(source, store.t)[0].transpose(0, 1),
        store.s,
    ).reshape(-1)
    assert tangent_w.numel() == 1
    torch.testing.assert_close(jac_w @ jac_s, jac_w.new_zeros(()), atol=2e-13, rtol=0)


def test_pullback_metric_tracks_a_maturity_gaussian_coordinate_step() -> None:
    module, store = _site()
    optimizer = PullbackAdam(
        module, moment_space="parameter", metric="block", cap_sigma=0.1
    )
    live = store.live_slots()
    gram_s, gram_t = optimizer._metric_gram(live)
    delta_s = torch.tensor([[1.3e-6]], dtype=store.s.dtype)
    delta_t = torch.tensor([[-0.8e-6]], dtype=store.t.dtype)
    before = module.dense_weight().detach()
    with torch.no_grad():
        store.s.add_(delta_s)
        store.t.add_(delta_t)
    actual = (module.dense_weight().detach() - before).square().sum()
    predicted = (
        delta_s[0] @ gram_s[0] @ delta_s[0]
        + delta_t[0] @ gram_t[0] @ delta_t[0]
    )
    torch.testing.assert_close(actual, predicted, rtol=3e-5, atol=1e-18)
