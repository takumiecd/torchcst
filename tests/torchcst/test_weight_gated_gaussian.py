"""Explicit amplitude-dependent factor state and its dense-free P2 chain."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from torchcst.compute import CSTLinear, Factored
from torchcst.optim import contracted_cst_linear_p2_model
from torchcst.representation import (
    FactorState,
    L2NormalizedColumns,
    RepresentationSpec,
    WeightGatedGaussianFactor,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore

DTYPE = torch.float64


def _layer() -> tuple[CSTLinear, SynapseStore]:
    theta = torch.tensor(
        [[0.005, 0.31, 0.58], [-0.012, 0.72, 0.37]], dtype=DTYPE
    )
    store = SynapseStore(
        "weight-gated-p2",
        1,
        1,
        theta.shape[0],
        spec=RepresentationSpec.continuous(
            1, 1, factor="weight_gated_gaussian"
        ),
        dtype=DTYPE,
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                theta[:, 1:2],
                theta[:, 2:3],
                theta[:, 0],
                torch.arange(theta.shape[0], dtype=torch.int64),
            )
        ]
    )
    layer = CSTLinear(
        NeuronStore(
            "weight-gated-p2-in",
            6,
            mu=torch.linspace(0, 1, 6, dtype=DTYPE)[:, None],
            initial_live=6,
            dtype=DTYPE,
        ),
        NeuronStore(
            "weight-gated-p2-out",
            5,
            mu=torch.linspace(0, 1, 5, dtype=DTYPE)[:, None],
            initial_live=5,
            dtype=DTYPE,
        ),
        store,
        WeightGatedGaussianFactor(
            0.16, tau=0.005, temperature=0.25, gated=False
        ).double(),
        WeightGatedGaussianFactor(
            0.16, tau=0.005, temperature=0.25, sigma_explore=float("inf")
        ).double(),
        track_mass=False,
        gauge=L2NormalizedColumns(),
        backend=Factored(),
    )
    return layer, store


def test_gate_is_continuous_and_zero_weight_casts_a_uniform_column():
    factor = WeightGatedGaussianFactor(
        0.1, tau=0.005, temperature=0.25, sigma_explore=float("inf")
    ).double()
    query = torch.linspace(0, 1, 7, dtype=DTYPE)[:, None]
    centers = torch.tensor([[0.23], [0.71]], dtype=DTYPE)
    amplitude = torch.tensor([0.0, 0.1], dtype=DTYPE)
    state = FactorState(centers, amplitude)
    columns = L2NormalizedColumns().columns(factor, query, state)

    torch.testing.assert_close(
        columns[:, 0],
        torch.full((7,), 7.0**-0.5, dtype=DTYPE),
        atol=1e-14,
        rtol=0.0,
    )
    assert float(columns[:, 1].max() - columns[:, 1].min()) > 0.1
    assert float(factor.gate(amplitude)[0]) < 1e-50
    assert float(factor.gate(amplitude)[1]) > 0.999


def test_gauge_jet_exposes_amplitude_center_mixed_second_derivative():
    factor = WeightGatedGaussianFactor(
        0.14, tau=0.005, temperature=0.25
    ).double()
    query = torch.linspace(0, 1, 6, dtype=DTYPE)[:, None]
    state = FactorState(
        torch.tensor([[0.43]], dtype=DTYPE),
        torch.tensor([0.005], dtype=DTYPE),
    )
    jet = L2NormalizedColumns().jet2(factor, query, state)

    assert jet.value.shape == (6, 1)
    assert jet.jacobian.shape == (6, 1, 2)
    assert jet.hessian.shape == (6, 1, 2, 2)
    assert bool(torch.isfinite(jet.hessian).all())
    assert float(jet.hessian[:, 0, 0, 1].abs().max()) > 1e-4
    torch.testing.assert_close(
        jet.hessian[:, 0, 0, 1], jet.hessian[:, 0, 1, 0]
    )


def test_weight_gated_factored_p2_matches_dense_hessian_oracle():
    layer, store = _layer()
    generator = torch.Generator().manual_seed(91)
    inputs = torch.randn(8, 6, generator=generator, dtype=DTYPE)
    targets = torch.randn(8, 5, generator=generator, dtype=DTYPE)

    factored_output = layer(inputs)
    torch.testing.assert_close(
        factored_output, F.linear(inputs, layer.dense_weight())
    )

    # Both the training forward and P2 path must remain factored.
    layer.dense_weight = lambda: (_ for _ in ()).throw(
        AssertionError("dense_weight must not be called")
    )
    output = layer(inputs)
    loss = F.mse_loss(output, targets)
    (grad_output,) = torch.autograd.grad(loss, output, retain_graph=True)
    model = contracted_cst_linear_p2_model(layer, inputs, grad_output)
    gradients = torch.autograd.grad(
        loss, (store.w, store.s, store.t), allow_unused=False
    )
    slots = store.live_slots().to(store.w.device)
    expected_linear = torch.cat(
        (
            gradients[0].index_select(0, slots)[:, None],
            gradients[1].index_select(0, slots),
            gradients[2].index_select(0, slots),
        ),
        dim=1,
    )
    torch.testing.assert_close(model.linear, expected_linear)

    view = store.view()
    theta = torch.cat(
        (view.w.detach()[:, None], view.s.detach(), view.t.detach()), dim=1
    )

    # Dense W exists only inside this tiny correctness oracle.
    def oracle_score(candidate: torch.Tensor) -> torch.Tensor:
        weights = candidate[:, 0]
        state_in = FactorState(candidate[:, 1:2], weights)
        state_out = FactorState(candidate[:, 2:3], weights)
        k_in = layer.gauge.columns(
            layer.factor_in, layer.in_neurons.mu, state_in
        )
        k_out = layer.gauge.columns(
            layer.factor_out, layer.out_neurons.mu, state_out
        )
        dense_weight = (k_out * weights) @ k_in.T
        return (grad_output.detach() * F.linear(inputs, dense_weight)).sum()

    oracle_hessian = torch.func.hessian(oracle_score)(theta)
    oracle_blocks = torch.stack(
        [oracle_hessian[k, :, k, :] for k in range(theta.shape[0])]
    )
    torch.testing.assert_close(model.curvature, oracle_blocks)
    assert float(model.curvature[:, 0, 2].abs().max()) > 1e-4
    assert model.curvature.shape == (2, 3, 3)


def test_weight_gate_uses_no_extra_atom_storage_column():
    factor = WeightGatedGaussianFactor(0.2)
    assert factor.atom_columns == ()
    assert RepresentationSpec.continuous(
        1, 1, factor="weight_gated_gaussian"
    ).atom_cost == 3
