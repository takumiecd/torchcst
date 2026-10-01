from __future__ import annotations

import pytest
import torch
from kernel_cases import amplitude_state, gaussian_state, separable_state

from torchcst import (
    Chart,
    CSTLinear,
    GaussianSpec,
    LinePattern,
    ProductChart,
    presets,
)


def make_kernel() -> amplitude_state:
    return amplitude_state(
        separable_state(
            input_profile=gaussian_state(0.4), output_profile=gaussian_state(0.7)
        )
    )


def make_model(*, backend: str = "factored") -> CSTLinear:
    return CSTLinear(
        Chart.linspace(5, low=-1.0, high=1.0),
        Chart.grid((2, 2), low=-1.0, high=1.0),
        atoms=3,
        kernel=make_kernel().declaration(),
        backend=backend,
    )


def test_module_owns_one_opaque_fixed_shape_atom_table() -> None:
    model = make_model()
    assert model.atom_count == 3
    assert model.atoms.p.shape == (3, 4)
    assert model.in_features == 5
    assert model.out_features == 4
    assert not hasattr(model.atoms, "weight")
    assert not hasattr(model, "source")
    assert not hasattr(model, "target")
    assert not hasattr(model, "amplitude")


def test_dense_weight_is_the_canonical_kernel_sum() -> None:
    model = make_model()
    expected = model.materialized_atoms().sum(dim=0)
    torch.testing.assert_close(model.dense_weight(), expected)


def test_factored_and_materialized_backends_are_equivalent() -> None:
    torch.manual_seed(7)
    factored = make_model(backend="factored")
    materialized = make_model(backend="materialized")
    materialized.load_state_dict(factored.state_dict())
    inputs = torch.randn(2, 4, factored.in_features)
    expected = torch.nn.functional.linear(inputs, factored.dense_weight())
    torch.testing.assert_close(factored(inputs), expected)
    torch.testing.assert_close(materialized(inputs), expected)


def test_forward_gradients_reach_the_opaque_atom_parameters() -> None:
    model = make_model()
    model(torch.randn(8, model.in_features)).square().mean().backward()
    assert model.atoms.p.grad is not None
    assert tuple(model.kernel.parameters()) == ()


def test_nonfactorized_kernel_uses_the_same_kernel_sum_semantics() -> None:
    model = CSTLinear(
        chart=ProductChart(
            shape=(2, 3),
            axes=(LinePattern(2, spacing=0.5), LinePattern(3, spacing=0.5)),
        ),
        atoms=2,
        kernel=presets.radial(presets.fixed_profile(GaussianSpec(), 1.0)),
        backend="auto",
    )
    inputs = torch.randn(4, 3)
    torch.testing.assert_close(model.dense_weight(), model.materialized_atoms().sum(0))
    torch.testing.assert_close(
        model(inputs), torch.nn.functional.linear(inputs, model.dense_weight())
    )


def test_factored_backend_rejects_a_kernel_without_that_capability() -> None:
    with pytest.raises(ValueError, match="does not support factorized"):
        CSTLinear(
            chart=ProductChart(
                shape=(2, 3),
                axes=(LinePattern(2, spacing=0.5), LinePattern(3, spacing=0.5)),
            ),
            atoms=2,
            kernel=presets.radial(presets.fixed_profile(GaussianSpec(), 1.0)),
            backend="factored",
        )


def test_constructor_dtype_covers_charts_kernel_and_atoms() -> None:
    model = CSTLinear(
        Chart.linspace(3, low=-1.0, high=1.0),
        Chart.linspace(2, low=-1.0, high=1.0),
        atoms=2,
        kernel=make_kernel().declaration(),
        dtype=torch.float64,
    )
    assert all(
        value.dtype == torch.float64
        for value in model.state_dict().values()
        if isinstance(value, torch.Tensor)
    )
    assert model(torch.ones(1, 3, dtype=torch.float64)).dtype == torch.float64


def test_forward_rejects_the_wrong_feature_count() -> None:
    model = make_model()
    with pytest.raises(ValueError, match="expected input shape"):
        model(torch.randn(2, model.in_features + 1))
