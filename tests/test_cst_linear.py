import pytest
import torch

from torchcst import Chart, CSTLinear, Gaussian


def make_model(*, backend: str = "factored") -> CSTLinear:
    return CSTLinear(
        Chart.linspace(5),
        Chart.grid((2, 2)),
        atoms=3,
        input_kernel=Gaussian(0.4),
        output_kernel=Gaussian(0.7),
        backend=backend,
    )


def test_module_owns_one_fixed_shape_atom_table() -> None:
    model = make_model()

    assert model.source.shape == (3, 1)
    assert model.target.shape == (3, 2)
    assert model.amplitude.shape == (3,)
    assert model.in_features == 5
    assert model.out_features == 4


def test_factored_and_materialized_backends_are_equivalent() -> None:
    torch.manual_seed(7)
    factored = make_model(backend="factored")
    materialized = make_model(backend="materialized")
    materialized.load_state_dict(factored.state_dict())
    inputs = torch.randn(2, 4, factored.in_features)

    expected = torch.nn.functional.linear(inputs, factored.dense_weight())

    torch.testing.assert_close(factored(inputs), expected)
    torch.testing.assert_close(materialized(inputs), expected)


def test_forward_gradients_reach_all_atom_parameters() -> None:
    model = make_model()

    model(torch.randn(8, model.in_features)).square().mean().backward()

    assert model.source.grad is not None
    assert model.target.grad is not None
    assert model.amplitude.grad is not None


def test_omitted_output_kernel_reuses_dimension_agnostic_gaussian() -> None:
    kernel = Gaussian(0.5, trainable=True)
    model = CSTLinear(
        Chart.linspace(3),
        Chart.grid((2, 2)),
        atoms=2,
        input_kernel=kernel,
    )

    assert model.input_kernel is kernel
    assert model.output_kernel is kernel
    assert sum(parameter is kernel._log_sigma for parameter in model.parameters()) == 1


def test_constructor_device_and_dtype_cover_the_composed_module() -> None:
    model = CSTLinear(
        Chart.linspace(3),
        Chart.linspace(2),
        atoms=2,
        input_kernel=Gaussian(0.5),
        dtype=torch.float64,
    )

    assert all(value.dtype == torch.float64 for value in model.state_dict().values())
    assert model(torch.ones(1, 3, dtype=torch.float64)).dtype == torch.float64


def test_forward_rejects_the_wrong_feature_count() -> None:
    model = make_model()

    with pytest.raises(ValueError, match="expected input shape"):
        model(torch.randn(2, model.in_features + 1))
