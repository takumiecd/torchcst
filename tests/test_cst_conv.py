import copy
import math

import pytest
import torch
import torch.nn.functional as F

from torchcst import (
    AdamWConfig,
    Amplitude,
    Chart,
    CSTConv2d,
    CSTLinear,
    CSTParameterAdam,
    CSTQuadraticAdam,
    Gaussian,
    Separable,
)


def make_kernel() -> Amplitude:
    return Amplitude(
        Separable(
            input_profile=Gaussian(0.4),
            output_profile=Gaussian(0.7),
        )
    )


def make_model(
    *,
    backend: str = "factored",
    in_channels: int = 2,
    out_channels: int = 4,
    kernel_size: tuple[int, int] = (2, 3),
    stride: tuple[int, int] = (2, 1),
    padding: tuple[int, int] = (1, 0),
    dilation: tuple[int, int] = (1, 2),
    groups: int = 1,
) -> CSTConv2d:
    patch_features = (in_channels // groups) * math.prod(kernel_size)
    return CSTConv2d(
        Chart.linspace(patch_features, low=-1.0, high=1.0),
        Chart.linspace(out_channels, low=-1.0, high=1.0),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        atoms=3,
        kernel=make_kernel(),
        backend=backend,
        dtype=torch.float64,
    )


def test_atoms_are_flattened_locally_and_exposed_as_conv_weights() -> None:
    model = make_model()

    flat = model.kernel.materialize_atoms(
        model.input_chart, model.output_chart, model.atoms.p
    )

    assert flat.shape == (3, 4, 12)
    assert model.materialized_atoms().shape == (3, 4, 2, 2, 3)
    assert model.dense_weight().shape == (4, 2, 2, 3)
    torch.testing.assert_close(
        model.materialized_atoms().flatten(start_dim=2),
        flat,
    )
    torch.testing.assert_close(
        model.dense_weight().flatten(start_dim=1),
        flat.sum(dim=0),
    )
    assert model.cst_derivatives().represented().shape == (4, 12)
    torch.testing.assert_close(model.cst_derivatives().represented(), flat.sum(dim=0))


def test_materialized_forward_matches_unfolded_local_linear_map() -> None:
    torch.manual_seed(7)
    model = make_model(backend="materialized")
    inputs = torch.randn(2, 2, 7, 9, dtype=torch.float64)

    outputs = model(inputs)
    patches = F.unfold(
        inputs,
        kernel_size=model.kernel_size,
        dilation=model.dilation,
        padding=model.padding,
        stride=model.stride,
    ).transpose(1, 2)
    expected = F.linear(patches, model.dense_weight().flatten(start_dim=1))
    expected = expected.transpose(1, 2).reshape_as(outputs)

    torch.testing.assert_close(outputs, expected)


def test_factored_and_materialized_backends_match_outputs_and_gradients() -> None:
    torch.manual_seed(11)
    factored = make_model(backend="factored")
    materialized = make_model(backend="materialized")
    materialized.load_state_dict(copy.deepcopy(factored.state_dict()))
    factored_inputs = torch.randn(
        2, factored.in_channels, 7, 9, dtype=torch.float64, requires_grad=True
    )
    materialized_inputs = factored_inputs.detach().clone().requires_grad_(True)

    factored_outputs = factored(factored_inputs)
    materialized_outputs = materialized(materialized_inputs)
    factored_outputs.square().mean().backward()
    materialized_outputs.square().mean().backward()

    torch.testing.assert_close(factored_outputs, materialized_outputs)
    torch.testing.assert_close(factored_inputs.grad, materialized_inputs.grad)
    torch.testing.assert_close(factored.atoms.p.grad, materialized.atoms.p.grad)


def test_grouped_materialized_convolution_matches_torch() -> None:
    model = make_model(
        backend="materialized",
        in_channels=4,
        out_channels=6,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=2,
    )
    inputs = torch.randn(2, 4, 8, 7, dtype=torch.float64)

    expected = F.conv2d(inputs, model.dense_weight(), padding=1, groups=2)

    torch.testing.assert_close(model(inputs), expected)


def test_auto_uses_materialized_execution_for_groups() -> None:
    model = make_model(
        backend="auto",
        in_channels=4,
        out_channels=4,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=2,
    )

    assert model._resolved_backend() == "materialized"
    with pytest.raises(ValueError, match="groups=1"):
        make_model(
            backend="factored",
            in_channels=4,
            out_channels=4,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=2,
        )


def test_constructor_and_forward_validate_flat_patch_shape() -> None:
    with pytest.raises(ValueError, match="flattened local patch"):
        CSTConv2d(
            Chart.linspace(17, low=-1.0, high=1.0),
            Chart.linspace(4, low=-1.0, high=1.0),
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            atoms=2,
            kernel=make_kernel(),
        )

    model = make_model()
    with pytest.raises(ValueError, match="expected input shape"):
        model(torch.randn(2, 3, 7, 9, dtype=torch.float64))


def test_parameter_adam_owns_factored_conv_atom_coordinates() -> None:
    model = make_model(backend="factored")
    optimizer = CSTParameterAdam(model)
    before = model.atoms.p.detach().clone()

    optimizer.zero_grad()
    model(torch.randn(2, 2, 7, 9, dtype=torch.float64)).square().mean().backward()
    optimizer.step()

    assert model.atoms.p.grad is not None
    assert not torch.equal(model.atoms.p, before)
    assert optimizer.param_groups[0]["schedule_step"] == 1


def test_parameter_adam_accepts_auto_when_conv_resolves_to_factored() -> None:
    model = make_model(backend="auto")

    assert model._resolved_backend() == "factored"
    CSTParameterAdam(model)


def test_nd_optimizer_rejects_conv_instead_of_treating_atoms_as_dense() -> None:
    class MixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = CSTLinear(
                Chart.linspace(4, spacing=1.0),
                Chart.linspace(2, spacing=1.0),
                atoms=1,
                kernel=make_kernel(),
            )
            self.conv = make_model()

    model = MixedModel()
    with pytest.raises(ValueError, match=r"unsupported CST sites: conv \(CSTConv2d\)"):
        CSTQuadraticAdam(model, dense=AdamWConfig())
    assert model.linear.atoms.grad is None


def test_repulsion_energy_is_shape_invariant() -> None:
    model = make_model()
    atoms = model.materialized_atoms()
    normalized = atoms / torch.linalg.vector_norm(
        atoms, dim=(1, 2, 3, 4), keepdim=True
    ).clamp_min(torch.finfo(atoms.dtype).tiny)
    expected = normalized.sum(dim=0).square().sum() - normalized.square().sum()

    torch.testing.assert_close(model.repulsion_energy(), expected)
