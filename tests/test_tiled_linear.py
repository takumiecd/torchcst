"""The explicit tiled CSTLinear backend matches the canonical dense operator."""

import math

import pytest
import torch

from torchcst import (
    CSTLinear,
    CSTParameterAdam,
    DirectAmpWidth,
    Gaussian,
    GridPattern,
    LinePattern,
    ParameterAdamConfig,
    ProductChart,
    StripChart,
    TorusGeometry,
    Triweight,
)
from torchcst.optim import LinearJGAtomGrad


def _kernel(*, compact: bool = True) -> DirectAmpWidth:
    profile = (
        Triweight(10.0, normalize_columns=False)
        if compact else Gaussian(10.0, normalize_columns=False)
    )
    return DirectAmpWidth(
        amplitude_max=1.0,
        sigma_min=10.0,
        sigma_birth=10.0,
        sigma_max=10.0,
        w_c=0.05,
        profile=profile,
        checkpoint_blocks=False,
    )


def _chart() -> StripChart:
    return StripChart(
        shape=(16, 4),
        tile_shape=(4, 4),
        axes=(LinePattern(16, spacing=2.0), GridPattern((2, 2), spacing=0.2)),
        axis=0,
        tile_pitch=25.0,
        geometry=TorusGeometry(
            3,
            major_radius=100 / (2 * math.pi),
            minor_radius=1.0,
            representation="intrinsic",
        ),
    )


def test_tiled_linear_matches_dense_forward_and_backward() -> None:
    torch.manual_seed(2)
    dense = CSTLinear(chart=_chart(), atoms=9, kernel=_kernel(), dtype=torch.float64)
    tiled = CSTLinear(
        chart=_chart(), atoms=9, kernel=_kernel(), backend="tiled",
        dtype=torch.float64,
    )
    with torch.no_grad():
        point = torch.tensor([[75.5, 0.0, 0.0]], dtype=torch.float64)
        center = dense.chart.geometry.lift_chart_coordinates(point)
        dense.atoms.p[0, 2:] = dense.chart.geometry.encode_centers(center)[0]
        dense.atoms.p[0, 0] = 0.4
    tiled.load_state_dict(dense.state_dict())
    assert tiled._resolved_backend() == "tiled"

    x_dense = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    x_tiled = x_dense.detach().clone().requires_grad_()
    expected = dense(x_dense)
    actual = tiled(x_tiled)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(x_tiled.grad, x_dense.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        tiled.atoms.p.grad, dense.atoms.p.grad, atol=1e-10, rtol=1e-10
    )


def test_tiled_linear_survives_parameter_adam_update() -> None:
    model = CSTLinear(
        chart=_chart(), atoms=5, kernel=_kernel(), backend="tiled",
        dtype=torch.float64,
    )
    optimizer = CSTParameterAdam(
        model, cst=ParameterAdamConfig(lr=0.001, decay_steps=None)
    )
    inputs = torch.randn(3, 4, dtype=torch.float64)
    model(inputs).square().sum().backward()
    optimizer.step()
    torch.testing.assert_close(
        model(inputs), torch.nn.functional.linear(inputs, model.dense_weight()),
        atol=1e-12, rtol=1e-12,
    )


def test_tiled_linear_supports_the_custom_atom_gradient_route() -> None:
    model = CSTLinear(
        chart=_chart(), atoms=5, kernel=_kernel(), backend="tiled",
        dtype=torch.float64,
    )
    collector = LinearJGAtomGrad(mode="custom", factored=False)
    model.atoms.set_grad(collector)
    inputs = torch.randn(2, 4, dtype=torch.float64)
    output_gradient = torch.randn(2, 16, dtype=torch.float64)
    collector.begin()
    (model(inputs) * output_gradient).sum().backward()
    collector.complete()
    assert collector.last_route == "custom"
    oracle_p = model.atoms.p.detach().clone().requires_grad_()
    oracle_output = torch.nn.functional.linear(
        inputs, model.kernel.weight(model.chart, oracle_p)
    )
    expected = torch.autograd.grad(
        (oracle_output * output_gradient).sum(), oracle_p
    )[0]
    torch.testing.assert_close(collector.snapshot().jg, expected)


def test_weight_tile_matches_selected_dense_entries() -> None:
    model = CSTLinear(chart=_chart(), atoms=5, kernel=_kernel(), dtype=torch.float64)
    rows = torch.tensor([0, 2, 7])
    columns = torch.tensor([1, 3])
    actual = model.kernel.weight_tile(model.chart, model.atoms.p, rows, columns)
    expected = model.dense_weight()[rows[:, None], columns[None, :]]
    torch.testing.assert_close(actual, expected)


def test_tiled_backend_rejects_unsupported_charts_and_profiles() -> None:
    with pytest.raises(TypeError, match=r"StripChart \+ TorusGeometry"):
        CSTLinear(
            chart=ProductChart(
                shape=(2, 2),
                axes=(LinePattern(2, spacing=0.2), LinePattern(2, spacing=0.2)),
            ),
            atoms=2, kernel=_kernel(), backend="tiled",
        )
    with pytest.raises(ValueError, match="raw compact profile"):
        CSTLinear(chart=_chart(), atoms=2, kernel=_kernel(compact=False), backend="tiled")
