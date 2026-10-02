from __future__ import annotations

from torchcst import BandwidthBounds, CSTOptimizer
from torchcst._backends.torch.charts import construction as _construction
from torchcst._backends.torch.geometry import execution as _geometry
from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.kernels.execution import KernelOptions

"The explicit tiled CSTLinear backend matches the canonical dense operator."
import math

import pytest
import torch
from kernel_cases import direct_state, gaussian_state, triweight_state

from torchcst import CSTLinear
from torchcst.charts import ChartState


def make_radial_state(*, compact: bool = True) -> direct_state:
    profile = (
        triweight_state(10.0, normalize_columns=False)
        if compact
        else gaussian_state(10.0, normalize_columns=False)
    )
    return direct_state(
        amplitude_max=1.0,
        w_c=0.05,
        profile=profile,
        input_bounds=BandwidthBounds(
            minimum=10.0, maximum=10.0, birth=10.0, upper_floor=10.0
        ),
        options=KernelOptions(checkpoint_blocks=False),
        composition="radial",
    )


def _chart() -> ChartState:
    return _construction.strip(
        shape=(16, 4),
        tile_shape=(4, 4),
        axes=(
            _construction.line_pattern(16, spacing=2.0),
            _construction.grid_pattern((2, 2), spacing=0.2),
        ),
        axis=0,
        tile_pitch=25.0,
        geometry=_construction.torus(
            3,
            major_radius=100 / (2 * math.pi),
            minor_radius=1.0,
            representation="intrinsic",
        ),
    )


def test_tiled_linear_matches_dense_forward_and_backward() -> None:
    torch.manual_seed(2)
    dense = CSTLinear(
        chart=_chart(),
        atoms=9,
        kernel=make_radial_state().declaration(),
        dtype=torch.float64,
    )
    tiled = CSTLinear(
        chart=_chart(),
        atoms=9,
        kernel=make_radial_state().declaration(),
        backend="tiled",
        dtype=torch.float64,
    )
    with torch.no_grad():
        point = torch.tensor([[75.5, 0.0, 0.0]], dtype=torch.float64)
        center = _geometry.lift_chart_coordinates(dense.chart.geometry, point)
        dense.atoms.p[0, 2:] = _geometry.encode_centers(dense.chart.geometry, center)[0]
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
        chart=_chart(),
        atoms=5,
        kernel=make_radial_state().declaration(),
        backend="tiled",
        dtype=torch.float64,
    )
    optimizer = CSTOptimizer(
        torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            betas=(0.5, 0.99),
            eps=1e-08,
            weight_decay=0.0,
            foreach=False,
        ),
        model=model,
    )
    inputs = torch.randn(3, 4, dtype=torch.float64)
    model(inputs).square().sum().backward()
    optimizer.step()
    torch.testing.assert_close(
        model(inputs),
        torch.nn.functional.linear(inputs, model.dense_weight()),
        atol=1e-12,
        rtol=1e-12,
    )


def test_weight_tile_matches_selected_dense_entries() -> None:
    model = CSTLinear(
        chart=_chart(),
        atoms=5,
        kernel=make_radial_state().declaration(),
        dtype=torch.float64,
    )
    rows = torch.tensor([0, 2, 7])
    columns = torch.tensor([1, 3])
    actual = _kernel.weight_tile(
        model.kernel, model.chart, model.atoms.p, rows, columns
    )
    expected = model.dense_weight()[rows[:, None], columns[None, :]]
    torch.testing.assert_close(actual, expected)


def test_tiled_backend_rejects_unsupported_charts_and_profiles() -> None:
    with pytest.raises(TypeError, match="StripChart \\+ TorusGeometry"):
        CSTLinear(
            chart=_construction.product(
                shape=(2, 2),
                axes=(
                    _construction.line_pattern(2, spacing=0.2),
                    _construction.line_pattern(2, spacing=0.2),
                ),
            ),
            atoms=2,
            kernel=make_radial_state().declaration(),
            backend="tiled",
        )
    with pytest.raises(ValueError, match="raw compact profile"):
        CSTLinear(
            chart=_chart(),
            atoms=2,
            kernel=make_radial_state(compact=False).declaration(),
            backend="tiled",
        )
