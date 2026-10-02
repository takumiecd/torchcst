from __future__ import annotations

import copy
import inspect

import pytest
import torch
from kernel_cases import direct_state, triweight_state

from torchcst import BandwidthBounds, CSTLinear, CSTOptimizer
from torchcst._backends.linear import resolve_backend
from torchcst._backends.torch.charts import construction as _construction
from torchcst._backends.torch.charts import execution as _charts
from torchcst._backends.torch.geometry import execution as _geometry
from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.kernels.execution import KernelOptions
from torchcst._backends.torch.patterns import execution as _patterns
from torchcst.charts import ChartState


def direct_kernel(
    sigma, *, sigma_min=None, sigma_max=None, site_chunk=2048, atom_chunk=64
):
    minimum = sigma if sigma_min is None else sigma_min
    maximum = sigma if sigma_max is None else sigma_max
    return direct_state(
        amplitude_max=1.0,
        w_c=0.05,
        kappa=3.0,
        profile=triweight_state(minimum, normalize_columns=False),
        input_bounds=BandwidthBounds(
            minimum=minimum, maximum=maximum, birth=sigma, upper_floor=minimum
        ),
        options=KernelOptions(site_chunk=site_chunk, atom_chunk=atom_chunk),
        composition="radial",
    )


def test_auto_keeps_the_exact_single_chart_backend() -> None:
    chart = _construction.product(
        shape=(2, 3),
        axes=(
            _construction.line_pattern(2, spacing=0.5),
            _construction.line_pattern(3, spacing=0.5),
        ),
    )
    model = CSTLinear(
        chart=chart, atoms=2, kernel=direct_kernel(1.0).declaration(), backend="auto"
    )
    assert resolve_backend(model) == "materialized"
    expected = model.dense_weight()
    x = torch.randn(4, model.in_features)
    torch.testing.assert_close(model(x), torch.nn.functional.linear(x, expected))


def test_chart_contract_and_shape_axes_product():
    assert inspect.isabstract(ChartState)
    assert not hasattr(ChartState, "positions")
    assert (
        isinstance(_construction.grid((2, 2), spacing=1.0), ChartState)
        and _construction.grid((2, 2), spacing=1.0).spec.kind == "explicit"
    )
    axes = (
        _construction.line_pattern(2, spacing=0.5),
        _construction.grid_pattern((2, 3), spacing=(0.2, 0.3)),
        _construction.line_pattern(3, spacing=0.4),
    )
    chart = _construction.product(shape=(2, 6, 3), axes=axes)
    indices = torch.tensor([0, 1, 17, 18, 35])
    expected = torch.cat(
        (
            _patterns.positions(axes[0], indices // 18),
            _patterns.positions(axes[1], indices % 18 // 3),
            _patterns.positions(axes[2], indices % 3),
        ),
        dim=-1,
    )
    torch.testing.assert_close(_charts.positions(chart, indices), expected)
    assert chart.shape == (2, 6, 3)
    assert chart.embedding_dim == 4
    assert sum(buffer.numel() for buffer in chart.buffers()) < chart.features
    kernel = direct_kernel(1.0)
    atoms = _kernel.initialize(kernel, chart, 2, mode="balanced")
    assert _kernel.weight(kernel, chart, atoms).shape == chart.shape


def test_product_rejects_missing_or_mismatched_axis_patterns():
    with pytest.raises(ValueError, match="one pattern per logical axis"):
        _construction.product(
            shape=(2, 3, 4), axes=(_construction.line_pattern(2, spacing=1),)
        )
    with pytest.raises(ValueError, match="match chart shape"):
        _construction.product(
            shape=(2, 3),
            axes=(
                _construction.line_pattern(2, spacing=1),
                _construction.line_pattern(4, spacing=1),
            ),
        )


def test_product_accepts_irregular_axis_and_bounded_initialization():
    chart = _construction.product(
        shape=(2, 4),
        axes=(
            _construction.points_pattern(torch.tensor([[1.0, 2.0], [3.0, 5.0]])),
            _construction.line_pattern(4, low=-2.0, high=4.0),
        ),
    )
    centers = _charts.initialize_centers(chart, 100, mode="uniform")
    assert centers.shape == (100, 3)
    assert bool(
        (
            (centers >= torch.tensor([1.0, 2.0, -2.0]))
            & (centers <= torch.tensor([3.0, 5.0, 4.0]))
        ).all()
    )


def test_spherical_product_lifts_sites_and_retracts_centers():
    chart = _construction.product(
        shape=(3, 4),
        axes=(
            _construction.line_pattern(3, spacing=0.5),
            _construction.line_pattern(4, spacing=0.4),
        ),
        geometry=_construction.sphere(2, radius=8.0),
    )
    sites = _charts.positions(chart, torch.arange(chart.features))
    _geometry.validate_points(chart.geometry, sites)
    assert sites.shape == (12, 3)
    model = CSTLinear(chart=chart, atoms=2, kernel=direct_kernel(2.5).declaration())
    loss = model(torch.randn(3, 4)).square().mean()
    loss.backward()
    CSTOptimizer(
        torch.optim.AdamW(
            model.parameters(),
            lr=0.03,
            betas=(0.5, 0.99),
            eps=1e-08,
            weight_decay=0.0,
            foreach=False,
        ),
        model=model,
    ).step()
    _geometry.validate_centers(chart.geometry, model.atoms.p[:, 2:])


def test_large_product_retains_only_axis_state():
    chart = _construction.product(
        shape=(100000, 100000),
        axes=(
            _construction.line_pattern(100000, spacing=0.01),
            _construction.line_pattern(100000, spacing=0.01),
        ),
    )
    model = CSTLinear(chart=chart, atoms=3, kernel=direct_kernel(0.5).declaration())
    assert model.atoms.p.shape == (3, 4)
    assert sum(value.numel() for value in chart.buffers()) < 20


def test_strip_line_axis_tiles_without_extra_dimension_or_site_table():
    axes = (
        _construction.line_pattern(5, spacing=0.5),
        _construction.grid_pattern((2, 3), spacing=0.2),
        _construction.line_pattern(3, spacing=0.1),
    )
    chart = _construction.strip(
        shape=(5, 6, 3), axes=axes, tile_shape=(2, 6, 3), axis=0, tile_pitch=2.0
    )
    assert chart.embedding_dim == 4
    assert chart.tile_grid == (3, 1, 1)
    sites = _charts.positions(chart, torch.tensor([0, 36, 72]))
    torch.testing.assert_close(sites[:, 0], torch.tensor([-1.0, 1.0, 3.0]))
    torch.testing.assert_close(sites[:, 1:], sites[0, 1:].expand(3, -1))
    logical = torch.cat(
        [_charts.tile_indices(chart, s)[0] for s in range(chart.tile_count)]
    )
    torch.testing.assert_close(torch.sort(logical).values, torch.arange(chart.features))
    _charts.validate_support(chart, 1.7)
    with pytest.raises(ValueError, match="more than two"):
        _charts.validate_support(chart, 1.75)


def test_strip_rejects_multidimensional_extension_axis_and_second_tiled_axis():
    with pytest.raises(ValueError, match="line pattern"):
        _construction.strip(
            shape=(6, 3),
            axes=(
                _construction.grid_pattern((2, 3), spacing=1),
                _construction.line_pattern(3, spacing=1),
            ),
            tile_shape=(2, 3),
            axis=0,
            tile_pitch=3,
        )
    with pytest.raises(ValueError, match="only the strip axis"):
        _construction.strip(
            shape=(6, 3),
            axes=(
                _construction.line_pattern(6, spacing=1),
                _construction.line_pattern(3, spacing=1),
            ),
            tile_shape=(2, 2),
            axis=0,
            tile_pitch=3,
        )
    with pytest.raises(TypeError):
        _construction.strip(shape=(6, 3), tile_shape=(2, 3), tile_pitch=3)


def test_spherical_strip_lifts_sites_and_checks_compact_support():
    chart = _construction.strip(
        shape=(6, 3),
        tile_shape=(2, 3),
        axes=(
            _construction.line_pattern(6, spacing=0.2),
            _construction.line_pattern(3, spacing=0.1),
        ),
        axis=0,
        tile_pitch=3.0,
        geometry=_construction.sphere(2, radius=32.0),
    )
    _geometry.validate_points(
        chart.geometry, _charts.positions(chart, torch.arange(chart.features))
    )
    _charts.validate_support(chart, 2.0)
    with pytest.raises(ValueError, match="more than two"):
        _charts.validate_support(chart, 3.0)


def test_new_strip_packed_weight_matches_dense_and_zero_pads_edge():
    chart = _construction.strip(
        shape=(5, 3),
        tile_shape=(2, 3),
        axes=(
            _construction.line_pattern(5, spacing=0.2),
            _construction.line_pattern(3, spacing=0.3),
        ),
        axis=0,
        tile_pitch=2.0,
    )
    model = CSTLinear(chart=chart, atoms=3, kernel=direct_kernel(1.0).declaration())
    packed = model.packed_weight()
    dense = model.dense_weight().flatten()
    assert packed.is_contiguous()
    assert packed.shape == (3, 2, 3)
    for station in range(chart.tile_count):
        logical, local = _charts.tile_indices(chart, station)
        torch.testing.assert_close(packed[station].flatten()[local], dense[logical])
        assert (
            torch.count_nonzero(packed[station].flatten().index_fill(0, local, 0)) == 0
        )
    packed.sum().backward()
    assert torch.isfinite(model.atoms.p.grad).all()


def test_single_chart_weight_gradient_matches_dense_oracle():
    chart = _construction.product(
        shape=(3, 4),
        axes=(
            _construction.line_pattern(3, spacing=0.8),
            _construction.grid_pattern((2, 2), spacing=0.4),
        ),
    )
    kernel = direct_kernel(1.7, site_chunk=3, atom_chunk=2)
    model = CSTLinear(
        chart=chart,
        atoms=3,
        kernel=kernel.declaration(),
        kernel_options=kernel.options,
        dtype=torch.float64,
    )
    oracle_p = model.atoms.p.detach().clone().requires_grad_()
    sites = _charts.positions(chart, torch.arange(chart.features))
    squared = (sites[:, None, :] - oracle_p[None, :, 2:]).square().sum(-1)
    radial = (
        (1 - squared / kernel.scalar("sigma_min_input").square()).clamp_min(0).pow(3)
    )
    oracle_weight = (radial * oracle_p[:, 0]).sum(-1).reshape(chart.shape)
    inputs = torch.randn(5, 4, dtype=torch.float64)
    expected = torch.nn.functional.linear(inputs, oracle_weight)
    actual = model(inputs)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(model.atoms.p.grad, oracle_p.grad)


def test_bandwidth_activity_and_strip_support_maximum():
    chart = _construction.product(
        shape=(2, 3),
        axes=(
            _construction.line_pattern(2, spacing=0.5),
            _construction.line_pattern(3, spacing=0.5),
        ),
    )
    kernel = direct_kernel(0.8, sigma_min=0.4, sigma_max=1.2, site_chunk=2)
    model = CSTLinear(
        chart=chart,
        atoms=2,
        kernel=kernel.declaration(),
        kernel_options=kernel.options,
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.atoms.p[:, 1] = torch.tensor([1.0, 2.5], dtype=torch.float64)
    p = model.atoms.p.detach().clone().requires_grad_()
    amplitude, alpha = _kernel.coordinate(kernel, "_amplitude_and_alpha", p[:, :2])
    widths, _, _ = _kernel.coordinate(kernel, "_sigma_bounds", amplitude, alpha)
    squared = _charts.squared_distance(chart, p[:, 2:])
    expected = (
        (1 - squared / widths.square().detach()).clamp_min(0).pow(3) * amplitude
    ).sum(-1)
    torch.testing.assert_close(model.dense_weight().flatten(), expected)
    model.dense_weight().sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(model.atoms.p.grad, p.grad)
    moved = _kernel.apply_parameter_update(
        kernel,
        chart,
        model.atoms.p.detach(),
        torch.full_like(model.atoms.p, 10.0),
        step_size=1.0,
    )
    assert bool((moved[:, 1] <= 4).all())
    CSTOptimizer(
        torch.optim.AdamW(
            model.parameters(),
            lr=0.03,
            betas=(0.5, 0.99),
            eps=1e-08,
            weight_decay=0.0,
            foreach=False,
        ),
        model=model,
    ).step()
    assert bool((model.atoms.p[:, 1] >= 1).all())
    assert bool((model.atoms.p[:, 1] <= 4).all())
    strip = _construction.strip(
        shape=(6, 3),
        tile_shape=(2, 3),
        axes=(
            _construction.line_pattern(6, spacing=0.1),
            _construction.line_pattern(3, spacing=0.1),
        ),
        axis=0,
        tile_pitch=1.0,
    )
    with pytest.raises(ValueError, match="more than two"):
        CSTLinear(
            chart=strip,
            atoms=2,
            kernel=kernel.declaration(),
            kernel_options=kernel.options,
        )


def test_single_chart_requests_bounded_site_slices(monkeypatch):
    chart = _construction.product(
        shape=(5, 7),
        axes=(
            _construction.line_pattern(5, spacing=0.2),
            _construction.line_pattern(7, spacing=0.3),
        ),
    )
    kernel = direct_kernel(0.8, site_chunk=4, atom_chunk=2)
    model = CSTLinear(
        chart=chart, atoms=5, kernel=kernel.declaration(), kernel_options=kernel.options
    )
    original = _charts.squared_distance
    selections = []

    def observed(state, centers, selection=None):
        assert isinstance(selection, slice)
        selections.append(selection)
        assert selection.stop - selection.start <= 4
        assert centers.shape[0] <= 2
        return original(state, centers, selection)

    monkeypatch.setattr(_charts, "squared_distance", observed)
    monkeypatch.setattr(
        _kernel,
        "materialize_atoms",
        lambda *args: (_ for _ in ()).throw(AssertionError()),
    )
    model(torch.randn(2, 7)).sum().backward()
    assert len(selections) >= 9


def test_strip_optimizer_and_checkpoint_round_trip():

    def make():
        chart = _construction.strip(
            shape=(5, 4),
            tile_shape=(2, 4),
            axes=(
                _construction.line_pattern(5, spacing=0.2),
                _construction.line_pattern(4, spacing=0.3),
            ),
            axis=0,
            tile_pitch=2.0,
        )
        return CSTLinear(
            chart=chart,
            atoms=4,
            kernel=direct_kernel(1.0).declaration(),
            dtype=torch.float64,
        )

    model = make()
    optimizer = CSTOptimizer(
        torch.optim.AdamW(
            model.parameters(),
            lr=0.03,
            betas=(0.5, 0.99),
            eps=1e-08,
            weight_decay=0.0,
            foreach=False,
        ),
        model=model,
    )
    before = model.atoms.p.detach().clone()
    model(torch.randn(6, 4, dtype=torch.float64)).square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, model.atoms.p)
    restored = make()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(restored.atoms.p, model.atoms.p)
    torch.testing.assert_close(restored.dense_weight(), model.dense_weight())


def test_checkpoint_rejects_different_grid_shape_with_same_site_count():

    def make(shape):
        chart = _construction.product(
            shape=(2, 12),
            axes=(
                _construction.line_pattern(2, spacing=1),
                _construction.grid_pattern(shape, spacing=1),
            ),
        )
        return CSTLinear(chart=chart, atoms=2, kernel=direct_kernel(1.0).declaration())

    saved = make((2, 6)).state_dict()
    with pytest.raises(RuntimeError, match="site pattern checkpoint contract"):
        make((3, 4)).load_state_dict(saved)
