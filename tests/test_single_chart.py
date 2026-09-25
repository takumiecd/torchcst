import copy
import inspect

import pytest
import torch

from torchcst import (
    Chart,
    CSTLinear,
    CSTNormalizedSGD,
    CSTParameterAdam,
    DirectAmpWidth,
    ExplicitChart,
    GridPattern,
    LinePattern,
    PointsPattern,
    ProductChart,
    SphereGeometry,
    StripChart,
    Triweight,
)


def direct_kernel(
    sigma, *, sigma_min=None, sigma_max=None, site_chunk=2048, atom_chunk=64
):
    minimum = sigma if sigma_min is None else sigma_min
    maximum = sigma if sigma_max is None else sigma_max
    return DirectAmpWidth(
        amplitude_max=1.0,
        sigma_min=minimum,
        sigma_birth=sigma,
        sigma_max=maximum,
        w_c=0.05,
        kappa=3.0,
        profile=Triweight(minimum, normalize_columns=False),
        site_chunk=site_chunk,
        atom_chunk=atom_chunk,
    )


def test_chart_contract_and_shape_axes_product():
    assert inspect.isabstract(Chart)
    assert isinstance(Chart.grid((2, 2), spacing=1.0), ExplicitChart)
    axes = (
        LinePattern(2, spacing=0.5),
        GridPattern((2, 3), spacing=(0.2, 0.3)),
        LinePattern(3, spacing=0.4),
    )
    chart = ProductChart(shape=(2, 6, 3), axes=axes)
    indices = torch.tensor([0, 1, 17, 18, 35])
    expected = torch.cat(
        (
            axes[0].positions(indices // 18),
            axes[1].positions(indices % 18 // 3),
            axes[2].positions(indices % 3),
        ),
        dim=-1,
    )
    torch.testing.assert_close(chart.positions(indices), expected)
    assert chart.shape == (2, 6, 3)
    assert chart.embedding_dim == 4
    assert sum(buffer.numel() for buffer in chart.buffers()) < chart.features
    kernel = direct_kernel(1.0)
    atoms = kernel.initialize(chart, 2, mode="balanced")
    assert kernel.weight(chart, atoms).shape == chart.shape


def test_product_rejects_missing_or_mismatched_axis_patterns():
    with pytest.raises(ValueError, match="one SitePattern"):
        ProductChart(shape=(2, 3, 4), axes=(LinePattern(2, spacing=1),))
    with pytest.raises(ValueError, match="match its shape"):
        ProductChart(
            shape=(2, 3),
            axes=(LinePattern(2, spacing=1), LinePattern(4, spacing=1)),
        )


def test_product_accepts_irregular_axis_and_bounded_initialization():
    chart = ProductChart(
        shape=(2, 4),
        axes=(
            PointsPattern(torch.tensor([[1.0, 2.0], [3.0, 5.0]])),
            LinePattern(4, low=-2.0, high=4.0),
        ),
    )
    centers = chart.initialize_centers(100, mode="uniform")
    assert centers.shape == (100, 3)
    assert bool(
        (
            (centers >= torch.tensor([1.0, 2.0, -2.0]))
            & (centers <= torch.tensor([3.0, 5.0, 4.0]))
        ).all()
    )


def test_spherical_product_lifts_sites_and_retracts_centers():
    chart = ProductChart(
        shape=(3, 4),
        axes=(LinePattern(3, spacing=0.5), LinePattern(4, spacing=0.4)),
        geometry=SphereGeometry(2, radius=8.0),
    )
    sites = chart.positions(torch.arange(chart.features))
    chart.geometry.validate_points(sites)
    assert sites.shape == (12, 3)
    model = CSTLinear(chart=chart, atoms=2, kernel=direct_kernel(2.5))
    loss = model(torch.randn(3, 4)).square().mean()
    loss.backward()
    CSTParameterAdam(model).step()
    chart.geometry.validate_centers(model.atoms.p[:, 2:])


def test_large_product_retains_only_axis_state():
    chart = ProductChart(
        shape=(100_000, 100_000),
        axes=(LinePattern(100_000, spacing=0.01), LinePattern(100_000, spacing=0.01)),
    )
    model = CSTLinear(chart=chart, atoms=3, kernel=direct_kernel(0.5))
    assert model.atoms.p.shape == (3, 4)
    assert sum(value.numel() for value in chart.buffers()) < 20


def test_strip_line_axis_tiles_without_extra_dimension_or_site_table():
    axes = (
        LinePattern(5, spacing=0.5),
        GridPattern((2, 3), spacing=0.2),
        LinePattern(3, spacing=0.1),
    )
    chart = StripChart(
        shape=(5, 6, 3),
        axes=axes,
        tile_shape=(2, 6, 3),
        axis=0,
        tile_pitch=2.0,
    )
    assert chart.embedding_dim == 4
    assert chart.tile_grid == (3, 1, 1)
    sites = chart.positions(torch.tensor([0, 36, 72]))
    torch.testing.assert_close(sites[:, 0], torch.tensor([-1.0, 1.0, 3.0]))
    torch.testing.assert_close(sites[:, 1:], sites[0, 1:].expand(3, -1))
    logical = torch.cat([chart.tile_indices(s)[0] for s in range(chart.tile_count)])
    torch.testing.assert_close(torch.sort(logical).values, torch.arange(chart.features))
    chart.validate_support(1.7)
    with pytest.raises(ValueError, match="more than two"):
        chart.validate_support(1.75)


def test_strip_rejects_multidimensional_extension_axis_and_second_tiled_axis():
    with pytest.raises(TypeError, match="one-dimensional LinePattern"):
        StripChart(
            shape=(6, 3),
            axes=(GridPattern((2, 3), spacing=1), LinePattern(3, spacing=1)),
            tile_shape=(2, 3),
            axis=0,
            tile_pitch=3,
        )
    with pytest.raises(ValueError, match="only the selected"):
        StripChart(
            shape=(6, 3),
            axes=(LinePattern(6, spacing=1), LinePattern(3, spacing=1)),
            tile_shape=(2, 2),
            axis=0,
            tile_pitch=3,
        )
    with pytest.raises(TypeError):
        StripChart(shape=(6, 3), tile_shape=(2, 3), tile_pitch=3)


def test_spherical_strip_lifts_sites_and_checks_compact_support():
    chart = StripChart(
        shape=(6, 3),
        tile_shape=(2, 3),
        axes=(LinePattern(6, spacing=0.2), LinePattern(3, spacing=0.1)),
        axis=0,
        tile_pitch=3.0,
        geometry=SphereGeometry(2, radius=32.0),
    )
    chart.geometry.validate_points(chart.positions(torch.arange(chart.features)))
    chart.validate_support(2.0)
    with pytest.raises(ValueError, match="more than two"):
        chart.validate_support(3.0)


def test_new_strip_packed_weight_matches_dense_and_zero_pads_edge():
    chart = StripChart(
        shape=(5, 3),
        tile_shape=(2, 3),
        axes=(LinePattern(5, spacing=0.2), LinePattern(3, spacing=0.3)),
        axis=0,
        tile_pitch=2.0,
    )
    model = CSTLinear(chart=chart, atoms=3, kernel=direct_kernel(1.0))
    packed = model.packed_weight()
    dense = model.dense_weight().flatten()
    assert packed.is_contiguous()
    assert packed.shape == (3, 2, 3)
    for station in range(chart.tile_count):
        logical, local = chart.tile_indices(station)
        torch.testing.assert_close(packed[station].flatten()[local], dense[logical])
        assert (
            torch.count_nonzero(packed[station].flatten().index_fill(0, local, 0)) == 0
        )
    packed.sum().backward()
    assert torch.isfinite(model.atoms.p.grad).all()


def test_single_chart_weight_gradient_matches_dense_oracle():
    chart = ProductChart(
        shape=(3, 4),
        axes=(LinePattern(3, spacing=0.8), GridPattern((2, 2), spacing=0.4)),
    )
    kernel = direct_kernel(1.7, site_chunk=3, atom_chunk=2)
    model = CSTLinear(chart=chart, atoms=3, kernel=kernel, dtype=torch.float64)
    oracle_p = model.atoms.p.detach().clone().requires_grad_()
    sites = chart.positions(torch.arange(chart.features))
    squared = (sites[:, None, :] - oracle_p[None, :, 2:]).square().sum(-1)
    radial = (1 - squared / kernel.sigma_min.square()).clamp_min(0).pow(3)
    oracle_weight = (radial * oracle_p[:, 0]).sum(-1).reshape(chart.shape)
    inputs = torch.randn(5, 4, dtype=torch.float64)
    expected = torch.nn.functional.linear(inputs, oracle_weight)
    actual = model(inputs)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(model.atoms.p.grad, oracle_p.grad)


def test_bandwidth_activity_and_strip_support_maximum():
    chart = ProductChart(
        shape=(2, 3), axes=(LinePattern(2, spacing=0.5), LinePattern(3, spacing=0.5))
    )
    kernel = direct_kernel(0.8, sigma_min=0.4, sigma_max=1.2, site_chunk=2)
    model = CSTLinear(chart=chart, atoms=2, kernel=kernel, dtype=torch.float64)
    with torch.no_grad():
        model.atoms.p[:, 1] = torch.tensor([1.0, 2.5], dtype=torch.float64)
    p = model.atoms.p.detach().clone().requires_grad_()
    amplitude, alpha = kernel._amplitude_and_alpha(p[:, :2])
    widths, _, _ = kernel._sigma_bounds(amplitude, alpha)
    squared = chart.squared_distance(p[:, 2:])
    expected = (
        (1 - squared / widths.square().detach()).clamp_min(0).pow(3) * amplitude
    ).sum(-1)
    torch.testing.assert_close(model.dense_weight().flatten(), expected)
    model.dense_weight().sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(model.atoms.p.grad, p.grad)
    moved = kernel.apply_parameter_update(
        chart,
        model.atoms.p.detach(),
        torch.full_like(model.atoms.p, 10.0),
        step_size=1.0,
    )
    assert bool((moved[:, 1] <= 4).all())
    CSTParameterAdam(model).step()
    assert bool((model.atoms.p[:, 1] >= 1).all())
    assert bool((model.atoms.p[:, 1] <= 4).all())
    strip = StripChart(
        shape=(6, 3),
        tile_shape=(2, 3),
        axes=(LinePattern(6, spacing=0.1), LinePattern(3, spacing=0.1)),
        axis=0,
        tile_pitch=1.0,
    )
    with pytest.raises(ValueError, match="more than two"):
        CSTLinear(chart=strip, atoms=2, kernel=kernel)


def test_single_chart_requests_bounded_site_slices(monkeypatch):
    chart = ProductChart(
        shape=(5, 7), axes=(LinePattern(5, spacing=0.2), LinePattern(7, spacing=0.3))
    )
    kernel = direct_kernel(0.8, site_chunk=4, atom_chunk=2)
    model = CSTLinear(chart=chart, atoms=5, kernel=kernel)
    original = chart.squared_distance
    selections = []

    def observed(centers, selection=None):
        assert isinstance(selection, slice)
        selections.append(selection)
        assert selection.stop - selection.start <= 4
        assert centers.shape[0] <= 2
        return original(centers, selection)

    monkeypatch.setattr(chart, "squared_distance", observed)
    monkeypatch.setattr(
        kernel,
        "materialize_atoms",
        lambda *args: (_ for _ in ()).throw(AssertionError()),
    )
    model(torch.randn(2, 7)).sum().backward()
    assert len(selections) >= 9


def test_strip_optimizer_and_checkpoint_round_trip():
    def make():
        chart = StripChart(
            shape=(5, 4),
            tile_shape=(2, 4),
            axes=(LinePattern(5, spacing=0.2), LinePattern(4, spacing=0.3)),
            axis=0,
            tile_pitch=2.0,
        )
        return CSTLinear(
            chart=chart, atoms=4, kernel=direct_kernel(1.0), dtype=torch.float64
        )

    model = make()
    optimizer = CSTParameterAdam(model)
    before = model.atoms.p.detach().clone()
    model(torch.randn(6, 4, dtype=torch.float64)).square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, model.atoms.p)
    restored = make()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(restored.atoms.p, model.atoms.p)
    torch.testing.assert_close(restored.dense_weight(), model.dense_weight())


def test_single_chart_normalized_optimizer_step():
    chart = ProductChart(
        shape=(2, 2), axes=(LinePattern(2, spacing=1), LinePattern(2, spacing=1))
    )
    model = CSTLinear(
        chart=chart, atoms=1, kernel=direct_kernel(2), dtype=torch.float64
    )
    optimizer = CSTNormalizedSGD(model, lr=0.01, trust_radius=0.2)
    before = model.atoms.p.detach().clone()
    optimizer.zero_grad()
    model(torch.tensor([[1.0, -0.5]], dtype=torch.float64)).square().mean().backward()
    optimizer.step()
    assert optimizer.last_step is not None
    assert not torch.equal(before, model.atoms.p)


def test_checkpoint_rejects_different_grid_shape_with_same_site_count():
    def make(shape):
        chart = ProductChart(
            shape=(2, 12),
            axes=(LinePattern(2, spacing=1), GridPattern(shape, spacing=1)),
        )
        return CSTLinear(chart=chart, atoms=2, kernel=direct_kernel(1.0))

    saved = make((2, 6)).state_dict()
    with pytest.raises(RuntimeError, match="site pattern checkpoint contract"):
        make((3, 4)).load_state_dict(saved)
