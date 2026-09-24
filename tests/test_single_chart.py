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
    sigma: float,
    *,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    site_chunk: int = 2048,
    atom_chunk: int = 64,
) -> DirectAmpWidth:
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


def test_chart_is_an_abstract_contract_with_explicit_factory_compatibility():
    assert inspect.isabstract(Chart)
    assert isinstance(Chart.grid((2, 2), spacing=1.0), ExplicitChart)


def test_product_chart_matches_explicit_cartesian_sites_without_storing_them():
    output = LinePattern(3, spacing=2.0)
    inputs = GridPattern((2, 3), spacing=(0.25, 0.5))
    chart = ProductChart(output, inputs)
    expected = torch.cat(
        (
            output.positions(torch.arange(3)).repeat_interleave(6, dim=0),
            inputs.positions(torch.arange(6)).repeat(3, 1),
        ),
        dim=1,
    )
    torch.testing.assert_close(chart.positions(torch.arange(18)), expected)
    assert chart.shape == (3, 6)
    assert not hasattr(chart, "coordinates")
    assert all(value.shape != expected.shape for value in chart.buffers())


def test_product_chart_accepts_irregular_patterns_and_exact_bounds():
    chart = ProductChart(
        PointsPattern(torch.tensor([[1.0, 2.0], [3.0, 5.0]])),
        LinePattern(4, low=-2.0, high=4.0),
    )
    centers = chart.initialize_centers(100, mode="uniform")
    assert centers.shape == (100, 3)
    assert bool(
        (
            (centers >= torch.tensor([1.0, 2.0, -2.0]))
            & (centers <= torch.tensor([3.0, 5.0, 4.0]))
        ).all()
    )


def test_product_chart_delegates_distance_and_center_updates_to_geometry():
    output = PointsPattern(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    inputs = PointsPattern(torch.zeros(2, 1))
    chart = ProductChart(output, inputs, geometry=SphereGeometry(2))
    model = CSTLinear(chart=chart, atoms=2, kernel=direct_kernel(2.5))
    chart.geometry.validate_centers(model.atoms.p[:, 2:])
    loss = model(torch.randn(3, 2)).square().mean()
    loss.backward()
    optimizer = CSTParameterAdam(model)
    optimizer.step()
    chart.geometry.validate_centers(model.atoms.p[:, 2:])


def test_large_product_chart_initialization_keeps_only_axis_state():
    chart = ProductChart(
        LinePattern(100_000, spacing=0.01), LinePattern(100_000, spacing=0.01)
    )
    model = CSTLinear(chart=chart, atoms=3, kernel=direct_kernel(0.5))
    assert model.atoms.p.shape == (3, 4)
    assert sum(value.numel() for value in chart.buffers()) < 20


def test_strip_chart_snake_order_and_local_geometry():
    chart = StripChart(
        (3, 4),
        (2, 2),
        tile_pitch=5.0,
        seam_gap=3.0,
        local_output=LinePattern(2, spacing=0.5),
        local_input=LinePattern(2, spacing=0.25),
    )
    indices = torch.tensor([0, 2, 8, 10])
    coordinates = chart.positions(indices)
    torch.testing.assert_close(coordinates[:, 0], torch.tensor([0.0, 5.0, 18.0, 13.0]))
    torch.testing.assert_close(
        coordinates[:, 1:],
        torch.tensor(
            [[-0.25, -0.125], [-0.25, -0.125], [-0.25, -0.125], [-0.25, -0.125]]
        ),
    )
    chart.validate_support(4.99)
    with pytest.raises(ValueError, match="smaller than tile_pitch"):
        chart.validate_support(5.0)


@pytest.mark.parametrize("sweep", ["input", "output"])
@pytest.mark.parametrize("snake", [True, False])
def test_strip_tile_order_maps_every_logical_site_once(sweep, snake):
    chart = StripChart((3, 5), (2, 2), tile_pitch=3.0, sweep=sweep, snake=snake)
    all_logical = []
    for station in range(chart.tile_count):
        logical, local = chart.tile_indices(station)
        assert logical.numel() == local.numel()
        assert torch.unique(local).numel() == local.numel()
        all_logical.append(logical)
    torch.testing.assert_close(
        torch.sort(torch.cat(all_logical)).values, torch.arange(chart.features)
    )


def test_strip_separate_seam_requires_enough_physical_gap():
    chart = StripChart(
        (4, 4), (2, 2), tile_pitch=2.0, seam_gap=0.1, seam_policy="separate"
    )
    with pytest.raises(ValueError, match="separate seams"):
        CSTLinear(chart=chart, atoms=2, kernel=direct_kernel(1.1))
    chart = StripChart(
        (4, 4), (2, 2), tile_pitch=2.0, seam_gap=0.3, seam_policy="separate"
    )
    CSTLinear(chart=chart, atoms=2, kernel=direct_kernel(1.1))


def test_strip_packed_weight_is_physical_tile_order_with_zero_padded_edges():
    chart = StripChart((3, 5), (2, 2), tile_pitch=2.0)
    model = CSTLinear(chart=chart, atoms=3, kernel=direct_kernel(1.0))
    packed = model.packed_weight()
    dense = model.dense_weight().flatten()
    assert packed.is_contiguous()
    assert packed.shape == (6, 2, 2)
    for station in range(chart.tile_count):
        logical, local = chart.tile_indices(station)
        torch.testing.assert_close(packed[station].flatten()[local], dense[logical])
        assert (
            torch.count_nonzero(packed[station].flatten().index_fill(0, local, 0)) == 0
        )
    packed.sum().backward()
    assert torch.isfinite(model.atoms.p.grad).all()


def test_one_kernel_weight_and_gradient_match_dense_oracle():
    chart = ProductChart(LinePattern(3, spacing=0.8), GridPattern((2, 2), spacing=0.4))
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
    torch.testing.assert_close(model.dense_weight(), model.materialized_atoms().sum(0))
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(model.atoms.p.grad, oracle_p.grad)


def test_direct_bandwidth_uses_activity_state_and_respects_support_maximum():
    chart = ProductChart(LinePattern(2, spacing=0.5), LinePattern(3, spacing=0.5))
    kernel = direct_kernel(0.8, sigma_min=0.4, sigma_max=1.2, site_chunk=2)
    model = CSTLinear(chart=chart, atoms=2, kernel=kernel, dtype=torch.float64)
    assert model.atoms.p.shape == (2, 4)
    with torch.no_grad():
        model.atoms.p[:, 1] = torch.tensor([1.0, 2.5], dtype=torch.float64)
    p = model.atoms.p.detach().clone().requires_grad_()
    amplitude, alpha = kernel._amplitude_and_alpha(p[:, :2])
    widths, _, _ = kernel._sigma_bounds(amplitude, alpha)
    squared = chart.squared_distance(p[:, 2:])
    expected = ((1 - squared / widths.square().detach()).clamp_min(0).pow(3) * amplitude).sum(-1)
    torch.testing.assert_close(model.dense_weight().flatten(), expected)
    model.dense_weight().sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(model.atoms.p.grad, p.grad)
    moved = kernel.apply_parameter_update(
        chart, model.atoms.p.detach(), torch.full_like(model.atoms.p, 10.0), step_size=1.0
    )
    assert bool((moved[:, 1] <= 4).all())
    CSTParameterAdam(model).step()
    assert bool((model.atoms.p[:, 1] >= 1).all())
    assert bool((model.atoms.p[:, 1] <= 4).all())

    strip = StripChart((2, 3), (1, 2), tile_pitch=1.0)
    with pytest.raises(ValueError, match="smaller than tile_pitch"):
        CSTLinear(
            chart=strip, atoms=2,
            kernel=direct_kernel(0.8, sigma_min=0.4, sigma_max=1.2),
        )


def test_single_chart_forward_requests_only_bounded_site_slices(monkeypatch):
    chart = ProductChart(LinePattern(5, spacing=0.2), LinePattern(7, spacing=0.3))
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


def test_strip_linear_parameter_adam_and_checkpoint_round_trip():
    def make():
        chart = StripChart(
            (3, 4),
            (2, 2),
            tile_pitch=2.0,
            local_output=LinePattern(2, spacing=0.2),
            local_input=LinePattern(2, spacing=0.3),
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


def test_single_chart_runs_one_normalized_optimizer_step():
    chart = ProductChart(LinePattern(2, spacing=1), LinePattern(2, spacing=1))
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


def test_checkpoint_rejects_a_different_grid_shape_with_same_site_count():
    def make(shape):
        chart = ProductChart(LinePattern(2, spacing=1), GridPattern(shape, spacing=1))
        return CSTLinear(chart=chart, atoms=2, kernel=direct_kernel(1.0))

    saved = make((2, 6)).state_dict()
    with pytest.raises(RuntimeError, match="site pattern checkpoint contract"):
        make((3, 4)).load_state_dict(saved)


def test_strip_requires_kernel_support_smaller_than_tile_pitch():
    chart = StripChart((4, 4), (2, 2), tile_pitch=1.0)
    with pytest.raises(ValueError, match="smaller than tile_pitch"):
        CSTLinear(chart=chart, atoms=2, kernel=direct_kernel(1.0))
