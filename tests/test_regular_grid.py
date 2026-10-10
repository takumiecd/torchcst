"""Grouped implicit grid layout and geometry-specific profile-product contracts."""

import copy
import itertools
from dataclasses import replace

import pytest
import torch

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    CSTOptimizer,
    RegularGridChartSpec,
    TriweightSpec,
    compile_chart,
    presets,
)
from torchcst import (
    chart_presets as charts,
)
from torchcst import (
    geometry_presets as geometries,
)
from torchcst._backends.torch.charts import execution as chart_ops


def kernel(dim, periodic, sigma=0.8, floor=1e-6):
    from torchcst import NormalizationSpec

    factory = (
        presets.polar_periodic_profile_product
        if periodic
        else presets.polar_profile_product
    )
    return factory(
        profiles=(TriweightSpec(),) * dim,
        amplitude_max=2.0,
        bounds=BandwidthBounds(
            minimum=sigma, birth=sigma, maximum=sigma, upper_floor=sigma
        ),
        w_c=0.5,
        normalization=NormalizationSpec(
            kind="discrete_l2", domain="operator_sites", floor=floor
        ),
    )


@pytest.mark.parametrize("periodic", [False, True])
def test_grouping_broadcast_indexing_and_bounded_storage(periodic):
    spec = charts.regular_grid(
        ((2, 3), (4,)),
        spacing=(0.5, 1.0, 2.0),
        origin=-0.5,
        geometry="flat_torus" if periodic else "euclidean",
    )
    assert isinstance(spec, RegularGridChartSpec)
    assert spec.shape == (6, 4)
    assert spec.coordinate_shape == (2, 3, 4)
    assert spec.origin == (-0.5,) * 3
    state = compile_chart(spec, dtype=torch.float64)
    sites = torch.tensor(
        list(itertools.product((-0.5, 0.0), (-0.5, 0.5, 1.5), (-0.5, 1.5, 3.5, 5.5)))
    )
    ids = torch.tensor([0, 1, 4, 11, 12, 23, 4])
    torch.testing.assert_close(chart_ops.positions(state, ids), sites[ids].double())
    assert sum(v.numel() for v in state.buffers()) == (9 if periodic else 6)
    large = compile_chart(charts.regular_grid(((10000, 20000), (30000,)), spacing=0.5))
    assert sum(v.numel() for v in large.buffers()) == 6
    assert chart_ops.positions(state, torch.empty(0, dtype=torch.long)).shape == (0, 3)
    if periodic:
        assert spec.geometry.periods == (1.0, 3.0, 8.0)
    assert charts.regular_grid(((256,), (16, 8))).shape == (256, 128)
    assert charts.regular_grid(((2,), (3,), (4,))).shape == (2, 3, 4)
    assert charts.regular_grid(((2,),)).shape == (2,)
    for invalid in (torch.tensor([-1]), torch.tensor([24])):
        with pytest.raises(IndexError):
            chart_ops.positions(state, invalid)
    with pytest.raises(ValueError):
        chart_ops.positions(state, torch.tensor([0.0]))


@pytest.mark.parametrize(
    "settings",
    [
        {"grid_shape": ()},
        {"grid_shape": (3, 4)},
        {"grid_shape": ((3,), ())},
        {"grid_shape": ((True,), (4,))},
        {"grid_shape": ((0,), (4,))},
        {"spacing": 0},
        {"spacing": -1},
        {"spacing": True},
        {"spacing": float("inf")},
        {"spacing": (1,)},
        {"spacing": (1, float("nan"))},
        {"spacing": (1, True)},
        {"origin": (0,)},
        {"origin": float("inf")},
        {"origin": True},
        {"geometry": geometries.euclidean(3)},
        {"geometry": geometries.sphere(2)},
        {"geometry": "unknown"},
    ],
)
def test_invalid_declarations(settings):
    values = {"grid_shape": ((3,), (4,))}
    values.update(settings)
    with pytest.raises((ValueError, TypeError)):
        charts.regular_grid(**values)


def test_scalar_spacing_is_equivalent_to_per_coordinate_values():
    for geometry in (None, "flat_torus"):
        assert charts.regular_grid(
            ((16, 16), (128,)), spacing=0.1, geometry=geometry
        ) == charts.regular_grid(
            ((16, 16), (128,)), spacing=(0.1, 0.1, 0.1), geometry=geometry
        )


def test_integer_origin_uses_floating_state_and_large_index_uniform_bounds():
    state = compile_chart(charts.regular_grid(((3,), (4,)), origin=0))
    assert state.origin.is_floating_point()
    assert state.spacing.is_floating_point()
    large = compile_chart(
        charts.regular_grid(((100000,), (3,)), spacing=0.0002), dtype=torch.float16
    )
    indices = torch.tensor([99999 * 3, 99999 * 3 + 2])
    sites = chart_ops.positions(large, indices)
    assert torch.isfinite(sites).all()
    centers = chart_ops.initialize_centers(large, 8, mode="uniform")
    assert torch.isfinite(centers).all()
    wide = compile_chart(
        charts.regular_grid(((5,), (1,)), spacing=30000, origin=(-60000, 0)),
        dtype=torch.float16,
    )
    assert torch.isfinite(chart_ops.initialize_centers(wide, 8, mode="uniform")).all()


@pytest.mark.parametrize("periodic", [False, True])
def test_live_layout_buffers_and_explicit_geometry(periodic):
    geometry = (
        geometries.flat_torus((1.3, 2.1, 3.2)) if periodic else geometries.euclidean(3)
    )
    spec = charts.regular_grid(
        ((2, 3), (4,)), spacing=(0.2, 0.3, 0.4), geometry=geometry
    )
    p = torch.tensor([[0.2, 1.2, 0.03, 0.11, 0.17]], dtype=torch.float64)
    model = CSTLinear(
        chart=spec,
        atoms=p,
        kernel=kernel(3, periodic),
        dtype=p.dtype,
        backend="factored",
    )
    with torch.no_grad():
        model.chart.origin.add_(0.07)
        model.chart.spacing.mul_(1.1)
        if periodic:
            model.chart.geometry.periods.mul_(1.2)
    live = model.chart.declaration()
    expected = oracle(live, model.atoms.p, float(model.kernel.sigma_min_input), 1e-6)
    torch.testing.assert_close(
        model.materialized_atoms(), expected, atol=2e-12, rtol=2e-11
    )
    assert live.spacing == tuple(model.chart.spacing.tolist())
    if periodic:
        assert live.geometry.periods != tuple(
            n * h for n, h in zip(live.coordinate_shape, live.spacing)
        )
    clone = CSTLinear(
        chart=spec,
        atoms=p,
        kernel=kernel(3, periodic),
        dtype=p.dtype,
        backend="factored",
    )
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    x = torch.randn(2, spec.shape[1], dtype=p.dtype)
    torch.testing.assert_close(clone(x), model(x))


def test_new_periodic_grid_preserves_existing_profile_values_and_gradients():
    spacing = (0.25, 0.5, 0.25)
    new_chart = charts.regular_grid(
        ((2, 3), (4,)), spacing=spacing, origin=-0.25, geometry="flat_torus"
    )
    old_chart = charts.periodic_grid(
        (2, 3, 4), output_dims=2, periods=(0.5, 1.5, 1.0), origin=(-0.25,) * 3
    )
    p = torch.tensor([[0.2, 1.2, 0.03, 0.11, 0.17]], dtype=torch.float64)
    new = CSTLinear(
        chart=new_chart,
        atoms=p,
        kernel=kernel(3, True),
        dtype=p.dtype,
        backend="factored",
    )
    old = CSTLinear(
        chart=old_chart,
        atoms=p,
        kernel=kernel(3, True),
        dtype=p.dtype,
        backend="factored",
    )
    x = torch.randn(2, 4, dtype=p.dtype)
    cotangent = torch.randn(2, 6, dtype=p.dtype)
    torch.testing.assert_close(new(x), old(x), atol=2e-12, rtol=2e-11)
    for model in (new, old):
        (model(x) * cotangent).sum().backward()
    torch.testing.assert_close(
        new.atoms.p.grad, old.atoms.p.grad, atol=2e-11, rtol=2e-10
    )


def oracle(spec, p, sigma, floor):
    """Whole-site product/norm without chart, profile, or factor helpers."""
    amp = 2 * p[:, 0] / p[:, :2].square().sum(-1).sqrt()
    raw = []
    for indices in itertools.product(*(range(n) for n in spec.coordinate_shape)):
        value = torch.ones_like(amp)
        for d, index in enumerate(indices):
            delta = spec.origin[d] + index * spec.spacing[d] - p[:, 2 + d]
            if spec.geometry.id == "flat_torus":
                period = spec.geometry.periods[d]
                residue = torch.remainder(delta, period)
                delta = torch.minimum(residue, period - residue)
            value = value * (1 - delta.square() / sigma**2).clamp_min(0).pow(3)
        raw.append(value)
    raw = torch.stack(raw)
    norm = torch.linalg.vector_norm(raw, dim=0).clamp_min(floor)
    return (raw * (amp / norm)).T.reshape(len(p), *spec.shape)


@pytest.mark.parametrize(
    "groups",
    [((3,), (4,)), ((2, 3), (4,)), ((3,), (2, 4)), ((2, 3), (2, 4)), ((1, 3), (1,))],
)
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("backend", ["factored", "materialized", "auto"])
@pytest.mark.parametrize(
    "sigma,floor", [(0.8, 1e-6), (0.02, 1e-6), (0.8, 10.0), (3.0, 1e-6)]
)
def test_independent_values_dx_and_all_atom_gradients(
    groups, periodic, backend, sigma, floor
):
    torch.manual_seed(17)
    dim = sum(map(len, groups))
    spacing = tuple(0.25 + 0.07 * d for d in range(dim))
    spec = charts.regular_grid(
        groups,
        spacing=spacing,
        origin=-0.15,
        geometry="flat_torus" if periodic else None,
    )
    p = torch.tensor(
        [[0.2, 1.2] + [0.03] * dim, [-0.3, 1.1] + [0.11] * dim], dtype=torch.float64
    )
    model = CSTLinear(
        chart=spec,
        atoms=p,
        kernel=kernel(dim, periodic, sigma, floor),
        backend=backend,
        dtype=p.dtype,
    )
    reference_p = p.clone().requires_grad_()
    # Existing KernelState builds fixed scalar buffers before dtype conversion.
    # Compare the same live fixed width, including its original storage rounding.
    effective_sigma = float(model.kernel.sigma_min_input)
    expected_atoms = oracle(spec, reference_p, effective_sigma, floor)
    torch.testing.assert_close(
        model.materialized_atoms(), expected_atoms, atol=2e-12, rtol=2e-11
    )
    x = torch.randn(2, 3, 2 * spec.shape[1], dtype=p.dtype)[..., ::2].requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    actual = model(x)
    expected = torch.nn.functional.linear(reference_x, expected_atoms.sum(0))
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-11)
    cotangent = torch.randn_like(actual)
    (actual * cotangent).sum().backward()
    (expected * cotangent).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=2e-11, rtol=2e-10)
    torch.testing.assert_close(
        model.atoms.p.grad, reference_p.grad, atol=2e-11, rtol=2e-10
    )


def test_geometry_profile_contracts_and_linear_rank():
    euclidean = charts.regular_grid(((3,), (4,)))
    periodic = charts.regular_grid(((3,), (4,)), geometry="flat_torus")
    for spec, wrong in ((euclidean, kernel(2, True)), (periodic, kernel(2, False))):
        with pytest.raises(ValueError):
            CSTLinear(chart=spec, atoms=2, kernel=wrong)
    with pytest.raises(ValueError):
        CSTLinear(chart=euclidean, atoms=2, kernel=kernel(3, False))
    with pytest.raises(ValueError):
        CSTLinear(
            chart=charts.regular_grid(((2,), (3,), (4,))),
            atoms=2,
            kernel=kernel(3, False),
        )
    with pytest.raises(ValueError):
        compile_chart(replace(euclidean, revision=2))


@pytest.mark.parametrize("periodic", [False, True])
def test_checkpoint_layout_mismatch_and_live_buffer_validation(periodic):
    spec = charts.regular_grid(
        ((2, 3), (4,)), spacing=0.5, geometry="flat_torus" if periodic else None
    )
    state = compile_chart(spec, dtype=torch.float64)
    saved = copy.deepcopy(state.state_dict())
    restored = compile_chart(spec, dtype=torch.float32).double()
    restored.load_state_dict(saved)
    assert restored.declaration() == state.declaration()
    with pytest.raises(RuntimeError):
        compile_chart(
            charts.regular_grid(
                ((3, 2), (4,)), geometry="flat_torus" if periodic else None
            )
        ).load_state_dict(saved)
    for name, bad in (("spacing", 0.0), ("origin", float("nan"))):
        broken = copy.deepcopy(saved)
        broken[name][0] = bad
        with pytest.raises(RuntimeError):
            restored.load_state_dict(broken)
    for settings in ({"spacing": 1e-20}, {"origin": 1e20}):
        with pytest.raises(ValueError):
            compile_chart(
                charts.regular_grid(((3,), (4,)), **settings), dtype=torch.float16
            )


def test_public_optimizer_wraps_periodic_centers_and_preserves_euclidean_centers():
    for periodic in (False, True):
        spec = charts.regular_grid(
            ((3,), (4,)), geometry="flat_torus" if periodic else None
        )
        p = torch.tensor([[0.2, 1.2, -0.3, 4.3]], dtype=torch.float64)
        model = CSTLinear(
            chart=spec,
            atoms=p,
            kernel=kernel(2, periodic),
            backend="factored",
            dtype=p.dtype,
        )
        optimizer = CSTOptimizer(
            torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0), model=model
        )
        model.atoms.p.grad = torch.zeros_like(model.atoms.p)
        optimizer.step()
        centers = model.atoms.p[:, 2:]
        if periodic:
            torch.testing.assert_close(
                centers, torch.tensor([[2.7, 0.3]], dtype=p.dtype)
            )
        else:
            torch.testing.assert_close(centers, p[:, 2:])
