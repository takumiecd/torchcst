"""Flat periodic-grid contract against independently enumerated full-site atoms."""

import copy
import itertools
from dataclasses import replace

import pytest
import torch

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    CSTOptimizer,
    FlatTorusGeometrySpec,
    GaussianSpec,
    NormalizationSpec,
    TriweightSpec,
    compile_chart,
    presets,
)
from torchcst import (
    chart_presets as charts,
)
from torchcst import (
    pattern_presets as patterns,
)
from torchcst._backends.torch.charts import execution as chart_ops
from torchcst._backends.torch.geometry import execution as geometry_ops
from torchcst._backends.torch.geometry.flat_torus import wrap_delta
from torchcst._backends.torch.kernels import execution as kernel_ops


def make_kernel(dim, sigma=0.7, floor=1e-6, mixed=False, live_width=False):
    bounds = BandwidthBounds(
        minimum=sigma, birth=sigma, maximum=sigma, upper_floor=sigma
    )
    if live_width:
        bounds = BandwidthBounds(minimum=0.1, birth=0.45, maximum=1.2, upper_floor=0.2)
    return presets.polar_periodic_profile_product(
        profiles=tuple(
            GaussianSpec() if mixed and d % 2 else TriweightSpec() for d in range(dim)
        ),
        amplitude_max=2.0,
        bounds=bounds,
        w_c=0.5,
        alpha_init=0.3,
        radial_regularization=0.2,
        normalization=NormalizationSpec(
            kind="discrete_l2", domain="operator_sites", floor=floor
        ),
    )


def oracle_atoms(model, p):
    """Enumerate sites and normalize complete atoms, without backend helpers."""
    state, chart = model.kernel, model.chart
    q = p[:, :2].square().sum(-1)
    amp = state.amplitude_max * p[:, 0] / q.clamp_min(torch.finfo(p.dtype).tiny).sqrt()
    alpha = ((q - 1) / 3).clamp(0, 1)
    x = (amp / state.w_c).square()
    lo = state.sigma_min_input + (state.sigma_birth_input - state.sigma_min_input) / (
        1 + state.lower_kappa * x
    )
    hi = state.sigma_min_input + (
        state.sigma_max_input - state.sigma_min_input
    ) * state.kappa / (state.kappa + x.pow(state.upper_decay_power))
    hi = torch.maximum(torch.maximum(hi, state.upper_floor_input), lo)
    sigma = (
        torch.exp((1 - alpha) * lo.log() + alpha * hi.log())
        .clamp(min=lo, max=hi)
        .detach()
    )
    raw = []
    for indices in itertools.product(*(range(n) for n in chart.grid_shape)):
        value = torch.ones_like(amp)
        for d, i in enumerate(indices):
            period = chart.geometry.periods[d]
            site = chart.origin[d] + i * period / chart.grid_shape[d]
            # Independent shortest-distance form. Its sign branch gives the
            # same VJP away from ties; cut-locus VJP is tested explicitly below.
            residue = torch.remainder(site - p[:, 2 + d], period)
            distance = torch.minimum(residue, period - residue)
            u = distance.square() / sigma.square()
            if isinstance(state.spec.profiles[d].profile, GaussianSpec):
                value = value * torch.exp(-0.5 * u)
            else:
                value = value * (1 - u).clamp_min(0).pow(3)
        raw.append(value)
    raw = torch.stack(raw)
    norm = raw.square().sum(0).sqrt().clamp_min(state.spec.normalization.floor)
    return (raw * (amp / norm)[None, :]).T.reshape(p.shape[0], *chart.shape)


@pytest.mark.parametrize(
    "shape,split",
    [((5, 7), 1), ((3, 4, 5), 1), ((3, 4, 5), 2), ((2, 3, 4, 5), 2), ((1, 3, 1), 2)],
)
@pytest.mark.parametrize("backend", ["auto", "factored", "materialized"])
@pytest.mark.parametrize("floor,mixed,live", [(1e-6, False, False), (10.0, True, True)])
def test_full_site_atoms_values_dx_and_all_parameter_gradients(
    shape, split, backend, floor, mixed, live
):
    torch.manual_seed(27)
    dim = len(shape)
    periods = tuple(1 + 0.37 * d for d in range(dim))
    origins = tuple(-0.23 + 0.11 * d for d in range(dim))
    p = torch.tensor(
        [
            [0.15, 1.2] + [origins[d] + 0.039 - 2 * periods[d] for d in range(dim)],
            [-0.21, 1.35] + [origins[d] + periods[d] - 0.071 for d in range(dim)],
        ],
        dtype=torch.float64,
    )
    model = CSTLinear(
        chart=charts.periodic_grid(
            shape, periods=periods, origin=origins, output_dims=split
        ),
        atoms=p,
        kernel=make_kernel(dim, floor=floor, mixed=mixed, live_width=live),
        backend=backend,
    )
    point = p.detach().clone().requires_grad_()
    expected_atoms = oracle_atoms(model, point)
    torch.testing.assert_close(
        model.materialized_atoms(), expected_atoms, rtol=2e-11, atol=2e-12
    )
    # Include noncontiguous, multidimensional input batches.
    x = torch.randn(2, 3, 2 * model.in_features, dtype=p.dtype)[
        ..., ::2
    ].requires_grad_()
    rx = x.detach().clone().requires_grad_()
    actual = model(x)
    expected = torch.nn.functional.linear(rx, expected_atoms.sum(0))
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    cot = torch.randn_like(actual)
    (actual * cot).sum().backward()
    (expected * cot).sum().backward()
    torch.testing.assert_close(x.grad, rx.grad, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(model.atoms.p.grad, point.grad, rtol=2e-10, atol=2e-11)
    assert torch.count_nonzero(point.grad[:, :2]) == 4


def test_implicit_storage_positions_and_lattice_index_order():
    spec = charts.periodic_grid(
        (2, 3, 4), periods=(2.0, 3.0, 8.0), output_dims=2, origin=(-1.0, 0.5, 7.0)
    )
    state = compile_chart(spec, dtype=torch.float64)
    expected = torch.tensor(
        list(itertools.product((-1.0, 0.0), (0.5, 1.5, 2.5), (7.0, 9.0, 11.0, 13.0))),
        dtype=torch.float64,
    )
    indices = torch.tensor([0, 1, 3, 4, 11, 12, 23, 4])
    torch.testing.assert_close(chart_ops.positions(state, indices), expected[indices])
    assert state.shape == (6, 4)
    assert state.features == 24
    assert sum(b.numel() for b in state.buffers()) == 6
    large = compile_chart(
        charts.periodic_grid(
            (10000, 20000, 30000), periods=(2.0, 3.0, 8.0), output_dims=2
        )
    )
    assert sum(b.numel() for b in large.buffers()) == 6
    assert chart_ops.positions(state, torch.empty(0, dtype=torch.long)).shape == (0, 3)
    for indices in (torch.tensor([-1]), torch.tensor([24])):
        with pytest.raises(IndexError):
            chart_ops.positions(state, indices)
    with pytest.raises(ValueError):
        chart_ops.positions(state, torch.tensor([0.0]))
    with pytest.raises(ValueError):
        chart_ops.positions(state, None)


@pytest.mark.parametrize(
    "settings",
    [
        {"grid_shape": (3,)},
        {"grid_shape": (3, 0)},
        {"grid_shape": (True, 3)},
        {"periods": (1.0,)},
        {"periods": (0.0, 1.0)},
        {"periods": (float("inf"), 1.0)},
        {"periods": (True, 1.0)},
        {"origin": (0.0,)},
        {"origin": (float("nan"), 0.0)},
        {"origin": (True, 0.0)},
        {"output_dims": 0},
        {"output_dims": 2},
        {"output_dims": True},
    ],
)
def test_invalid_declarations(settings):
    kwargs = {"grid_shape": (3, 4), "periods": (1.0, 2.0)}
    kwargs.update(settings)
    with pytest.raises((ValueError, TypeError)):
        charts.periodic_grid(**kwargs)


def test_geometry_and_kernel_revisions_are_distinct_contracts():
    spec = charts.periodic_grid((3, 4), periods=(1.0, 2.0))
    with pytest.raises(ValueError):
        replace(spec, shape=(4, 3))
    with pytest.raises(ValueError):
        FlatTorusGeometrySpec(intrinsic_dim=2, periods=[1.0, 2.0])
    with pytest.raises(ValueError):
        CSTLinear(chart=spec, atoms=2, kernel=replace(make_kernel(2), revision=1))
    with pytest.raises(ValueError):
        CSTLinear(chart=spec, atoms=2, kernel=make_kernel(3))
    with pytest.raises(ValueError):
        compile_chart(replace(spec, revision=2))
    with pytest.raises(ValueError):
        compile_chart(replace(spec, geometry=replace(spec.geometry, revision=2)))
    with pytest.raises(TypeError):
        CSTLinear(chart=spec, atoms=2, kernel=make_kernel(2), backend="tiled")


@pytest.mark.parametrize(
    "sigma,center", [(0.01, 0.13), (0.12, 0.0), (0.6, 0.07), (2.0, 0.11)]
)
def test_empty_singleton_and_wider_than_half_period(sigma, center):
    model = CSTLinear(
        chart=charts.periodic_grid((5, 5), periods=(1.0, 1.0)),
        atoms=torch.tensor([[0.2, 1.2, center, center]], dtype=torch.float64),
        kernel=make_kernel(2, sigma=sigma),
        backend="factored",
    )
    actual = model.materialized_atoms()
    expected = oracle_atoms(model, model.atoms.p)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    actual.sum().backward()
    assert torch.isfinite(model.atoms.p.grad).all()
    if sigma == 0.01:
        assert not actual.count_nonzero()
        assert not model.atoms.p.grad.count_nonzero()


def test_period_translation_invariance_and_cut_locus_branch():
    model = CSTLinear(
        chart=charts.periodic_grid((4, 5, 3), periods=(1.0, 2.0, 3.0)),
        atoms=torch.tensor([[0.2, 1.2, 0.07, -0.03, 2.91]], dtype=torch.float64),
        kernel=make_kernel(3, sigma=2.0),
        backend="factored",
    )
    p = model.atoms.p.detach().clone().requires_grad_()
    shifted = (p.detach() + p.new_tensor([0.0, 0.0, 3.0, -4.0, 6.0])).requires_grad_()
    a = kernel_ops.materialize_atoms(model.kernel, model.chart, p)
    b = kernel_ops.materialize_atoms(model.kernel, model.chart, shifted)
    torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)
    g = torch.randn_like(a)
    torch.testing.assert_close(
        torch.autograd.grad((a * g).sum(), p)[0],
        torch.autograd.grad((b * g).sum(), shifted)[0],
        rtol=1e-10,
        atol=1e-11,
    )
    delta = torch.tensor(
        [-1.5, -0.5, 0.5, 1.5], dtype=torch.float64, requires_grad=True
    )
    wrapped = wrap_delta(delta, delta.new_tensor(1.0))
    torch.testing.assert_close(wrapped, torch.full_like(delta, -0.5))
    torch.testing.assert_close(
        torch.autograd.grad(wrapped.square().sum(), delta)[0],
        torch.full_like(delta, -1.0),
    )


def test_center_gradient_finite_difference_with_fixed_width():
    model = CSTLinear(
        chart=charts.periodic_grid((5, 7), periods=(1.0, 1.3)),
        atoms=torch.tensor([[0.2, 1.2, 0.037, 1.257]], dtype=torch.float64),
        kernel=make_kernel(2),
        backend="factored",
    )
    p = model.atoms.p
    cot = torch.randn_like(model.materialized_atoms())
    loss = (model.materialized_atoms() * cot).sum()
    grad = torch.autograd.grad(loss, p)[0]
    eps = 1e-6
    for d in (2, 3):
        plus, minus = p.detach().clone(), p.detach().clone()
        plus[0, d] += eps
        minus[0, d] -= eps
        fd = (
            (
                kernel_ops.materialize_atoms(model.kernel, model.chart, plus)
                - kernel_ops.materialize_atoms(model.kernel, model.chart, minus)
            )
            * cot
        ).sum() / (2 * eps)
        torch.testing.assert_close(grad[0, d], fd, rtol=1e-7, atol=1e-8)


def test_live_buffers_checkpoint_structure_and_dtype():
    spec = charts.periodic_grid((3, 4, 5), periods=(1.0, 2.0, 3.0), output_dims=2)
    model = CSTLinear(
        chart=spec,
        atoms=2,
        kernel=make_kernel(3),
        dtype=torch.float64,
        backend="factored",
    )
    with torch.no_grad():
        model.chart.origin.add_(0.17)
        model.chart.geometry.periods.mul_(1.2)
    x = torch.randn(2, 5, dtype=torch.float64)
    torch.testing.assert_close(
        model(x),
        torch.nn.functional.linear(x, oracle_atoms(model, model.atoms.p).sum(0)),
    )
    clone = CSTLinear(
        chart=spec,
        atoms=2,
        kernel=make_kernel(3),
        dtype=torch.float64,
        backend="factored",
    )
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(clone(x), model(x))
    assert clone.chart.declaration() == model.chart.declaration()
    clone.float()
    assert clone.chart.spacing.dtype == torch.float32
    incompatible = compile_chart(
        charts.periodic_grid((2, 6, 5), periods=(1.0, 2.0, 3.0), output_dims=2),
        dtype=torch.float64,
    )
    with pytest.raises(RuntimeError, match="contract differs"):
        incompatible.load_state_dict(model.chart.state_dict())
    bad = copy.deepcopy(model.chart.state_dict())
    bad["geometry.periods"][0] = 0
    with pytest.raises(RuntimeError, match="invalid coordinate checkpoint"):
        compile_chart(spec, dtype=torch.float64).load_state_dict(bad)
    with pytest.raises(ValueError):
        compile_chart(
            charts.periodic_grid((3, 4), periods=(1e-20, 1.0)), dtype=torch.float16
        )


def test_periodic_retraction_optimizer_moments_and_live_width():
    model = CSTLinear(
        chart=charts.periodic_grid((3, 4), periods=(1.0, 2.0)),
        atoms=torch.tensor([[0.2, 1.2, 0.99, 0.01]], dtype=torch.float64),
        kernel=make_kernel(2, live_width=True),
        backend="factored",
    )
    base = torch.optim.Adam(model.parameters(), lr=0.05)
    opt = CSTOptimizer(base, model=model)
    previous = model.atoms.p.detach().clone()
    old_sigma = kernel_ops.coordinate(
        model.kernel, "bandwidth_sigma", model.chart, previous
    )
    model.atoms.p.grad = torch.tensor([[0.3, -0.1, -1.0, 1.0]], dtype=torch.float64)
    opt.step()
    new = model.atoms.p.detach()
    torch.testing.assert_close(
        new[:, 2:],
        torch.tensor([[0.04, 1.96]], dtype=torch.float64),
        rtol=1e-8,
        atol=1e-9,
    )
    assert base.state[model.atoms.p]["step"] == 1
    torch.testing.assert_close(
        base.state[model.atoms.p]["exp_avg"][:, 2:],
        torch.tensor([[-0.1, 0.1]], dtype=torch.float64),
    )
    new_sigma = kernel_ops.coordinate(model.kernel, "bandwidth_sigma", model.chart, new)
    assert not torch.equal(old_sigma, new_sigma)
    torch.testing.assert_close(
        model.materialized_atoms(),
        oracle_atoms(model, model.atoms.p),
        rtol=1e-11,
        atol=1e-12,
    )
    vector = torch.ones_like(previous[:, 2:])
    assert (
        geometry_ops.transport(
            model.chart.geometry, previous[:, 2:], new[:, 2:], vector
        )
        is vector
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_low_precision_empty_support_finite(dtype):
    model = CSTLinear(
        chart=charts.periodic_grid((3, 4), periods=(1.0, 2.0)),
        atoms=torch.tensor([[0.2, 1.2, 0.15, 0.2]], dtype=dtype),
        kernel=make_kernel(2, sigma=0.01),
        backend="factored",
    )
    x = torch.ones(2, 4, dtype=dtype, requires_grad=True)
    y = model(x)
    y.sum().backward()
    assert (
        torch.isfinite(y).all()
        and torch.isfinite(x.grad).all()
        and torch.isfinite(model.atoms.p.grad).all()
    )
    assert not y.count_nonzero()


def test_initialization_and_wrapped_geometry_distance():
    spec = charts.periodic_grid(
        (3, 4, 2), periods=(1.0, 2.0, 3.0), origin=(-2.0, 0.3, 7.0), output_dims=2
    )
    chart = compile_chart(spec, dtype=torch.float64)
    balanced = chart_ops.initialize_centers(chart, 3, mode="balanced")
    torch.testing.assert_close(
        balanced, chart_ops.positions(chart, torch.tensor([0, 12, 23]))
    )
    uniform = chart_ops.initialize_centers(chart, 100, mode="uniform")
    assert (
        (uniform >= chart.origin) & (uniform < chart.origin + chart.geometry.periods)
    ).all()
    sites = chart_ops.positions(chart, torch.tensor([0, 23]))
    centers = sites + sites.new_tensor([1.0, -4.0, 6.0]) + 0.01
    distance = geometry_ops.squared_distance(chart.geometry, sites, centers)
    torch.testing.assert_close(
        distance.diagonal(), torch.full((2,), 0.0003, dtype=torch.float64)
    )
    for atoms, mode in [(True, "balanced"), (0, "balanced"), (1, "invalid")]:
        with pytest.raises(ValueError):
            chart_ops.initialize_centers(chart, atoms, mode=mode)


def test_ordinary_grid_rejects_periodic_kernel_revision():
    ordinary = charts.product(
        (3, 4),
        (patterns.line(3, low=0.0, high=1.0), patterns.line(4, low=0.0, high=1.0)),
    )
    with pytest.raises(ValueError, match="revision 3"):
        CSTLinear(chart=ordinary, atoms=2, kernel=make_kernel(2))


def test_large_axis_half_precision_spacing_avoids_count_overflow():
    chart = compile_chart(
        charts.periodic_grid((100000, 3), periods=(1.0, 2.0)), dtype=torch.float16
    )
    sites = chart_ops.positions(chart, torch.tensor([0, 299997]))
    assert torch.isfinite(sites).all()
    torch.testing.assert_close(sites[-1, 0], torch.tensor(1.0, dtype=torch.float16))
    assert chart.spacing[0] > 0
    torch.testing.assert_close(
        chart.spacing[0].float(), torch.tensor(1e-5), rtol=0.01, atol=0.0
    )


def test_two_outstanding_forwards_and_retained_backward():
    torch.manual_seed(5)
    model = CSTLinear(
        chart=charts.periodic_grid((3, 4, 5), periods=(1.0, 1.4, 1.8), output_dims=2),
        atoms=torch.tensor([[0.15, 1.2, 0.047, 1.351, 0.037]], dtype=torch.float64),
        kernel=make_kernel(3, sigma=0.9),
        backend="factored",
    )
    p = model.atoms.p.detach().clone().requires_grad_()
    x1 = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    x2 = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    y1, y2 = model(x1), model(x2)
    w = oracle_atoms(model, p).sum(0)
    expected = (
        torch.nn.functional.linear(x1, w).square().sum()
        + torch.nn.functional.linear(x2, w).square().sum()
    )
    loss = y1.square().sum() + y2.square().sum()
    grads = torch.autograd.grad(loss, (model.atoms.p, x1, x2), retain_graph=True)
    oracle_grads = torch.autograd.grad(expected, (p, x1, x2))
    for actual, reference in zip(grads, oracle_grads):
        torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-11)
    repeated = torch.autograd.grad(loss, (model.atoms.p, x1, x2))
    for first, second in zip(grads, repeated):
        torch.testing.assert_close(first, second)
