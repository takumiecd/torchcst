"""Independent full-site oracle for the single-chart Polar product contract."""

import copy
import io
import itertools
from dataclasses import replace

import pytest
import torch

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    CSTOptimizer,
    GaussianSpec,
    NormalizationSpec,
    TriweightSpec,
    presets,
)
from torchcst import (
    chart_presets as charts,
)
from torchcst import (
    geometry_presets as geometry,
)
from torchcst import (
    pattern_presets as patterns,
)
from torchcst._backends.torch.kernels import execution as kernels


def make_chart(kind="product", *, multidimensional=False, axis=0):
    axes = (
        patterns.line(4, low=-0.2, high=0.85),
        patterns.grid((2, 3), low=(0.1, -0.2), high=(0.4, 0.3))
        if multidimensional
        else patterns.line(5, low=0.1, high=1.1),
    )
    shape = (4, 6 if multidimensional else 5)
    if kind == "strip":
        tile = (3, shape[1]) if axis == 0 else (4, 3)
        return charts.strip(shape, tile, axes=axes, axis=axis, tile_pitch=1.4)
    return charts.product(shape, axes)


def make_spec(dims=2, *, floor=1e-6, fixed_width=False, mixed=False):
    bounds = (
        BandwidthBounds(minimum=1.3, birth=1.3, maximum=1.3, upper_floor=1.3)
        if fixed_width
        else BandwidthBounds(minimum=0.15, birth=1.1, maximum=2.0, upper_floor=0.2)
    )
    shapes = tuple(
        GaussianSpec() if mixed and d % 2 else TriweightSpec() for d in range(dims)
    )
    return presets.polar_profile_product(
        profiles=shapes,
        amplitude_max=2.0,
        bounds=bounds,
        w_c=0.5,
        alpha_init=0.3,
        radial_regularization=0.2,
        normalization=NormalizationSpec(
            kind="discrete_l2", domain="operator_sites", floor=floor
        ),
    )


def oracle_polar(state, p):
    """Recompute the declared map without backend decode helpers."""
    q = p[:, :2].square().sum(-1)
    amp = state.amplitude_max * p[:, 0] / q.clamp_min(torch.finfo(p.dtype).tiny).sqrt()
    alpha = ((q - 1) / 3).clamp(0, 1)
    x = (amp / state.w_c).square()
    lower = state.sigma_min_input + (
        state.sigma_birth_input - state.sigma_min_input
    ) / (1 + state.lower_kappa * x)
    upper = state.sigma_min_input + (
        state.sigma_max_input - state.sigma_min_input
    ) * state.kappa / (state.kappa + x.pow(state.upper_decay_power))
    upper = torch.maximum(torch.maximum(upper, state.upper_floor_input), lower)
    sigma = torch.exp((1 - alpha) * lower.log() + alpha * upper.log()).clamp(
        min=lower, max=upper
    )
    return amp, sigma


def oracle_coordinates(chart):
    """Enumerate actual sites without using chart/backend position evaluation."""
    logical_axes = []
    for index, pattern in enumerate(chart.axes):
        sites = []
        for flat, indices in enumerate(
            itertools.product(*(range(n) for n in pattern.spec.shape))
        ):
            row = []
            for dim, local in enumerate(indices):
                coordinate = pattern.start[dim] + local * pattern.spacing[dim]
                if chart.spec.kind == "strip" and index == chart.axis:
                    tile = chart.tile_shape[index]
                    coordinate = (
                        pattern.start[dim]
                        + (flat % tile) * pattern.spacing[dim]
                        + (flat // tile) * chart.tile_pitch
                    )
                row.append(coordinate)
            sites.append(torch.stack(row))
        logical_axes.append(torch.stack(sites))
    return logical_axes


def oracle_atoms(model, p):
    amp, sigma = oracle_polar(model.kernel, p)
    output_sites, input_sites = oracle_coordinates(model.chart)
    rows = []
    for output in output_sites:
        columns = []
        for source in input_sites:
            position = torch.cat((output, source))
            value = torch.ones_like(amp)
            for dim, profile in enumerate(model.kernel.spec.profiles):
                squared = (
                    position[dim] - p[:, 2 + dim]
                ).square() / sigma.detach().square()
                if isinstance(profile.profile, TriweightSpec):
                    value = value * (1 - squared).clamp_min(0).pow(3)
                else:
                    value = value * torch.exp(-0.5 * squared)
            columns.append(value)
        rows.append(torch.stack(columns))
    raw = torch.stack(rows)
    norm = torch.linalg.vector_norm(raw.flatten(0, 1), dim=0)
    return (
        raw
        * (amp / norm.clamp_min(model.kernel.spec.normalization.floor))[None, None, :]
    ).permute(2, 0, 1)


@pytest.mark.parametrize("backend", ["auto", "materialized", "factored"])
@pytest.mark.parametrize(
    "kind,multi,axis",
    [
        ("product", False, 0),
        ("product", True, 0),
        ("strip", False, 0),
        ("strip", False, 1),
        ("strip", True, 0),
    ],
)
def test_full_site_values_dx_and_all_source_gradients(backend, kind, multi, axis):
    torch.manual_seed(9)
    model = CSTLinear(
        chart=make_chart(kind, multidimensional=multi, axis=axis),
        atoms=3,
        kernel=make_spec(3 if multi else 2, mixed=True),
        dtype=torch.float64,
        backend=backend,
    )
    with torch.no_grad():
        model.atoms.p[:, :2].copy_(torch.tensor([[0.3, 1.2], [-0.4, 1.1], [0.1, 1.3]]))
        model.atoms.p[:, 2:].add_(0.07)
    point = model.atoms.p.detach().clone().requires_grad_()
    expected_atoms = oracle_atoms(model, point)
    torch.testing.assert_close(
        model.materialized_atoms(), expected_atoms, rtol=1e-11, atol=1e-12
    )
    x = torch.randn(2, 3, model.in_features, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    expected = torch.nn.functional.linear(reference_x, expected_atoms.sum(0))
    actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    cotangent = torch.randn_like(actual)
    (actual * cotangent).sum().backward()
    (expected * cotangent).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(model.atoms.p.grad, point.grad, rtol=1e-10, atol=1e-11)
    assert torch.count_nonzero(point.grad[:, 2:]) == point[:, 2:].numel()


@pytest.mark.parametrize("backend", ["materialized", "factored"])
@pytest.mark.parametrize("case", ["empty", "singleton", "floor", "global_floor"])
def test_support_and_one_global_floor(case, backend):
    chart = charts.product(
        (2, 2),
        (patterns.line(2, low=0.0, high=1.0), patterns.line(2, low=0.0, high=1.0)),
    )
    floor = 0.5 if case == "global_floor" else 1e-6
    spec = make_spec(floor=floor)
    bounds = BandwidthBounds(minimum=0.3, birth=0.3, maximum=0.3, upper_floor=0.3)
    spec = replace(
        spec,
        parameterization=replace(
            spec.parameterization, input_bounds=bounds, output_bounds=bounds
        ),
    )
    center = {
        "empty": 10.0,
        "singleton": 0.0,
        "floor": 0.3 * (1 - 1e-4),
        "global_floor": 0.3 * (1 - 0.8 ** (1 / 3)) ** 0.5,
    }[case]
    p = torch.tensor([[0.4, 1.2, center, center]], dtype=torch.float64)
    model = CSTLinear(chart=chart, atoms=p, kernel=spec, backend=backend)
    point = model.atoms.p.detach().clone().requires_grad_()
    expected = oracle_atoms(model, point)
    actual = model.materialized_atoms()
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-14)
    actual.sum().backward()
    expected.sum().backward()
    assert torch.isfinite(model.atoms.p.grad).all()
    torch.testing.assert_close(model.atoms.p.grad, point.grad, rtol=1e-9, atol=1e-12)
    if case == "global_floor":
        # Each axis norm is 0.8 > floor, and this product is floor-inactive.
        # Change to 0.6 per axis: both remain above floor, product falls below it.
        c = model.kernel.sigma_birth_input * (1 - 0.6 ** (1 / 3)) ** 0.5
        with torch.no_grad():
            model.atoms.p[:, 2:].fill_(c)
        amp = kernels.coordinate(model.kernel, "amplitude", model.chart, model.atoms.p)
        torch.testing.assert_close(
            model.materialized_atoms()[0, 0, 0], amp[0] * 0.36 / floor
        )


def test_checkpoint_and_optimizer_match_existing_polar_shared_width_law():
    torch.manual_seed(17)
    chart = make_chart()
    spec = make_spec()
    single = CSTLinear(
        chart=chart, atoms=3, kernel=spec, dtype=torch.float64, backend="factored"
    )
    pair_spec = presets.polar_activity(
        amplitude_max=2.0,
        input_bounds=spec.parameterization.input_bounds,
        w_c=0.5,
        profile=presets.profile(TriweightSpec()),
        alpha_init=0.3,
        radial_regularization=0.2,
    )
    pair = CSTLinear(
        charts.product((5,), (chart.axes[1],)),
        charts.product((4,), (chart.axes[0],)),
        atoms=single.atoms.p.detach()[:, [0, 1, 3, 2]].clone(),
        kernel=pair_spec,
        backend="factored",
        dtype=torch.float64,
    )
    single_opt = CSTOptimizer(
        torch.optim.AdamW(single.parameters(), lr=0.01), model=single
    )
    pair_opt = CSTOptimizer(torch.optim.AdamW(pair.parameters(), lr=0.01), model=pair)
    initial_sigma = (
        kernels.coordinate(
            single.kernel, "bandwidth_sigma", single.chart, single.atoms.p
        )
        .detach()
        .clone()
    )
    for _ in range(5):
        x = torch.randn(3, 5, dtype=torch.float64)
        target = torch.randn(3, 4, dtype=torch.float64)
        outputs = [model(x) for model in (single, pair)]
        torch.testing.assert_close(*outputs, rtol=1e-10, atol=1e-11)
        for model, opt, y in zip((single, pair), (single_opt, pair_opt), outputs):
            opt.zero_grad()
            (y - target).square().sum().backward()
            opt.step()
        torch.testing.assert_close(
            single.atoms.p, pair.atoms.p[:, [0, 1, 3, 2]], rtol=1e-10, atol=1e-11
        )
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                single_opt.state[single.atoms.p][name],
                pair_opt.state[pair.atoms.p][name][:, [0, 1, 3, 2]],
                rtol=1e-10,
                atol=1e-11,
            )
        torch.testing.assert_close(
            single_opt.state[single.atoms.p]["step"],
            pair_opt.state[pair.atoms.p]["step"],
        )
    assert not torch.equal(
        initial_sigma,
        kernels.coordinate(
            single.kernel, "bandwidth_sigma", single.chart, single.atoms.p
        ),
    )
    stream = io.BytesIO()
    torch.save(
        {"model": single.state_dict(), "optimizer": single_opt.state_dict()}, stream
    )
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    restored = copy.deepcopy(single)
    restored.load_state_dict(saved["model"])
    restored_opt = CSTOptimizer(
        torch.optim.AdamW(restored.parameters(), lr=0.01), model=restored
    )
    restored_opt.load_state_dict(saved["optimizer"])
    assert restored.declaration() == single.declaration()
    torch.testing.assert_close(restored.dense_weight(), single.dense_weight())
    changed = CSTLinear(
        chart=chart, atoms=3, kernel=make_spec(floor=1e-4), dtype=torch.float64
    )
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        changed.load_state_dict(saved["model"])


def test_factorized_derivatives_and_no_execution_snapshots(monkeypatch):
    model = CSTLinear(
        chart=make_chart(),
        atoms=2,
        kernel=make_spec(fixed_width=True),
        dtype=torch.float64,
        backend="factored",
    )
    point = model.atoms.p.detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(model._materialize_atoms, (point,))
    derivatives = model.cst_derivatives()
    assert derivatives.factor_atoms is not None
    direction = torch.randn_like(point)
    jacobian = torch.func.jacfwd(model._materialize_atoms)(point)
    expected = torch.einsum("aoikp,kp->oi", jacobian, direction)
    torch.testing.assert_close(derivatives.jvp(direction), expected)

    def forbidden(*args, **kwargs):
        raise AssertionError("snapshot during execution")

    for obj in (model, model.chart, model.kernel):
        monkeypatch.setattr(obj, "declaration", forbidden)
    model(torch.ones(2, 5, dtype=torch.float64)).sum().backward()


def test_reject_incompatible_declarations_and_layouts():
    spec = make_spec()
    for changes in (
        {"profiles": ()},
        {"normalization": None},
        {"normalization": NormalizationSpec()},
        {
            "parameterization": replace(
                spec.parameterization,
                output_bounds=BandwidthBounds(
                    minimum=0.2, birth=1.0, maximum=1.0, upper_floor=0.2
                ),
            )
        },
        {"profiles": (presets.profile(TriweightSpec()),) * 2},
        {"parameterization": None},
    ):
        with pytest.raises(ValueError):
            replace(spec, **changes)
    with pytest.raises(ValueError, match="coordinate dimension"):
        CSTLinear(chart=make_chart(multidimensional=True), atoms=2, kernel=spec)
    with pytest.raises(ValueError, match="composition"):
        CSTLinear(
            charts.linspace(4, spacing=0.3),
            charts.linspace(5, spacing=0.3),
            atoms=2,
            kernel=spec,
        )
    torus = charts.product(
        (4, 5),
        (patterns.line(4, spacing=0.1), patterns.line(5, spacing=0.1)),
        geometry=geometry.torus(2, major_radius=3.0, minor_radius=1.0),
    )
    with pytest.raises(ValueError, match="Euclidean"):
        CSTLinear(chart=torus, atoms=2, kernel=spec)


def test_operator_state_ownership_and_live_physical_chart():
    model = CSTLinear(
        chart=make_chart("strip"), atoms=3, kernel=make_spec(), dtype=torch.float64
    )
    operator = model.operator
    assert operator.charts == (model.chart,)
    assert operator.kernel is model.kernel and operator.p is model.atoms.p
    before = model.dense_weight().detach().clone()
    with torch.no_grad():
        model.chart.tile_pitch.add_(0.2)
    assert not torch.equal(before, model.dense_weight())
    torch.testing.assert_close(
        model.materialized_atoms(), oracle_atoms(model, model.atoms.p)
    )
    model.float()
    assert operator.p is model.atoms.p and operator.p.dtype == torch.float32


def test_evolving_width_task_derivative_is_held_fixed():
    model = CSTLinear(
        chart=make_chart(), atoms=2, kernel=make_spec(), dtype=torch.float64
    )
    sigma = kernels.coordinate(
        model.kernel, "bandwidth_sigma", model.chart, model.atoms.p
    )
    assert (
        torch.count_nonzero(
            torch.autograd.grad(sigma.sum(), model.atoms.p, retain_graph=True)[0][:, :2]
        )
        > 0
    )
    actual = model.materialized_atoms()
    point = model.atoms.p.detach().clone().requires_grad_()
    expected = oracle_atoms(model, point)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), model.atoms.p)[0],
        torch.autograd.grad(expected.sum(), point)[0],
    )


@pytest.mark.parametrize("backend", ["auto", "materialized", "factored"])
def test_amplitude_wrapper_keeps_single_chart_factors_and_update(backend):
    inner = CSTLinear(
        chart=make_chart(), atoms=2, kernel=make_spec(), dtype=torch.float64
    )
    p = torch.cat(
        (torch.tensor([[0.7], [-0.4]], dtype=torch.float64), inner.atoms.p.detach()),
        dim=1,
    )
    model = CSTLinear(
        chart=make_chart(),
        atoms=p,
        kernel=presets.amplitude(inner.kernel.spec),
        dtype=torch.float64,
        backend=backend,
    )
    x = torch.randn(3, 5, dtype=torch.float64)
    expected = torch.nn.functional.linear(
        x, (inner.materialized_atoms() * p[:, 0, None, None]).sum(0)
    )
    torch.testing.assert_close(model(x), expected)
    assert model.cst_derivatives().factor_atoms is not None
    assert model.operator.factors()[0].shape == (5, 2)
    optimizer = CSTOptimizer(
        torch.optim.AdamW(model.parameters(), lr=0.001), model=model
    )
    model(x).square().sum().backward()
    optimizer.step()
    assert torch.isfinite(model.atoms.p).all()
