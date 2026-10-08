"""Full physical-query oracle for the declared S1 x S2 fibre product."""

import copy
from dataclasses import replace

import pytest
import torch
from test_profile_product import oracle_coordinates, oracle_polar

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    CSTOptimizer,
    NormalizationSpec,
    presets,
)
from torchcst import chart_presets as charts
from torchcst import geometry_presets as geometries
from torchcst import pattern_presets as patterns
from torchcst._backends.torch.geometry import torus
from torchcst.geometry.state import GeometryState
from torchcst.kernels.profiles import GaussianSpec, TriweightSpec


def model(
    kind="strip",
    *,
    reverse=False,
    representation="intrinsic",
    floor=1e-6,
    fixed=False,
    backend="factored",
    device="cpu",
):
    circle = patterns.line(17, low=-0.3, high=1.3)
    section = patterns.grid((3, 4), low=(-0.12, -0.15), high=(0.12, 0.15))
    geometry = geometries.torus(
        3,
        major_radius=3.5,
        minor_radius=0.4,
        circle_axis=2 if reverse else 0,
        representation=representation,
        max_arc_step=0.2,
    )
    shape, axes = (
        ((12, 17), (section, circle)) if reverse else ((17, 12), (circle, section))
    )
    chart = charts.product(shape, axes, geometry=geometry)
    if kind == "strip":
        chart = charts.strip(
            shape,
            (12, 8) if reverse else (8, 12),
            axes=axes,
            axis=1 if reverse else 0,
            tile_pitch=2.1,
            geometry=geometry,
        )
    bounds = BandwidthBounds(minimum=0.15, birth=0.8, maximum=1.5, upper_floor=0.2)
    if fixed:
        bounds = BandwidthBounds(minimum=0.6, birth=0.6, maximum=0.6, upper_floor=0.6)
    spec = presets.polar_torus_profile_product(
        profiles=(TriweightSpec(), GaussianSpec()),
        amplitude_max=1.0,
        bounds=bounds,
        w_c=0.5,
        alpha_init=0.3,
        normalization=NormalizationSpec(
            kind="discrete_l2", domain="operator_sites", floor=floor
        ),
    )
    p = torch.tensor(
        [
            [0.3, 1.5, 0.45, 0.05, -0.09],
            [-0.4, 1.4, 2.3, -0.15, 0.21],
            [0.2, 1.6, -0.15, 0.0, 0.0],
            [0.1, 1.4, 8.1, 0.8, 0.0],
        ],
        dtype=torch.float64,
    )
    if representation == "ambient":
        intrinsic = GeometryState(
            replace(geometry, representation="intrinsic")
        ).double()
        p = torch.cat((p[:, :2], torus.decode_centers(intrinsic, p[:, 2:])), dim=-1)
    return CSTLinear(
        chart=chart,
        atoms=p,
        kernel=spec,
        dtype=torch.float64,
        backend=backend,
        device=device,
    )


def physical_atoms(layer, p, *, detach_circle_section=False):
    """Query actual embedded fibres and normalize every matrix entry directly."""
    geometry = layer.chart.geometry
    major, minor = geometry.major_radius, geometry.minor_radius
    decoded = torus.decode_centers(geometry, p[:, 2:])
    radius = torch.linalg.vector_norm(decoded[:, :2], dim=-1)
    q = torch.cat((((radius - major) / minor)[:, None], decoded[:, 2:] / minor), dim=-1)
    angle = torch.atan2(decoded[:, 1], decoded[:, 0])
    reverse = layer.chart.spec.geometry.circle_axis == 2
    coordinates = oracle_coordinates(layer.chart)
    circle, cross = coordinates[::-1] if reverse else coordinates
    sites = torch.cat((minor.expand(len(cross), 1), cross), dim=-1)
    sites = sites / torch.linalg.vector_norm(sites, dim=-1, keepdim=True)
    circle_q = q.detach() if detach_circle_section else q
    circular = torus._embed(
        geometry,
        (circle[:, 0] / major)[:, None].expand(-1, len(p)),
        circle_q[None, :, :].expand(len(circle), -1, -1),
    )
    circular_center = decoded.detach() if detach_circle_section else decoded
    # The negative control only removes section dependence from the circle;
    # retain angle dependence so it remains a meaningful coupled-VJP check.
    if detach_circle_section:
        circular_center = torus._embed(geometry, angle, q.detach())
    section = torus._embed(
        geometry,
        angle[None, :].expand(len(sites), -1),
        sites[:, None, :].expand(-1, len(p), -1),
    )
    squared = [
        (circular - circular_center[None, :, :]).square().sum(-1),
        (section - decoded[None, :, :]).square().sum(-1),
    ]
    amplitude, sigma = oracle_polar(layer.kernel, p)
    raw = [
        (1 - squared[0] / sigma.detach().square()[None, :]).clamp_min(0).pow(3),
        torch.exp(-0.5 * squared[1] / sigma.detach().square()[None, :]),
    ]
    out, source = raw[::-1] if reverse else raw
    matrix = out[:, None, :] * source[None, :, :]
    norm = torch.linalg.vector_norm(matrix.flatten(0, 1), dim=0)
    return (
        matrix
        * (amplitude / norm.clamp_min(layer.kernel.spec.normalization.floor))[
            None, None, :
        ]
    ).permute(2, 0, 1)


@pytest.mark.parametrize("backend", ["materialized", "factored", "auto"])
@pytest.mark.parametrize(
    "kind,reverse", [("product", False), ("strip", False), ("strip", True)]
)
@pytest.mark.parametrize("representation", ["intrinsic", "ambient"])
@pytest.mark.parametrize("floor", [1e-6, 0.5])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_physical_values_dx_and_every_parameter_gradient(
    backend, kind, reverse, representation, floor, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(41)
    layer = model(
        kind,
        reverse=reverse,
        representation=representation,
        floor=floor,
        backend=backend,
        device=device,
    )
    p = layer.atoms.p.detach().clone().requires_grad_()
    expected = physical_atoms(layer, p)
    torch.testing.assert_close(
        layer.materialized_atoms(), expected, rtol=1e-10, atol=1e-12
    )
    x = torch.randn(
        2, 3, layer.in_features, dtype=torch.float64, device=device, requires_grad=True
    )
    rx = x.detach().clone().requires_grad_()
    dy = torch.randn(2, 3, layer.out_features, dtype=torch.float64, device=device)
    y, reference = layer(x), rx @ expected.sum(0).T
    actual = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    truth = torch.autograd.grad(reference, (rx, p), dy)
    torch.testing.assert_close(y, reference, rtol=1e-10, atol=1e-12)
    for value, target in zip(actual, truth, strict=True):
        torch.testing.assert_close(value, target, rtol=1e-9, atol=1e-11)


def test_circle_section_coupling_is_required():
    layer = model(fixed=True)
    p = layer.atoms.p.detach().clone().requires_grad_()
    dy = torch.arange(layer.out_features * layer.in_features, dtype=p.dtype).reshape(
        layer.out_features, layer.in_features
    )
    good = torch.autograd.grad(physical_atoms(layer, p).sum(0), p, dy)[0]
    wrong = torch.autograd.grad(
        physical_atoms(layer, p, detach_circle_section=True).sum(0), p, dy
    )[0]
    assert (good[:, 3:] - wrong[:, 3:]).abs().max() > 1e-5


@pytest.mark.parametrize("representation", ["intrinsic", "ambient"])
def test_optimizer_retraction_moments_and_checkpoint(representation):
    torch.manual_seed(41)
    actual = model(representation=representation)
    reference = copy.deepcopy(actual)
    optimizers = [
        CSTOptimizer(torch.optim.AdamW(m.parameters(), lr=1e-3), model=m)
        for m in (actual, reference)
    ]
    for _ in range(5):
        x = torch.randn(3, actual.in_features, dtype=torch.float64)
        dy = torch.randn(3, actual.out_features, dtype=torch.float64)
        ys = [actual(x), x @ physical_atoms(reference, reference.atoms.p).sum(0).T]
        for opt, y in zip(optimizers, ys, strict=True):
            opt.zero_grad()
            (y * dy).sum().backward()
            opt.step()
        torch.testing.assert_close(
            actual.atoms.p, reference.atoms.p, rtol=1e-9, atol=1e-11
        )
        torus.validate_centers(actual.chart.geometry, actual.atoms.p[:, 2:])
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                optimizers[0].state[actual.atoms.p][key],
                optimizers[1].state[reference.atoms.p][key],
                rtol=1e-9,
                atol=1e-11,
            )
    restored = copy.deepcopy(actual)
    restored.load_state_dict(actual.state_dict())
    assert restored.declaration() == actual.declaration()
    torch.testing.assert_close(restored.dense_weight(), actual.dense_weight())


def test_declaration_keeps_old_metrics_distinct():
    layer = model()
    with pytest.raises(ValueError, match="revision 1"):
        CSTLinear(
            chart=layer.chart.spec,
            atoms=2,
            kernel=replace(layer.kernel.spec, revision=1),
        )
    with pytest.raises(ValueError, match="two profiles"):
        replace(layer.kernel.spec, profiles=layer.kernel.spec.profiles[:1])


@pytest.mark.parametrize("representation", ["intrinsic", "ambient"])
def test_initialization_and_zero_atom_operator(representation):
    layer = model(representation=representation)
    initialized = CSTLinear(
        chart=layer.chart.spec, atoms=7, kernel=layer.kernel.spec, dtype=torch.float64
    )
    torus.validate_centers(initialized.chart.geometry, initialized.atoms.p[:, 2:])
    assert torch.isfinite(initialized.dense_weight()).all()
    empty = CSTLinear(
        chart=layer.chart.spec,
        atoms=torch.empty(0, layer.atoms.p.shape[1], dtype=torch.float64),
        kernel=layer.kernel.spec,
        dtype=torch.float64,
        backend="factored",
    )
    x = torch.randn(3, empty.in_features, dtype=torch.float64, requires_grad=True)
    y = empty(x)
    dx, dp = torch.autograd.grad(y.sum(), (x, empty.atoms.p))
    assert (
        not y.count_nonzero()
        and not dx.count_nonzero()
        and dp.shape == empty.atoms.p.shape
    )


@pytest.mark.parametrize("representation", ["intrinsic", "ambient"])
def test_explicit_polar_update_matches_general_geometry_update(representation):
    from torchcst import AtomUpdateBinding, Dispatcher
    from torchcst._backends.catalog import get_registry
    from torchcst._backends.schema import DefaultRecipe, ExecutionPlan
    from torchcst._backends.torch.kernels import execution
    from torchcst.operators.atom_update import AtomUpdateInputs

    layer = model(representation=representation)
    previous = layer.atoms.p.detach().clone()
    displacement = torch.ones_like(previous) * 0.01
    displacement[:, 2] = 0.4  # Exercise the declared maximum arc step.
    with torch.no_grad():
        layer.atoms.p.copy_(previous + displacement)
    expected = execution.apply_parameter_update(
        layer.kernel,
        layer.chart,
        previous,
        layer.atoms.p.detach() - previous,
        step_size=1e-3,
    )
    Dispatcher(registry=get_registry()).run(
        AtomUpdateBinding(layer.operator),
        AtomUpdateInputs(previous=previous, step_size=1e-3),
        plan=ExecutionPlan("torch_polar_update", "v1", DefaultRecipe()),
    )
    torch.testing.assert_close(layer.atoms.p, expected, rtol=0, atol=0)
