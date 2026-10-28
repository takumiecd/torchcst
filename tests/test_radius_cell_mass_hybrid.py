"""Hybrid radius routes retain independent masses and every physical VJP."""

import copy
import json
from dataclasses import replace

import pytest
import torch
from test_radius_cell_mass_cuda import (
    GPU,
    atoms,
    axis_plan,
    quadrature_weight,
    reference,
    registry,
)

from torchcst import (
    ChebyshevRadiusSpec,
    CSTLinear,
    Dispatcher,
    LinearInputs,
    RadiusBounds,
    chart_presets,
    presets,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    BLOCK_LOCAL,
    HYBRID_OVERLAP,
    HYBRID_SERIAL,
    REFERENCE,
)

VARIANTS = (HYBRID_SERIAL, HYBRID_OVERLAP, BLOCK_LOCAL)


@pytest.mark.parametrize("plan", VARIANTS)
def test_recipe_json_preserves_hybrid_choices(plan):
    reg = registry()
    assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    fields = json.loads(reg.dumps_plan(plan))["recipe"]
    assert fields["input_direct_levels"] == fields["output_direct_levels"] == 2
    assert fields["schedule"] == plan.recipe.schedule
    assert fields["output_tile_sites"] == 32
    assert REFERENCE.recipe.input_direct_levels == 0
    assert REFERENCE.recipe.output_direct_levels == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"input_direct_levels": -1},
        {"output_direct_levels": -1},
        {"input_direct_levels": True},
        {"output_direct_levels": False},
        {"input_direct_levels": 1.5},
        {"output_direct_levels": "2"},
        {"schedule": "unknown"},
        {"schedule": True},
        {"output_tile_sites": 0},
        {"output_tile_sites": -1},
        {"output_tile_sites": True},
        {"output_tile_sites": 1.5},
    ],
)
def test_invalid_hybrid_recipe_rejected(changes):
    with pytest.raises(ValueError):
        registry().validate_plan(
            replace(HYBRID_SERIAL, recipe=replace(HYBRID_SERIAL.recipe, **changes))
        )


def test_unknown_hybrid_json_field_rejected():
    reg = registry()
    payload = json.loads(reg.dumps_plan(HYBRID_SERIAL))
    payload["recipe"]["unknown_direct_choice"] = 2
    with pytest.raises((ValueError, TypeError)):
        reg.loads_plan(json.dumps(payload))


def check_vjp(plan, recipe, x, p, dy):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
        radius_linear,
    )

    x = x.detach().requires_grad_()
    p = p.detach().requires_grad_()
    expected = reference(plan, x, p)
    gradients = torch.autograd.grad(expected, (x, p), dy.double())
    weight = quadrature_weight(plan, p.detach().double().numpy())
    tolerance = (
        {"atol": 3e-5, "rtol": 3e-4}
        if x.dtype == torch.float32
        else {"atol": 2e-9, "rtol": 2e-8}
    )
    results = []
    for choice in (REFERENCE.recipe, recipe):
        cx, cp = x.detach().cuda().requires_grad_(), p.detach().cuda().requires_grad_()
        y = radius_linear(plan, cx, cp, recipe=choice)
        gx, gp = torch.autograd.grad(y, (cx, cp), dy.cuda(), retain_graph=True)
        again = torch.autograd.grad(y, (cx, cp), dy.cuda())
        for first, second in zip((gx, gp), again):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        for actual, target in zip((y, gx, gp), (expected, *gradients)):
            torch.testing.assert_close(
                actual.cpu().double(), target.double(), **tolerance
            )
        torch.testing.assert_close(
            y.cpu().double(), x.detach().double() @ weight.T, **tolerance
        )
        torch.testing.assert_close(gx.cpu().double(), dy.double() @ weight, **tolerance)
        results.append((y, gx, gp))
    for baseline, hybrid in zip(*results):
        torch.testing.assert_close(hybrid, baseline, **tolerance)


@GPU
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n,merge,modes", [(16, 2, 8), (27, 3, 3)])
@pytest.mark.parametrize("thresholds", [(0, 2), (2, 0), (2, 2), ("all", "all")])
def test_hybrid_routes_all_gradients(variant, n, merge, modes, thresholds):
    a = axis_plan(n, merge=merge, modes=modes)
    plan = HierarchyPlan(input=a, output=a)
    levels = len(a.banks) // 2
    start = tuple(levels if value == "all" else value for value in thresholds)
    recipe = replace(
        variant.recipe, input_direct_levels=start[0], output_direct_levels=start[1]
    )
    rng = torch.Generator().manual_seed(731)
    check_vjp(
        plan,
        recipe,
        torch.randn(3, n, dtype=torch.float64, generator=rng),
        atoms(plan),
        torch.randn(3, n, dtype=torch.float64, generator=rng),
    )


@GPU
@pytest.mark.parametrize("variant", VARIANTS)
def test_asymmetric_hybrid_axes(variant):
    plan = HierarchyPlan(
        input=axis_plan(27, merge=3, modes=3, spacing=0.7, origin=-1.1),
        output=axis_plan(16, modes=8, spacing=1.3, origin=2.1),
    )
    check_vjp(
        plan,
        variant.recipe,
        torch.randn(3, 27, dtype=torch.float64),
        atoms(plan),
        torch.randn(3, 16, dtype=torch.float64),
    )


@GPU
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hybrid_128_five_percent_all_gradients(variant, dtype):
    a = axis_plan(128, modes=8)
    plan = HierarchyPlan(input=a, output=a)
    rng = torch.Generator().manual_seed(9107)
    p = torch.empty(819, 4, dtype=dtype)
    p[:, 0] = torch.randn(819, dtype=dtype, generator=rng) * 0.1
    p[:, 1] = torch.logspace(-0.301, 1.806, 819, dtype=dtype)
    p[:, 2:] = torch.rand(819, 2, dtype=dtype, generator=rng) * 128
    check_vjp(
        plan,
        variant.recipe,
        torch.randn(4, 128, dtype=dtype, generator=rng),
        p,
        torch.randn(4, 128, dtype=dtype, generator=rng),
    )


@GPU
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("batch,count", [(0, 8), (3, 0)])
def test_hybrid_empty_shapes(variant, batch, count):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
        radius_linear,
    )

    a = axis_plan()
    plan = HierarchyPlan(input=a, output=a)
    x = torch.randn(batch, 16, dtype=torch.float64, device="cuda", requires_grad=True)
    p = atoms(plan)[:count].cuda().requires_grad_()
    y = radius_linear(plan, x, p, recipe=variant.recipe)
    assert y.shape == (batch, 16)
    for gradient in torch.autograd.grad(y.sum(), (x, p)):
        assert torch.equal(gradient, torch.zeros_like(gradient))


@GPU
@pytest.mark.parametrize("variant", VARIANTS)
def test_hybrid_public_polar_distinct_bounds(variant):
    chart = chart_presets.regular_grid(((16,),), spacing=1.0, geometry="flat_torus")
    kernel = presets.polar_radius_cell_mass(
        amplitude_max=1.3,
        w_c=0.3,
        input_radius_bounds=RadiusBounds(
            minimum=0.5, birth=1.0, maximum=4.0, upper_floor=1.0
        ),
        output_radius_bounds=RadiusBounds(
            minimum=0.6, birth=1.3, maximum=6.0, upper_floor=1.5
        ),
        profile=ChebyshevRadiusSpec(modes=3),
    )
    p = torch.tensor(
        [[1.0, 0.5, 2.17, 3.31], [0.1, 0.1, 15.77, 0.23]], dtype=torch.float64
    )
    cpu = CSTLinear(
        chart, chart, atoms=p, kernel=kernel, dtype=torch.float64, backend="factored"
    )
    gpu = copy.deepcopy(cpu).cuda()
    x = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    cx = x.detach().cuda().requires_grad_()
    dy = torch.randn_like(x)
    expected = cpu(x)
    actual = Dispatcher(registry=registry()).run(gpu, LinearInputs(cx), plan=variant)
    cg = torch.autograd.grad(expected, (x, cpu.atoms.p), dy)
    gg = torch.autograd.grad(actual, (cx, gpu.atoms.p), dy.cuda())
    for value, target in zip((actual, *gg), (expected, *cg)):
        torch.testing.assert_close(value.cpu(), target, atol=2e-9, rtol=2e-8)
    radial = (gg[1][:, :2] * gpu.atoms.p[:, :2]).sum(1)
    torch.testing.assert_close(radial, torch.zeros_like(radial), atol=2e-10, rtol=0)


@GPU
@pytest.mark.parametrize("variant", VARIANTS)
def test_hybrid_nondefault_stream_temporary_lifetimes(variant):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
        radius_linear,
    )

    a = axis_plan()
    plan = HierarchyPlan(input=a, output=a)
    stream = torch.cuda.Stream()
    pending = []
    for seed in range(4):
        rng = torch.Generator().manual_seed(seed)
        x = torch.randn(3, 16, dtype=torch.float64, generator=rng, requires_grad=True)
        p = atoms(plan).requires_grad_()
        expected = reference(plan, x, p)
        expected_gradients = torch.autograd.grad(expected.sum(), (x, p))
        with torch.cuda.stream(stream):
            cx, cp = (
                x.detach().cuda().requires_grad_(),
                p.detach().cuda().requires_grad_(),
            )
            y = radius_linear(plan, cx, cp, recipe=variant.recipe)
            gradients = torch.autograd.grad(y.sum(), (cx, cp))
            pending.append((y, gradients, expected, expected_gradients))
            del cx, cp
            # Reuse allocator bins while work on the route's streams is outstanding.
            scratch = torch.empty(8192, device="cuda", dtype=torch.float64)
            scratch.fill_(seed)
            del scratch
    stream.synchronize()
    for y, gradients, expected, expected_gradients in pending:
        torch.testing.assert_close(y.cpu(), expected, atol=2e-9, rtol=2e-8)
        for gradient, target in zip(gradients, expected_gradients):
            torch.testing.assert_close(gradient.cpu(), target, atol=2e-9, rtol=2e-8)


@GPU
def test_compact_storage_omits_low_banks_and_all_direct_has_no_moments_or_coeffs():
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
        _extension,
        build_moments,
        prepare_layout,
    )

    a = axis_plan()
    plan = HierarchyPlan(input=a, output=a)
    x = torch.randn(3, 16, device="cuda", dtype=torch.float64)
    p = atoms(plan).cuda()
    full = build_moments(x, a)
    compact = build_moments(x, a, start_level=2)
    assert compact.shape == (3, a.active_nodes(2), a.basis_count)
    torch.testing.assert_close(
        compact, full[:, a.prefix_nodes(2) :], atol=3e-11, rtol=3e-11
    )
    start = len(a.banks) // 2
    empty = build_moments(x, a, start_level=start)
    assert empty.shape == (3, 0, a.basis_count) and empty.numel() == 0
    native = _extension()
    rows, offsets = prepare_layout(plan, p)
    h = native.direct_response(x, p, a.native_config(), False, start)
    c = native.compact_coefficients(p, rows, offsets, h, a.native_config(), True, start)
    assert c.shape == empty.shape and c.numel() == 0
    y = native.expand_hybrid(p, rows, offsets, h, c, x, a.native_config(), True, start)
    torch.testing.assert_close(
        y.cpu(), reference(plan, x.cpu(), p.cpu()), atol=2e-9, rtol=2e-8
    )
