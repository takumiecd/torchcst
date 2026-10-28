"""Opt-in CUDA radius stage: independent masses, hierarchy and complete VJPs."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_radius_cell_mass import axis, oracle_mass

from torchcst import (
    ChebyshevRadiusSpec,
    CSTLinear,
    Dispatcher,
    LinearInputs,
    RadiusBounds,
    chart_presets,
    presets,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    RadiusCellMassAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    AxisPlan,
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    DIRECT,
    REFERENCE,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.recipe import (
    RadiusCellMassRecipe,
)
from torchcst._backends.registry import Registry
from torchcst._backends.torch.kernels.radius_cell_mass import axis_masses

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


def axis_plan(n=16, *, modes=8, merge=2, spacing=1.0, origin=0.0):
    return AxisPlan(
        site_count=n,
        base_tile_sites=1,
        modes=modes,
        merge_factor=merge,
        spacing=spacing,
        origin=origin,
    )


def grid(a):
    return axis(
        a.site_count,
        modes=a.modes,
        merge=a.merge_factor,
        spacing=a.spacing,
        origin=a.origin,
    )


def registry():
    result = Registry()
    result.register(RadiusCellMassAlgorithm())
    return result


@pytest.mark.parametrize("n,merge", [(16, 2), (27, 3), (25, 5), (128, 2)])
def test_banks_include_both_padded_periods_at_every_level(n, merge):
    a = axis_plan(n, merge=merge, modes=3, spacing=0.37, origin=-1.2)
    offset = 0
    for level in range(len(a.banks) // 2):
        for k in (2, 3):
            bank = a.banks[2 * level + k - 2]
            tile = merge**level
            assert (
                bank.level,
                bank.tile_sites,
                bank.tile_count,
                bank.window_tiles,
            ) == (level, tile, n // tile, k)
            assert bank.period == k * tile * a.spacing
            assert bank.node_offset == offset
            offset += n // tile
    assert a.node_count == offset
    assert a.banks[-1].period > a.period  # P is never capped to the chart Q.
    assert a.basis_count == 7


@pytest.mark.parametrize(
    "changes",
    [
        {"site_count": 129},
        {"modes": 0},
        {"modes": 17},
        {"modes": True},
        {"merge_factor": 1},
        {"merge_factor": 3},
        {"base_tile_sites": 3},
        {"spacing": 0},
        {"origin": float("nan")},
    ],
)
def test_axis_rejections(changes):
    with pytest.raises(ValueError):
        replace(axis_plan(), **changes)


def test_recipe_registry_roundtrip_and_rejections():
    reg = registry()
    for plan in (REFERENCE, DIRECT):
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for recipe in (
        RadiusCellMassRecipe("fft"),
        RadiusCellMassRecipe(save_response=False),
        RadiusCellMassRecipe(save_response=1),
    ):
        with pytest.raises(ValueError):
            reg.validate_plan(replace(REFERENCE, recipe=recipe))
    with pytest.raises(ValueError, match="unknown"):
        reg.validate_plan(replace(REFERENCE, algorithm_revision="missing"))


def reference(plan, x, p):
    inputs = axis_masses(grid(plan.input), p[:, 1], p[:, 3])
    outputs = axis_masses(grid(plan.output), p[:, 1], p[:, 2])
    return (x.double() @ inputs.T) @ (outputs * p[:, :1])


def quadrature_weight(plan, p):
    result = np.zeros((plan.output.site_count, plan.input.site_count))
    for amplitude, radius, output_center, input_center in p:
        result += amplitude * np.outer(
            oracle_mass(grid(plan.output), radius, output_center),
            oracle_mass(grid(plan.input), radius, input_center),
        )
    return torch.from_numpy(result)


def atoms(plan, dtype=torch.float64):
    # Both K=2 and K=3, every hierarchy level, seam crossing and P>Q.
    maximum = min(plan.input.period, plan.output.period) / 2
    radii = torch.tensor([0.5, 0.63, 1.0, 1.3, 2.7, 4.7, 0.81, maximum], dtype=dtype)
    radii[:-1] *= min(plan.input.spacing, plan.output.spacing)
    p = torch.empty(8, 4, dtype=dtype)
    p[:, 0] = torch.linspace(-0.4, 0.5, 8)
    p[:, 1] = radii
    p[:, 2] = plan.output.origin + torch.linspace(-0.23, plan.output.period + 0.17, 8)
    p[:, 3] = plan.input.origin + torch.linspace(plan.input.period - 0.11, 0.31, 8)
    return p


@GPU
@pytest.mark.parametrize("n,merge,modes", [(16, 2, 8), (27, 3, 3), (128, 2, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_y_dx_all_physical_columns_against_independent_mass(n, merge, modes, dtype):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
        radius_linear,
    )

    a = axis_plan(n, modes=modes, merge=merge)
    plan = HierarchyPlan(input=a, output=a)
    p = atoms(plan, dtype).requires_grad_()
    rng = torch.Generator().manual_seed(731)
    x = torch.randn(3, n, dtype=dtype, generator=rng, requires_grad=True)
    dy = torch.randn(3, n, dtype=dtype, generator=rng)
    expected = reference(plan, x, p)
    gx, gp = torch.autograd.grad(expected, (x, p), dy.double())
    weight = quadrature_weight(plan, p.detach().double().numpy())
    cuda_x = x.detach().cuda().requires_grad_()
    cuda_p = p.detach().cuda().requires_grad_()
    actual = radius_linear(plan, cuda_x, cuda_p)
    actual_gx, actual_gp = torch.autograd.grad(
        actual, (cuda_x, cuda_p), dy.cuda(), retain_graph=True
    )
    tol = (
        {"atol": 3e-5, "rtol": 3e-4}
        if dtype == torch.float32
        else {"atol": 2e-9, "rtol": 2e-8}
    )
    torch.testing.assert_close(
        actual.cpu().double(), x.detach().double() @ weight.T, **tol
    )
    torch.testing.assert_close(actual_gx.cpu().double(), dy.double() @ weight, **tol)
    torch.testing.assert_close(actual_gp.cpu(), gp, **tol)
    torch.testing.assert_close(actual_gx.cpu(), gx, **tol)
    again = torch.autograd.grad(actual, (cuda_x, cuda_p), dy.cuda())
    for first, second in zip((actual_gx, actual_gp), again):
        torch.testing.assert_close(first, second, atol=0, rtol=0)


@GPU
def test_asymmetric_axes_and_direct_merge_moments():
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
        build_moments,
        radius_linear,
    )

    plan = HierarchyPlan(
        input=axis_plan(27, merge=3, modes=3, spacing=0.7, origin=-1.1),
        output=axis_plan(16, modes=8, spacing=1.3, origin=2.1),
    )
    x = torch.randn(3, 27, device="cuda", dtype=torch.float64)
    p = atoms(plan).cuda()
    direct = build_moments(x, plan.input, method="direct")
    merged = build_moments(x, plan.input, method="merge")
    torch.testing.assert_close(direct, merged, atol=2e-11, rtol=2e-11)
    ys = [
        radius_linear(plan, x, p, recipe=r) for r in (REFERENCE.recipe, DIRECT.recipe)
    ]
    expected = reference(plan, x.cpu(), p.cpu())
    for y in ys:
        torch.testing.assert_close(y.cpu(), expected, atol=2e-9, rtol=2e-8)


@GPU
@pytest.mark.parametrize("batch,count", [(0, 8), (3, 0)])
def test_low_level_empty_shapes(batch, count):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
        radius_linear,
    )

    a = axis_plan()
    plan = HierarchyPlan(input=a, output=a)
    x = torch.randn(batch, 16, dtype=torch.float64, device="cuda", requires_grad=True)
    p = atoms(plan)[:count].cuda().requires_grad_()
    y = radius_linear(plan, x, p)
    assert y.shape == (batch, 16)
    gx, gp = torch.autograd.grad(y.sum(), (x, p))
    assert torch.equal(gx, torch.zeros_like(gx))
    assert torch.equal(gp, torch.zeros_like(gp))


@GPU
@pytest.mark.parametrize("polar", [False, True])
def test_public_cpu_initialization_explicit_cuda_plan_and_polar_task_gradient(polar):
    chart = chart_presets.regular_grid(((16,),), spacing=1.0, geometry="flat_torus")
    if polar:
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
    else:
        kernel = presets.radius_cell_mass(profile=ChebyshevRadiusSpec(modes=3))
        p = torch.tensor(
            [[0.3, 0.63, 2.17, 3.31], [-0.2, 1.3, 15.77, 0.23]], dtype=torch.float64
        )
    cpu = CSTLinear(
        chart, chart, atoms=p, kernel=kernel, dtype=torch.float64, backend="factored"
    )
    gpu = copy.deepcopy(cpu).to("cuda")
    x = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    cx = x.detach().cuda().requires_grad_()
    dy = torch.randn_like(x)
    expected = cpu(x)
    actual = Dispatcher(registry=registry()).run(gpu, LinearInputs(cx), plan=REFERENCE)
    cg = torch.autograd.grad(expected, (x, cpu.atoms.p), dy)
    gg = torch.autograd.grad(actual, (cx, gpu.atoms.p), dy.cuda())
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-9, rtol=2e-8)
    for first, second in zip(cg, gg):
        torch.testing.assert_close(second.cpu(), first, atol=2e-9, rtol=2e-8)
    if polar:
        radial = (gg[1][:, :2] * gpu.atoms.p[:, :2]).sum(1)
        torch.testing.assert_close(radial, torch.zeros_like(radial), atol=2e-10, rtol=0)


@GPU
def test_m8_half_site_radius_one_hot_and_torus_seam():
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
        radius_linear,
    )

    a = axis_plan(modes=8)
    plan = HierarchyPlan(input=a, output=a)
    p = torch.tensor(
        [[1.0, 0.5, 0.0, 0.0], [-0.3, 0.5, 15.91, -0.13]], dtype=torch.float64
    )
    x = torch.eye(16, dtype=torch.float64, device="cuda")
    actual = radius_linear(plan, x, p.cuda())
    weight = quadrature_weight(plan, p.numpy())
    torch.testing.assert_close(actual.cpu(), weight.T, atol=2e-9, rtol=2e-8)
    mass = oracle_mass(grid(a), 0.5, 0.0)
    assert (
        mass[0] > 0.99999
    )  # Retain the sharp profile; no width or normalization floor.


@GPU
@pytest.mark.parametrize("modes", [1, 3, 8, 16])
@pytest.mark.parametrize("n,merge", [(16, 2), (27, 3)])
def test_direct_merge_moment_banks_all_degrees(modes, n, merge):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
        build_moments,
    )

    signal = torch.randn(3, n, dtype=torch.float64, device="cuda")
    a = axis_plan(n, modes=modes, merge=merge, spacing=0.37, origin=1.2)
    direct = build_moments(signal, a, method="direct")
    merge = build_moments(signal, a, method="merge")
    torch.testing.assert_close(direct, merge, atol=3e-11, rtol=3e-11)


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_128_five_percent_atoms_all_gradients(dtype):
    """A=819 is the requested 5% density; this is a gate, not a speed claim."""
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
        radius_linear,
    )

    rng = torch.Generator().manual_seed(9107)
    a = axis_plan(128, modes=8)
    plan = HierarchyPlan(input=a, output=a)
    count = int(128 * 128 * 0.05)
    p = torch.empty(count, 4, dtype=dtype)
    p[:, 0] = torch.randn(count, dtype=dtype, generator=rng) * 0.1
    p[:, 1] = torch.logspace(-0.301, 1.806, count, dtype=dtype)
    p[:, 2:] = torch.rand(count, 2, dtype=dtype, generator=rng) * 128
    p.requires_grad_()
    x = torch.randn(4, 128, dtype=dtype, generator=rng, requires_grad=True)
    dy = torch.randn(4, 128, dtype=dtype, generator=rng)
    expected = reference(plan, x, p)
    gx, gp = torch.autograd.grad(expected, (x, p), dy.double())
    cx = x.detach().cuda().requires_grad_()
    cp = p.detach().cuda().requires_grad_()
    actual = radius_linear(plan, cx, cp)
    agx, agp = torch.autograd.grad(actual, (cx, cp), dy.cuda())
    tolerance = (
        {"atol": 3e-5, "rtol": 3e-4}
        if dtype == torch.float32
        else {"atol": 2e-9, "rtol": 2e-8}
    )
    for actual_value, expected_value in ((actual, expected), (agx, gx), (agp, gp)):
        torch.testing.assert_close(
            actual_value.cpu().double(), expected_value.double(), **tolerance
        )


def test_reference_helper_shape_and_autograd_on_cpu():
    plan = HierarchyPlan(input=axis_plan(27, merge=3), output=axis_plan())
    p = atoms(plan).requires_grad_()
    x = torch.randn(3, 27, dtype=torch.float64, requires_grad=True)
    y = reference(plan, x, p)
    assert y.shape == (3, 16)
    torch.testing.assert_close(
        y, x @ quadrature_weight(plan, p.detach().numpy()).T, atol=2e-11, rtol=2e-10
    )
    gx, gp = torch.autograd.grad(y.square().sum(), (x, p))
    assert gx.shape == x.shape and gp.shape == p.shape
    assert torch.isfinite(gx).all() and torch.isfinite(gp).all()


@GPU
def test_public_integer_initialization_then_cuda_fixed_selector():
    from torchcst._backends.dispatch.select import FixedSelector

    chart = chart_presets.regular_grid(((16,),), spacing=1.0, geometry="flat_torus")
    cpu = CSTLinear(
        chart,
        chart,
        atoms=8,
        kernel=presets.radius_cell_mass(),
        dtype=torch.float64,
        backend="factored",
    )
    gpu = copy.deepcopy(cpu).to("cuda")
    gpu.selector = FixedSelector(REFERENCE, registry=registry())
    x = torch.randn(3, 16, dtype=torch.float64)
    torch.testing.assert_close(gpu(x.cuda()).cpu(), cpu(x), atol=2e-9, rtol=2e-8)


@GPU
@pytest.mark.parametrize("value", [0.0, -1.0, 8.1, float("nan"), float("inf")])
def test_invalid_low_level_radius_rejected(value):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
        radius_linear,
    )

    a = axis_plan()
    plan = HierarchyPlan(input=a, output=a)
    p = atoms(plan).cuda()
    p[0, 1] = value
    with pytest.raises(ValueError):
        radius_linear(plan, torch.ones(3, 16, dtype=torch.float64, device="cuda"), p)


@GPU
def test_low_level_autocast_and_deterministic_rejected():
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
        radius_linear,
    )

    a = axis_plan()
    plan = HierarchyPlan(input=a, output=a)
    x, p = torch.ones(3, 16, device="cuda"), atoms(plan, torch.float32).cuda()
    with torch.autocast("cuda"), pytest.raises(ValueError, match="autocast"):
        radius_linear(plan, x, p)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(ValueError, match="deterministic"):
            radius_linear(plan, x, p)
    finally:
        torch.use_deterministic_algorithms(previous)
