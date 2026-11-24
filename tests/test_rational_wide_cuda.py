"""Scope and all first-order gradients for the public wide rational plans."""

from dataclasses import replace

import pytest
import torch
from dispatch_cases import fixed_plan

from torchcst import (
    ChebyshevSpec,
    CSTLinear,
    FixedSelector,
    chart_presets,
    kernels,
    make_algorithm_registry,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.rational_wide import (
    RATIONAL_WIDE_AXIS_GRADIENT,
    RATIONAL_WIDE_FOURIER_DIRECT,
    RATIONAL_WIDE_FOURIER_WEIGHT,
    RATIONAL_WIDE_ORDERED,
    RATIONAL_WIDE_REVERSE_ORDERED,
    RATIONAL_WIDE_SAVED,
    RATIONAL_WIDE_SCALE_COHORTS,
    RATIONAL_WIDE_SCALE_WEIGHT,
    RATIONAL_WIDE_SHARED,
    RationalWideOrderedAlgorithm,
    RationalWideReverseOrderedAlgorithm,
    RationalWideSavedAlgorithm,
    RationalWideSharedAlgorithm,
)
from torchcst._backends.schema import DeviceInfo, PrecisionPolicy
from torchcst._backends.torch.algorithms.linear.plans import TORCH_FACTORED_LINEAR
from torchcst.operators.context import LinearContext

PLANS = (RATIONAL_WIDE_FOURIER_WEIGHT, RATIONAL_WIDE_FOURIER_DIRECT)


def layer(width, atoms, *, dtype=torch.float32, coordinates=None):
    grid = chart_presets.regular_grid(width, geometry="flat_torus")
    return CSTLinear(
        charts=(grid, grid),
        atoms=atoms,
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8)),
            coordinates=coordinates or kernels.rational_radius(radius_max=512.0),
        ),
        dtype=dtype,
        selector=fixed_plan(TORCH_FACTORED_LINEAR),
    )


def test_wide_rational_plans_are_explicit_and_check_metadata():
    width, atoms = 4096, 131072
    op = layer(width, atoms).execution_declaration()
    context = LinearContext(
        operator=op,
        input_shape=(256, width),
        input_strides=(width, 1),
        dtype=torch.float32,
        atom_count=atoms,
        parameter_dim=4,
        device=DeviceInfo("cuda", 0),
        precision=PrecisionPolicy(autocast=False, allow_tf32=False),
    )
    registry = make_algorithm_registry()
    registry.register(RationalWideSharedAlgorithm())
    for plan in (*PLANS, RATIONAL_WIDE_SHARED):
        assert registry.load_plan(registry.dump_plan(plan)) == plan
        algo = registry.validate_plan(plan)
        assert algo.supports(context, plan.recipe).supported
        for changed in (
            replace(context, atom_count=65536),
            replace(context, dtype=torch.float64),
            replace(context, device=DeviceInfo("cpu")),
            replace(context, precision=PrecisionPolicy(allow_tf32=True)),
            replace(context, precision=PrecisionPolicy(autocast=True)),
            replace(context, execution_mode="cuda_graph"),
            replace(context, input_shape=(257, width)),
        ):
            assert not algo.supports(changed, plan.recipe).supported
        if plan == RATIONAL_WIDE_SHARED:
            assert not algo.supports(
                replace(context, atom_count=4096 * 48), plan.recipe
            ).supported


@pytest.mark.parametrize(("width", "coefficient"), RATIONAL_WIDE_SCALE_COHORTS[1:])
def test_wide_scaling_separates_public_and_research_plans(width, coefficient):
    from benchmarks.cuda.linear.manifest import REGISTRY

    atoms = width * coefficient
    op = layer(width, atoms).execution_declaration()
    context = LinearContext(
        operator=op,
        input_shape=(128, width),
        input_strides=(width, 1),
        dtype=torch.float32,
        atom_count=atoms,
        parameter_dim=4,
        device=DeviceInfo("cuda", 0),
        precision=PrecisionPolicy(autocast=False, allow_tf32=False),
    )
    public = make_algorithm_registry()
    direct = RATIONAL_WIDE_FOURIER_DIRECT
    assert not public.validate_plan(direct).supports(context, direct.recipe).supported
    for plan in (
        RATIONAL_WIDE_FOURIER_WEIGHT,
        RATIONAL_WIDE_SCALE_WEIGHT,
        RATIONAL_WIDE_AXIS_GRADIENT,
    ):
        assert public.load_plan(public.dump_plan(plan)) == plan
        assert public.validate_plan(plan).supports(context, plan.recipe).supported
    for plan in (
        RATIONAL_WIDE_SCALE_WEIGHT,
        RATIONAL_WIDE_SHARED,
        RATIONAL_WIDE_SAVED,
        RATIONAL_WIDE_ORDERED,
        RATIONAL_WIDE_REVERSE_ORDERED,
        RATIONAL_WIDE_AXIS_GRADIENT,
    ):
        algo = REGISTRY.validate_plan(plan)
        assert REGISTRY.load_plan(REGISTRY.dump_plan(plan)) == plan
        assert algo.supports(context, plan.recipe).supported
        assert not algo.supports(
            replace(context, atom_count=atoms + 1), plan.recipe
        ).supported


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("width", "coefficient"),
    ((2048, 32), (2048, 48), (4096, 32), (4096, 48), *RATIONAL_WIDE_SCALE_COHORTS[1:]),
)
def test_wide_rational_matches_cpu_for_admitted_plans(width, coefficient):
    initial_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        _check_wide_gradients(width, coefficient)
    finally:
        torch.set_float32_matmul_precision(initial_precision)


def _check_wide_gradients(width, coefficient):
    generator = torch.Generator().manual_seed(78)
    count = 128
    model = layer(width, width * coefficient).cuda()
    cpu = layer(width, count, dtype=torch.float64)
    rows = torch.empty((count, 4))
    rows[:, 0] = torch.linspace(-0.85, 0.67, count)
    rows[:, 1] = torch.linspace(0, 1, count)
    rows[:, 2:] = torch.rand(count, 2, generator=generator) * width - width / 2
    indices = torch.randperm(model.atoms.count, generator=generator)[:count].cuda()
    with torch.no_grad():
        cpu.atoms.p.copy_(rows)
        model.atoms.p[:, 0].zero_()
        model.atoms.p.index_copy_(0, indices, rows.cuda())

    host_x, host_dy = (torch.randn(width, 3, generator=generator).T for _ in range(2))
    x = host_x.double().requires_grad_()
    ref_y = cpu(x)
    ref_dx, ref_dp = torch.autograd.grad(ref_y, (x, cpu.atoms.p), host_dy.double())
    expected = (ref_y.detach(), ref_dx, *ref_dp.unbind(1))
    registry = make_algorithm_registry()
    plans = PLANS
    if width >= 8192:
        plans = (RATIONAL_WIDE_FOURIER_WEIGHT, RATIONAL_WIDE_SCALE_WEIGHT)
    if (width, coefficient) in RATIONAL_WIDE_SCALE_COHORTS:
        registry.register(RationalWideSharedAlgorithm())
        registry.register(RationalWideSavedAlgorithm())
        registry.register(RationalWideOrderedAlgorithm())
        registry.register(RationalWideReverseOrderedAlgorithm())
        plans = (
            *plans,
            RATIONAL_WIDE_SHARED,
            RATIONAL_WIDE_SAVED,
            RATIONAL_WIDE_ORDERED,
            RATIONAL_WIDE_REVERSE_ORDERED,
            RATIONAL_WIDE_AXIS_GRADIENT,
        )
    for plan in plans:
        model.selector = FixedSelector(plan, registry=registry)
        x = host_x.cuda().requires_grad_()
        output = model(x)
        dx, dp = torch.autograd.grad(output, (x, model.atoms.p), host_dy.cuda())
        actual = (output.detach(), dx, *dp[indices].unbind(1))
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a.cpu().double(), b, rtol=9e-4, atol=2e-4)
