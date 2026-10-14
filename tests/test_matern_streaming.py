"""Metadata and exact same-profile regressions for bounded width streaming."""

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from benchmarks.cuda.linear import matern_mixture_product as mm
from torchcst import BandwidthBounds, FixedSelector, MaternMixtureSpec
from torchcst._backends.cuda.algorithms.linear.matern_streaming.algorithm import (
    MaternStreamingAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.matern_streaming.executor import (
    _scratch,
    static_axes,
)
from torchcst._backends.cuda.algorithms.linear.matern_streaming.recipe import (
    MaternStreamingRecipe,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DeviceInfo, ExecutionPlan
from torchcst._backends.torch.profiles.matern_moments import static_tables
from torchcst.operators.context import LinearContext

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
KNOTS = tuple(math.sqrt(10 / 2**n) for n in reversed(range(8)))


def context(knots=KNOTS):
    model = mm.fixture(mm.case_definition(knots=8), atoms=17)
    spec = model.execution_declaration()
    bindings = tuple(
        replace(binding, profile=MaternMixtureSpec(width_knots=knots))
        for binding in spec.kernel.profiles
    )
    bounds = BandwidthBounds(
        minimum=knots[0], birth=knots[0], maximum=knots[-1], upper_floor=knots[0]
    )
    parameterization = replace(
        spec.kernel.parameterization, input_bounds=bounds, output_bounds=bounds
    )
    return LinearContext(
        operator=replace(
            spec,
            kernel=replace(
                spec.kernel, profiles=bindings, parameterization=parameterization
            ),
        ),
        input_shape=(32, 1024),
        input_strides=(1024, 1),
        dtype=torch.float32,
        atom_count=52428,
        device=DeviceInfo("cuda", 0),
        parameter_dim=4,
    )


@pytest.mark.parametrize("group", [1, 2])
def test_recipe_roundtrip_support_and_bounded_scratch(group):
    recipe = MaternStreamingRecipe(width_group=group)
    algorithm = MaternStreamingAlgorithm()
    registry = Registry()
    registry.register(algorithm)
    plan = ExecutionPlan(algorithm.id, algorithm.revision, recipe)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    ctx = context()
    assert algorithm.supports(ctx, recipe).supported
    buffers = _scratch(torch.empty(()), 1024, 32, recipe, inject=True)
    actual = sum(t.numel() * t.element_size() for t in buffers)
    assert algorithm.workspace_bound(ctx, recipe) is None
    assert buffers[0].shape == (2, group, 1024, 32, 2)
    assert actual == group * 1064960


def test_power_bank_rejects_non_dyadic_and_skipped_widths():
    algorithm = MaternStreamingAlgorithm()
    generic = tuple(mm.case_definition(knots=8)["width_knots"])
    assert not algorithm.supports(context(generic), MaternStreamingRecipe()).supported
    assert algorithm.supports(
        context(generic), MaternStreamingRecipe(power_basis=False)
    ).supported
    skipped = tuple(math.sqrt(10 / 2**n) for n in (8, 6, 5, 4, 3, 2, 1, 0))
    assert not algorithm.supports(context(skipped), MaternStreamingRecipe()).supported


@pytest.mark.parametrize(
    "arguments",
    [
        {"width_group": True},
        {"width_group": 3},
        {"block_sites": 32},
        {"power_basis": 1},
    ],
)
def test_recipe_rejects_invalid_types(arguments):
    with pytest.raises(ValueError):
        MaternStreamingRecipe(**arguments)


def test_fp64_repeated_squaring_changes_only_static_step_attenuation():
    profile = MaternMixtureSpec(width_knots=KNOTS)
    binding = SimpleNamespace(profile=profile)
    constants, moments, _knots = static_axes(
        torch.empty(0), ((1024, 102.4, 0.1),), (binding,), MaternStreamingRecipe()
    )[0]
    direct, direct_moments = static_tables(profile, 1024, 102.4)
    torch.testing.assert_close(constants, direct, rtol=2e-14, atol=1e-15)
    assert constants.dtype == torch.float64
    torch.testing.assert_close(moments, direct_moments, rtol=0, atol=0)
    torch.testing.assert_close(constants[:, :4], direct[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(constants[:, 5:], direct[:, 5:], rtol=0, atol=0)


def bind(layer, group, *, power_basis=False):
    registry = Registry()
    algorithm = MaternStreamingAlgorithm()
    registry.register(algorithm)
    layer.selector = FixedSelector(
        ExecutionPlan(
            algorithm.id,
            algorithm.revision,
            MaternStreamingRecipe(width_group=group, power_basis=power_basis),
        ),
        registry=registry,
    )


@GPU
@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_generic_streamed_all_sites_vjps_and_retained_saves(group, floor):
    from benchmarks.cuda.linear.matern_native_oracle import oracle_vjp

    layer = mm.fixture(
        mm.case_definition(knots=8), atoms=17, size=128, floor=floor
    ).cuda()
    bind(layer, group)
    generator = torch.Generator().manual_seed(1151)
    x = torch.randn(3, 256, generator=generator).cuda()[:, ::2].requires_grad_()
    dy = torch.randn(3, 256, generator=generator).cuda()[:, ::2]
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected = oracle_vjp(layer, x, dy)
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
        again = torch.autograd.grad(y, (x, layer.atoms.p), dy)
        for actual, truth in zip((y, dx, dp), expected, strict=True):
            torch.testing.assert_close(
                actual.double().cpu(), truth, rtol=4e-4, atol=4e-4
            )
        for actual, truth in zip((dx, dp), again, strict=True):
            torch.testing.assert_close(actual, truth, rtol=2e-5, atol=2e-6)
        yy = layer(x)
        with torch.no_grad():
            layer.atoms.p[:, 2:].add_(0.025)
        old = torch.autograd.grad(yy, (x, layer.atoms.p), dy)
        for actual, truth in zip(old, (dx, dp), strict=True):
            torch.testing.assert_close(actual, truth, rtol=2e-5, atol=2e-6)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before


@GPU
@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_dyadic_power_all_vjps_cross_group_pairs_and_public_update_graph(group, floor):
    import copy

    from benchmarks.cuda.linear import matern_dyadic_product as md
    from benchmarks.cuda.linear.matern_native_oracle import oracle_vjp
    from benchmarks.cuda.linear.periodic_comparison import TrainingStep
    from torchcst import CSTOptimizer

    layer = md.fixture(md.case_definition(), atoms=17, size=128, floor=floor).cuda()
    bind(layer, group, power_basis=True)
    generator = torch.Generator().manual_seed(1251)
    x = torch.randn(3, 256, generator=generator).cuda()[:, ::2].requires_grad_()
    dy = torch.randn(3, 256, generator=generator).cuda()[:, ::2]
    target = torch.randn(3, 128, generator=generator).cuda()
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected = oracle_vjp(layer, x, dy)
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
        again = torch.autograd.grad(y, (x, layer.atoms.p), dy)
        for actual, truth in zip((y, dx, dp), expected, strict=True):
            torch.testing.assert_close(
                actual.double().cpu(), truth, rtol=4e-4, atol=4e-4
            )
        for actual, truth in zip((dx, dp), again, strict=True):
            torch.testing.assert_close(actual, truth, rtol=2e-5, atol=2e-6)
        yy = layer(x)
        with torch.no_grad():
            layer.atoms.p[:, 2:].add_(0.025)
        old = torch.autograd.grad(yy, (x, layer.atoms.p), dy)
        for actual, truth in zip(old, (dx, dp), strict=True):
            torch.testing.assert_close(actual, truth, rtol=2e-5, atol=2e-6)
        # 17 live widths cover all adjacent pairs, including pairs separated by
        # a width-group boundary. Graph updates must keep rebuilding weights.
        intervals = (
            torch.searchsorted(
                torch.tensor(KNOTS, dtype=torch.float64, device="cuda"),
                md.widths(layer),
                right=True,
            )
            .sub(1)
            .clamp(0, len(KNOTS) - 2)
        )
        assert set(intervals.cpu().tolist()) == set(range(7))
        layer(x)
        replica = copy.deepcopy(layer)
        step = TrainingStep(layer, x, target)
        optimizer = torch.optim.AdamW(
            replica.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            fused=True,
            capturable=False,
        )
        public = CSTOptimizer(optimizer, model=replica)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        torch.cuda.current_stream().wait_stream(stream)
        xx = x.detach().clone().requires_grad_()
        optimizer.zero_grad(set_to_none=True)
        (replica(xx) - target).square().mean().backward()
        public.step()
        width0 = md.user_widths(layer).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            xx.grad = None
            (replica(xx) - target).square().mean().backward()
            public.step()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                layer.atoms.p, replica.atoms.p, atol=3e-6, rtol=2e-6
            )
            left, right = (
                step.opt.state[layer.atoms.p],
                optimizer.state[replica.atoms.p],
            )
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(left[key], right[key], atol=2e-6, rtol=2e-5)
        assert not torch.equal(width0, md.user_widths(layer))
        expected = oracle_vjp(layer, xx, dy)
        yy = layer(xx)
        vx, vp = torch.autograd.grad(yy, (xx, layer.atoms.p), dy)
        for actual, truth in zip((yy, vx, vp), expected, strict=True):
            torch.testing.assert_close(
                actual.double().cpu(), truth, rtol=4e-4, atol=4e-4
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before
