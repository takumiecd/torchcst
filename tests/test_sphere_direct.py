"""Independent contracts for CUDA-Core Sphere contractions without W or dW."""

import math
import os
import subprocess
import sys
from dataclasses import fields, replace

import pytest
import torch

from benchmarks.cuda.linear.sphere_baseline import error, fixture, oracle_vjp
from torchcst._backends.cuda.algorithms.linear.sphere_polar.direct_algorithm import (
    SphereDirectAlgorithm,
    SphereDirectRecipe,
)
from torchcst._backends.dispatch import Dispatcher
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DeviceInfo, ExecutionPlan
from torchcst.operators.context import context_from_tensors
from torchcst.operators.execution import LinearBinding, LinearInputs

VARIANTS = ((1, False), (1, True), (4, False), (4, True))
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def recipe(options=(4, True), **kwargs):
    group, merge = options
    return SphereDirectRecipe(atom_group=group, merge_output_vjp=merge, **kwargs)


def binding(model, options=(4, True), **kwargs):
    algorithm = SphereDirectAlgorithm()
    registry = Registry()
    registry.register(algorithm)
    live = LinearBinding(model.operator, model.atoms.p)
    plan = ExecutionPlan(algorithm.id, algorithm.revision, recipe(options, **kwargs))
    dispatch = Dispatcher(registry=registry)
    return lambda x: dispatch.run(live, LinearInputs(x), plan=plan)


def gate(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert a.shape == b.shape
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        if a.numel():
            metrics = error(a, b)
            assert metrics["max_abs"] <= 4e-4, metrics
            assert metrics["relative_l2"] <= 4e-4, metrics


@pytest.fixture(autouse=True)
def ieee():
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = before


@pytest.mark.parametrize(
    "raw_scale,at_floor", [(0.0, False), (0.005, False), (0.03, False), (0.005, True)]
)
def test_merged_output_projection_is_independent_autograd_vjp(raw_scale, at_floor):
    """G and the two 3-vector sums are sufficient, including active norm floor."""
    dtype = torch.float64
    gap = torch.tensor([raw_scale, raw_scale * 0.7, raw_scale * 0.5], dtype=dtype)
    sites = torch.zeros(3, 3, dtype=dtype)
    sites[:, 0] = (1 - gap).sqrt()
    q = torch.zeros(3, dtype=dtype, requires_grad=True)
    delta = sites - q
    positive = (1 - delta.square().sum(-1)).clamp_min(0)
    raw = positive.pow(3)
    norm = raw.norm()
    floor = float(norm.detach()) if at_floor else 1e-6
    denominator = norm.clamp_min(floor)
    phi = raw / denominator
    h = torch.tensor([0.7, -0.2, 1.3], dtype=dtype)
    dy = torch.tensor(
        [[0.2, 0.9, -0.1], [-0.3, 0.5, 0.8], [0.4, -0.7, 0.6]], dtype=dtype
    )
    amp = 0.4
    truth = torch.autograd.grad((amp * h[:, None] * phi * dy).sum(), q)[0]
    g = (dy * phi).sum(-1)
    dot = (h * g).sum()
    tangent = 6 * positive[:, None].square() * delta / denominator
    unnormalized = amp * (tangent * (dy * h[:, None]).sum(0)[:, None]).sum(0)
    normalizer = (phi[:, None] * tangent).sum(0)
    merged = unnormalized - torch.where(
        norm >= floor, amp * dot * normalizer, torch.zeros_like(normalizer)
    )
    torch.testing.assert_close(merged, truth, atol=2e-12, rtol=2e-12)
    if at_floor:
        assert float(norm.detach()) == floor
    else:
        assert bool(norm < floor) == (raw_scale < 0.01)


def test_recipe_roundtrip_and_strict_metadata():
    registry = Registry()
    algorithm = SphereDirectAlgorithm()
    registry.register(algorithm)
    for options in VARIANTS:
        r = recipe(options)
        algorithm.validate_recipe(r)
        plan = ExecutionPlan(algorithm.id, algorithm.revision, r)
        assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    assert algorithm.workspace_bound(None, recipe()) is None
    for name, value in (
        ("support_capacity", 16),
        ("support_tile", 32),
        ("atom_group", 8),
        ("index_bits", 32),
        ("merge_output_vjp", 1),
    ):
        with pytest.raises(ValueError):
            SphereDirectRecipe(**{name: value})
    for field in fields(recipe()):
        if field.name != "merge_output_vjp":
            with pytest.raises(ValueError):
                SphereDirectRecipe(**{field.name: True})
    with pytest.raises(TypeError):
        algorithm.validate_recipe(object())
    model, x, *_ = fixture(17, 3.0, atoms=9)
    context = replace(
        context_from_tensors(model.declaration(), x, model.atoms.p),
        device=DeviceInfo("cuda", 0),
    )
    assert algorithm.supports(context, recipe()).supported
    for bad in (
        replace(context, device=DeviceInfo("cpu", None)),
        replace(context, dtype=torch.float64),
        replace(context, deterministic=True),
        replace(context, parameter_dim=5),
        replace(context, precision=replace(context.precision, allow_tf32=True)),
        replace(context, precision=replace(context.precision, autocast=True)),
    ):
        assert not algorithm.supports(bad, recipe()).supported
    chart = context.operator.charts[0]
    too_large = replace(
        chart, shape=(32769,), coordinates=(chart.coordinates[0],) * 32769
    )
    wrong = replace(
        context,
        input_shape=(context.input_shape[0], 32769),
        input_strides=(32769, 1),
        operator=replace(
            context.operator,
            layout=replace(context.operator.layout, input_chart=too_large),
        ),
    )
    assert any(
        "32768" in reason for reason in algorithm.supports(wrong, recipe()).reasons
    )
    for geometry in (
        replace(chart.geometry, revision=2),
        replace(chart.geometry, representation="ambient"),
    ):
        wrong = replace(
            context.operator,
            layout=replace(
                context.operator.layout, input_chart=replace(chart, geometry=geometry)
            ),
        )
        assert not algorithm.supports(
            replace(context, operator=wrong), recipe()
        ).supported
    profile = context.operator.kernel.profiles[0]
    kernel = replace(
        context.operator.kernel,
        profiles=(
            replace(profile, normalization=replace(profile.normalization, floor=2e-6)),
            context.operator.kernel.profiles[1],
        ),
    )
    wrong = replace(context, operator=replace(context.operator, kernel=kernel))
    assert not algorithm.supports(wrong, recipe()).supported


def test_declaration_import_is_lazy():
    code = """
import sys
from torchcst._backends.cuda.algorithms.linear.sphere_polar.direct_algorithm import SphereDirectAlgorithm, SphereDirectRecipe
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
algorithm = SphereDirectAlgorithm()
registry = Registry()
registry.register(algorithm)
plan = ExecutionPlan(algorithm.id, algorithm.revision, SphereDirectRecipe())
assert registry.loads_plan(registry.dumps_plan(plan)) == plan
assert 'triton' not in sys.modules
assert not any(name.endswith(('.direct_executor', '.direct_kernels')) for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )


@cuda
@pytest.mark.parametrize("options", VARIANTS)
@pytest.mark.parametrize(
    "size,sigma,atoms", [(17, 1.0, 9), (64, 3.0, 19), (128, 16.0, 9)]
)
@pytest.mark.parametrize("need_x,need_p", [(True, True), (True, False), (False, True)])
def test_full_fp64_all_six_and_requested_gradients(
    options, size, sigma, atoms, need_x, need_p
):
    model, x, _, dy = fixture(size, sigma, batch=3, atoms=atoms)
    model, x, dy = model.cuda(), x.cuda().requires_grad_(need_x), dy.cuda()
    expected = oracle_vjp(model, x, dy)
    model.atoms.p.requires_grad_(need_p)
    y = binding(model, options)(x)
    targets = tuple(t for t, needed in ((x, need_x), (model.atoms.p, need_p)) if needed)
    truth = tuple(
        t for t, needed in ((expected[1], need_x), (expected[2], need_p)) if needed
    )
    gate((y, *torch.autograd.grad(y, targets, dy)), (expected[0], *truth))


@cuda
@pytest.mark.parametrize("atoms", [0, 1, 3, 4, 5])
def test_empty_atoms_and_group_tails(atoms):
    model, x, _, dy = fixture(17, 1.0, batch=3, atoms=max(atoms, 1))
    if not atoms:
        model.atoms.p = torch.nn.Parameter(model.atoms.p[:0].clone())
    assert model.atom_count == atoms
    model, x, dy = model.cuda(), x.cuda().requires_grad_(), dy.cuda()
    y = binding(model)(x)
    actual = (y, *torch.autograd.grad(y, (x, model.atoms.p), dy))
    if atoms:
        gate(actual, oracle_vjp(model, x, dy))
    else:
        assert actual[2].shape == (0, 6)
        assert all(not torch.count_nonzero(value) for value in actual)


@cuda
@pytest.mark.parametrize("options", VARIANTS)
@pytest.mark.parametrize("gap", [0.0, 2**-9, 2**-8, 2**-6, 1.0])
def test_singleton_empty_and_both_norm_floor_branches(options, gap):
    from torchcst import (
        BandwidthBounds,
        CSTLinear,
        TriweightSpec,
        chart_presets,
        geometry_presets,
        presets,
    )

    radius = 1.0
    angle = math.acos(1 - (1 - gap) / 2)
    site = (math.cos(angle), math.sin(angle), 0.0) if gap else (-1.0, 0.0, 0.0)
    chart = chart_presets.points(
        [site],
        geometry=geometry_presets.sphere(2, radius=radius, representation="intrinsic"),
    )
    kernel = presets.polar_activity(
        amplitude_max=1.0,
        w_c=1e6,
        input_bounds=BandwidthBounds(
            minimum=1.0, birth=1.0, maximum=1.0, upper_floor=1.0
        ),
        profile=presets.profile(TriweightSpec()),
    )
    model = CSTLinear(
        chart,
        chart,
        atoms=torch.tensor([[0.3, 0.8, 0.0, 0.0, 0.0, 0.0]]),
        kernel=kernel,
        backend="factored",
    ).cuda()
    x = torch.tensor([[0.7], [-0.2]], device="cuda", requires_grad=True)
    dy = torch.tensor([[0.9], [0.3]], device="cuda")
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.fused_prepare import (
        prepare,
    )

    packed = prepare(
        x,
        model.atoms.p,
        model.kernel,
        model.cst_charts(),
        recipe(options),
        index_dtype=torch.int16,
    )
    for side in packed[5]:
        assert int(side[7][0]) == (1 if gap else 0)
        assert bool(side[6][0] < 1e-6) == (gap < 0.01)
    expected = oracle_vjp(model, x, dy)
    y = binding(model, options)(x)
    gate((y, *torch.autograd.grad(y, (x, model.atoms.p), dy)), expected)


@cuda
@pytest.mark.parametrize("options", VARIANTS)
def test_rectangular_strides_asymmetric_widths_and_antipodes(options):
    from torchcst import BandwidthBounds, CSTLinear

    a, *_ = fixture(17, 3.0, atoms=9)
    b, *_ = fixture(23, 3.0, atoms=9)
    parameterization = replace(
        a.kernel.spec.parameterization,
        output_bounds=BandwidthBounds(
            minimum=2.0, birth=2.0, maximum=8.0, upper_floor=2.0
        ),
    )
    model = CSTLinear(
        a.cst_charts()[0],
        b.cst_charts()[1],
        atoms=a.atoms.p.detach().clone(),
        kernel=replace(a.kernel.spec, parameterization=parameterization),
        backend="factored",
    ).cuda()
    with torch.no_grad():
        for col, chart in zip((2, 4), model.cst_charts()):
            model.atoms.p[0, col : col + 2].zero_()
            model.atoms.p[1, col : col + 2].zero_()
            model.atoms.p[1, col] = float(chart.geometry.radius) * (math.pi - 0.002)
    x = torch.randn(4, 34, device="cuda")[:, ::2].requires_grad_()
    dy = torch.randn(4, 46, device="cuda")[:, ::2]
    y = binding(model, options)(x)
    gate((y, *torch.autograd.grad(y, (x, model.atoms.p), dy)), oracle_vjp(model, x, dy))


@cuda
@pytest.mark.parametrize("options", VARIANTS)
def test_retained_forward_owns_live_geometry_and_scalars(options):
    model, x, _, dy = fixture(64, 3.0, batch=3, atoms=9)
    model, x, dy = model.cuda(), x.cuda().requires_grad_(), dy.cuda()
    expected = oracle_vjp(model, x, dy)
    call = binding(model, options)
    y = call(x)
    first = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
    gate((y, *first), expected)
    with torch.no_grad():
        model.atoms.p[:, :2].mul_(1.04)
        model.atoms.p[:, 2:].add_(0.001)
        model.kernel.scalar("amplitude_max").mul_(0.8)
        model.kernel.scalar("sigma_max_input").mul_(0.9)
        for chart in model.cst_charts():
            chart.coordinates.copy_(chart.coordinates.roll(1, dims=0))
            chart.geometry.radius.mul_(0.9)
    newer = call(x)
    gate(
        (newer, *torch.autograd.grad(newer, (x, model.atoms.p), dy)),
        oracle_vjp(model, x, dy),
    )
    after = torch.autograd.grad(y, (x, model.atoms.p), dy)
    for aa, bb in zip(first, after):
        torch.testing.assert_close(aa, bb, atol=0, rtol=0)
    gate((y, *after), expected)


@cuda
@pytest.mark.parametrize("options", VARIANTS)
def test_graph_replays_live_widths_geometry_and_full_gradients(options):
    model, x, _, _ = fixture(64, 3.0, batch=3, atoms=9)
    model, x = model.cuda(), x.cuda().requires_grad_()
    call = binding(model, options)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            x.grad, model.atoms.p.grad = None, None
            call(x).sum().backward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    x.grad, model.atoms.p.grad = None, None
    with torch.cuda.graph(graph, stream=stream):
        y = call(x)
        y.sum().backward()
    for scale in (0.96, 1.08):
        with torch.no_grad():
            model.atoms.p[:, :2].mul_(scale)
            model.atoms.p[:, 2:].add_(0.01)
            model.kernel.scalar("amplitude_max").mul_(scale)
            model.kernel.scalar("sigma_max_output").mul_(scale)
            for chart in model.cst_charts():
                chart.coordinates.copy_(chart.coordinates.roll(1, dims=0))
                chart.geometry.radius.mul_(scale)
        graph.replay()
        torch.cuda.synchronize()
        gate((y, x.grad, model.atoms.p.grad), oracle_vjp(model, x, torch.ones_like(y)))


@cuda
def test_direct_allocations_do_not_materialize_weight_or_weight_gradient():
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoDenseWeight(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))

            def inspect(value):
                if isinstance(value, torch.Tensor):
                    assert value.shape != (17, 17), "W/dW-sized tensor materialized"
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        inspect(item)

            inspect(result)
            return result

    model, x, _, dy = fixture(17, 3.0, batch=3, atoms=9)
    model, x, dy = model.cuda(), x.cuda().requires_grad_(), dy.cuda()
    call = binding(model)
    # Compile/warm outside the dispatch probe; runtime allocations remain checked.
    warm = call(x)
    torch.autograd.grad(warm, (x, model.atoms.p), dy)
    with NoDenseWeight():
        y = call(x)
        actual = (y, *torch.autograd.grad(y, (x, model.atoms.p), dy))
    gate(actual, oracle_vjp(model, x, dy))


@cuda
@pytest.mark.parametrize("options", VARIANTS)
def test_twenty_public_updates_keep_parameters_moments_and_exact_clock(options):
    import copy

    from torchcst import CSTOptimizer

    base, x, target, _ = fixture(32, 3.0, batch=3, atoms=9)
    models = [copy.deepcopy(base).cuda() for _ in range(2)]
    optimizers = [
        CSTOptimizer(
            torch.optim.AdamW(
                m.parameters(), lr=1e-4, weight_decay=0.01, foreach=False
            ),
            model=m,
        )
        for m in models
    ]
    xs = [x.cuda().requires_grad_() for _ in models]
    target = target.cuda()
    candidate = binding(models[1], options)
    for index in range(20):
        losses = []
        for model, xx, opt, call in zip(models, xs, optimizers, (models[0], candidate)):
            opt.zero_grad(set_to_none=True)
            xx.grad = None
            loss = (call(xx) - target).square().mean()
            losses.append(loss.detach())
            loss.backward()
            opt.step()
        torch.testing.assert_close(losses[0], losses[1], atol=4e-4, rtol=4e-4)
        torch.testing.assert_close(
            models[0].atoms.p, models[1].atoms.p, atol=2e-6, rtol=0
        )
        torch.testing.assert_close(xs[0].grad, xs[1].grad, atol=4e-4, rtol=4e-4)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizers[0].state[models[0].atoms.p][key],
                optimizers[1].state[models[1].atoms.p][key],
                atol=4e-4,
                rtol=0,
            )
        counters = [opt.state[m.atoms.p]["step"] for opt, m in zip(optimizers, models)]
        assert all(float(counter) == index + 1 for counter in counters)


@cuda
@pytest.mark.parametrize("options", VARIANTS)
@pytest.mark.parametrize("need_x,need_p", [(True, True), (True, False), (False, True)])
def test_one_group_mixes_all_complete_overflow_paths_and_signed_amplitudes(
    options, need_x, need_p
):
    from torchcst import (
        BandwidthBounds,
        CSTLinear,
        TriweightSpec,
        chart_presets,
        geometry_presets,
        presets,
    )
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.fused_prepare import (
        prepare,
    )

    radius = 4.0
    north, south = [radius, 0.0, 0.0], [-radius, 0.0, 0.0]
    charts = [
        chart_presets.points(
            sites,
            geometry=geometry_presets.sphere(
                2, radius=radius, representation="intrinsic"
            ),
        )
        for sites in ([north] * 80 + [south] * 48, [north] * 48 + [south] * 80)
    ]
    kernel = presets.polar_activity(
        amplitude_max=1.0,
        w_c=1e6,
        input_bounds=BandwidthBounds(
            minimum=1.0, birth=1.0, maximum=1.0, upper_floor=1.0
        ),
        profile=presets.profile(TriweightSpec()),
    )
    p = torch.zeros(9, 6)
    p[:, 0], p[:, 1] = 0.3, 0.8
    far = radius * (math.pi - 0.02)
    for atom, (isouth, osouth) in enumerate(
        [(False, False), (True, True), (False, True), (True, False)] * 2
        + [(False, False)]
    ):
        p[atom, 2] = far if isouth else 0.0
        p[atom, 4] = far if osouth else 0.0
    p[1, 0] = -0.3
    p[2, 0] = 0.0
    model = CSTLinear(*charts, atoms=p, kernel=kernel, backend="factored").cuda()
    gen = torch.Generator().manual_seed(147)
    x = torch.randn(3, 128, generator=gen).cuda().requires_grad_(need_x)
    dy = torch.randn(3, 128, generator=gen).cuda()
    packed = prepare(
        x,
        model.atoms.p,
        model.kernel,
        model.cst_charts(),
        recipe(options),
        index_dtype=torch.int16,
    )
    assert packed[5][0][7].tolist() == [80, 48, 80, 48, 80, 48, 80, 48, 80]
    assert packed[5][1][7].tolist() == [48, 80, 80, 48, 48, 80, 80, 48, 48]
    assert float(packed[2][1]) < 0 and float(packed[2][2]) == 0
    expected = oracle_vjp(model, x, dy)
    model.atoms.p.requires_grad_(need_p)
    y = binding(model, options)(x)
    targets = tuple(t for t, needed in ((x, need_x), (model.atoms.p, need_p)) if needed)
    truth = tuple(
        t for t, needed in ((expected[1], need_x), (expected[2], need_p)) if needed
    )
    gate((y, *torch.autograd.grad(y, targets, dy)), (expected[0], *truth))


@cuda
def test_direct_block_widens_maximum_signed_site_id_before_addressing():
    import triton
    import triton.language as tl

    from torchcst._backends.cuda.algorithms.linear.sphere_polar.direct_kernels import (
        _block,
    )

    @triton.jit
    def probe(S, Q, P, I, F, N, C, Out):
        a = tl.arange(0, 4)
        live = a < 1
        count = tl.load(C + a, live, 0)
        idx, _, phi, gap, _, _, _, _ = _block(
            S, Q, P, I, F, N, a, live, count, 0, 32768, 64, 16, False, 1e-6
        )
        t = tl.arange(0, 16)
        tl.store(Out + a[:, None] * 16 + t[None, :], idx)
        tl.store(Out + 64 + a[:, None] * 16 + t[None, :], phi * gap)

    sites = torch.zeros(32768, 3, device="cuda")
    sites[-1, 0] = 2.0
    q = torch.tensor([[2.0, 0.0, 0.0]], device="cuda")
    precision = torch.ones(1, device="cuda")
    index = torch.zeros(1, 64, device="cuda", dtype=torch.int16)
    index[0, 0] = 32767
    phi = torch.ones(1, 64, device="cuda")
    norm = torch.ones(1, device="cuda")
    count = torch.ones(1, device="cuda", dtype=torch.int32)
    output = torch.empty(128, device="cuda")
    probe[(1,)](
        sites,
        q,
        precision,
        index,
        phi,
        norm,
        count,
        output,
        num_warps=4,
        enable_fp_fusion=False,
    )
    assert float(output[0]) == 32767 and float(output[64]) == 1.0


def ptx_arithmetic_report(ptx):
    import re

    instructions = [line.strip() for line in ptx.splitlines()]
    fp32 = {
        op: [
            line
            for line in instructions
            if re.search(r"\b" + op + r"(?:\.[A-Za-z0-9_]+)*\.f32\b", line)
        ][:8]
        for op in ("mul", "add")
    }
    return {
        "mma_present": bool(re.search(r"\b(?:mma|wmma)\.", ptx)),
        "tf32_present": bool(re.search(r"\.tf32\b", ptx)),
        "fp32_instruction_samples": fp32,
    }


def test_compiler_report_opcode_parser_accepts_modifiers_and_finds_tensor_core_ops():
    report = ptx_arithmetic_report(
        "mul.rn.f32 %f0, %f1, %f2;\nadd.ftz.rn.f32 %f0, %f1, %f2;"
    )
    assert all(report["fp32_instruction_samples"].values())
    assert not report["mma_present"] and not report["tf32_present"]
    assert ptx_arithmetic_report("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32;")[
        "mma_present"
    ]
    assert ptx_arithmetic_report("wmma.mma.sync.aligned.row.col.f32.f32;")[
        "mma_present"
    ]
    assert ptx_arithmetic_report("cvt.rna.tf32.f32 %r0, %f0;")["tf32_present"]


@cuda
@pytest.mark.parametrize("options", VARIANTS)
def test_l4_compiled_direct_uses_fp32_cuda_core_instructions(
    options, monkeypatch, tmp_path
):
    import hashlib
    import json
    from pathlib import Path

    from torchcst._backends.cuda.algorithms.linear.sphere_polar import direct_kernels

    if torch.cuda.get_device_name(0) != "NVIDIA L4":
        pytest.skip("compiler resource diagnostic requires NVIDIA L4")
    captured = {}

    class CaptureCompiled:
        def __init__(self, kernel, name):
            self.kernel, self.name = kernel, name

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def run(*args, **kwargs):
                compiled = launch(*args, **kwargs)
                captured[self.name] = compiled
                return compiled

            return run

    for name in ("forward", "backward"):
        monkeypatch.setattr(
            direct_kernels, name, CaptureCompiled(getattr(direct_kernels, name), name)
        )
    model, x, _, dy = fixture(2048, 3.0, batch=32, atoms=int(0.05 * 2048**2))
    model, x, dy = model.cuda(), x.cuda().requires_grad_(), dy.cuda()
    y = binding(model, options)(x)
    gradients = torch.autograd.grad(y, (x, model.atoms.p), dy)
    torch.cuda.synchronize()
    assert all(torch.isfinite(value).all() for value in (y, *gradients))
    records = {}
    folder = Path(os.environ.get("CST_JOB_OUTPUT", str(tmp_path)))
    folder.mkdir(parents=True, exist_ok=True)
    group, merge = options
    stem = f"compiler-direct-g{group}-merge{int(merge)}"
    for name, compiled in captured.items():
        ptx = compiled.asm["ptx"]
        ptx_path = folder / f"{stem}-{name}.ptx"
        ptx_path.write_text(ptx)
        records[name] = {
            "n_regs": compiled.n_regs,
            "n_spills": compiled.n_spills,
            "shared_bytes": compiled.metadata.shared,
            "ptx_file": ptx_path.name,
            "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
            **ptx_arithmetic_report(ptx),
        }
    report = {
        "scope": "separate compiled-resource diagnostic; not a timing/adoption gate",
        "gpu": torch.cuda.get_device_name(0),
        "size": 2048,
        "batch": 32,
        "atoms": int(0.05 * 2048**2),
        "recipe": vars(recipe(options)),
        "kernels": records,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (folder / f"{stem}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    assert set(records) == {"forward", "backward"}
    for record in records.values():
        assert (
            record["n_regs"] > 0
            and record["n_spills"] >= 0
            and record["shared_bytes"] >= 0
        )
        assert not record["mma_present"] and not record["tf32_present"]
        assert all(record["fp32_instruction_samples"].values())
