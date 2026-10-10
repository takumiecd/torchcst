"""Physical FP64 contracts for fixed-threshold Sphere precision repair."""

import copy
import math
import os
import subprocess
import sys
from dataclasses import fields, replace

import pytest
import test_sphere_direct as direct
import torch

from torchcst._backends.cuda.algorithms.linear.sphere_polar.precision_algorithm import (
    SpherePrecisionAdaptiveAlgorithm,
    SpherePrecisionAdaptiveRecipe,
    SpherePrecisionCompactAlgorithm,
    SpherePrecisionCompactRecipe,
    SpherePrecisionDirectAlgorithm,
    SpherePrecisionDirectRecipe,
)
from torchcst._backends.dispatch import Dispatcher
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DeviceInfo, ExecutionPlan
from torchcst.operators.context import context_from_tensors
from torchcst.operators.execution import LinearBinding, LinearInputs

KINDS = ("compact", "direct", "adaptive")
TYPES = {
    "compact": (SpherePrecisionCompactAlgorithm, SpherePrecisionCompactRecipe),
    "direct": (SpherePrecisionDirectAlgorithm, SpherePrecisionDirectRecipe),
    "adaptive": (SpherePrecisionAdaptiveAlgorithm, SpherePrecisionAdaptiveRecipe),
}
cuda = direct.cuda


@pytest.fixture(autouse=True)
def ieee():
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = before


def bind(model, kind):
    cls, recipe_cls = TYPES[kind]
    algorithm, registry = cls(), Registry()
    registry.register(algorithm)
    live = LinearBinding(model.operator, model.atoms.p)
    plan = ExecutionPlan(algorithm.id, algorithm.revision, recipe_cls())
    dispatch = Dispatcher(registry=registry)
    return lambda x: dispatch.run(live, LinearInputs(x), plan=plan)


def reuse(monkeypatch, kind):
    monkeypatch.setattr(
        direct, "binding", lambda model, options=(4, True), **kw: bind(model, kind)
    )
    monkeypatch.setattr(
        direct, "recipe", lambda options=(4, True), **kw: TYPES[kind][1](**kw)
    )


@pytest.mark.parametrize("kind", KINDS)
def test_precision_strict_fixed_threshold_metadata_and_roundtrip(kind):
    cls, recipe_cls = TYPES[kind]
    algorithm, registry = cls(), Registry()
    registry.register(algorithm)
    recipe = recipe_cls()
    plan = ExecutionPlan(algorithm.id, algorithm.revision, recipe)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    for bad in (0.0, 1e-4, 0.0010000001, True, 1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            recipe_cls(precision_norm_threshold=bad)
    for field in fields(recipe):
        if field.name == "precision_norm_threshold":
            continue
        with pytest.raises(ValueError):
            recipe_cls(**{field.name: False if field.name == "merge_output_vjp" else 0})
    with pytest.raises(TypeError):
        algorithm.validate_recipe(object())
    model, x, *_ = direct.fixture(17, 1.25, atoms=9)
    context = replace(
        context_from_tensors(model.declaration(), x, model.atoms.p),
        device=DeviceInfo("cuda", 0),
    )
    assert algorithm.supports(context, recipe).supported
    for wrong in (
        replace(context, device=DeviceInfo("cpu", None)),
        replace(context, dtype=torch.float64),
        replace(context, deterministic=True),
        replace(context, precision=replace(context.precision, allow_tf32=True)),
    ):
        assert not algorithm.supports(wrong, recipe).supported


def test_precision_declarations_remain_lazy():
    code = """
import sys
from torchcst._backends.cuda.algorithms.linear.sphere_polar.precision_algorithm import *
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
for cls,recipe in [(SpherePrecisionCompactAlgorithm,SpherePrecisionCompactRecipe),(SpherePrecisionDirectAlgorithm,SpherePrecisionDirectRecipe),(SpherePrecisionAdaptiveAlgorithm,SpherePrecisionAdaptiveRecipe)]:
 a=cls();r=Registry();r.register(a);p=ExecutionPlan(a.id,a.revision,recipe())
 assert r.loads_plan(r.dumps_plan(p))==p
assert 'triton' not in sys.modules
assert not any(n.endswith(('.precision_executor','.precision_prepare','.precision_prepare_kernels')) for n in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )


@cuda
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("need_x,need_p", [(True, True), (True, False), (False, True)])
def test_precision_full_six_independent_requested_gradients(
    kind, need_x, need_p, monkeypatch
):
    reuse(monkeypatch, kind)
    direct.test_full_fp64_all_six_and_requested_gradients(
        (4, True), 17, 1.25, 9, need_x, need_p
    )


@cuda
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("atoms", [0, 1, 5])
def test_precision_empty_and_group_tail(kind, atoms, monkeypatch):
    reuse(monkeypatch, kind)
    direct.test_empty_atoms_and_group_tails(atoms)


@cuda
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("gap", [0.0, 2**-9, 0.02])
def test_precision_singleton_subfloor_nonzero_abovefloor_center_zero(
    kind, gap, monkeypatch
):
    reuse(monkeypatch, kind)
    direct.test_singleton_empty_and_both_norm_floor_branches((4, True), gap)
    from torchcst import (
        BandwidthBounds,
        CSTLinear,
        TriweightSpec,
        chart_presets,
        geometry_presets,
        presets,
    )

    angle = math.acos(1 - (1 - gap) / 2)
    chart = chart_presets.points(
        [[math.cos(angle), math.sin(angle), 0]],
        geometry=geometry_presets.sphere(2, radius=1, representation="intrinsic"),
    )
    kernel = presets.polar_activity(
        amplitude_max=1,
        w_c=1e6,
        input_bounds=BandwidthBounds(minimum=1, birth=1, maximum=1, upper_floor=1),
        profile=presets.profile(TriweightSpec()),
    )
    model = CSTLinear(
        chart,
        chart,
        atoms=torch.tensor([[0.3, 0.8, 0, 0, 0, 0]]),
        kernel=kernel,
        backend="factored",
    ).cuda()
    x = torch.tensor([[0.7], [-0.2]], device="cuda", requires_grad=True)
    dy = torch.tensor([[0.9], [0.3]], device="cuda")
    y = bind(model, kind)(x)
    dx, dp = torch.autograd.grad(y, (x, model.atoms.p), dy)
    truth = direct.oracle_vjp(model, x, dy)
    direct.gate((y, dx, dp), truth)
    if 0 < gap < 0.01:
        assert truth[2][:, 2:].abs().max() > 1e-3
        assert dp[:, 2:].abs().max() > 1e-3
    else:
        assert dp[:, 2:].abs().max() < 1e-9


def low_norm_model(device="cuda"):
    """Minimal two-centre witness derived from real subfloor/two-site failures.

    Every supplied site participates in the independent full-chart oracle;
    points shared across centres retain both physical support sets.
    """
    from torchcst import CSTLinear, chart_presets, geometry_presets

    radius = 9.027033805847168
    input_sites = [
        [-5.832011699676514, 4.45291805267334, -5.257993698120117],
        [-5.199273109436035, -2.968770742416382, -6.755835056304932],
        [-4.652808666229248, -3.662935256958008, -6.81334114074707],
    ]
    output_sites = [
        [-2.078908920288086, -4.4483962059021, -7.574776649475098],
        [-3.08583402633667, -4.725176811218262, -7.045400619506836],
        [-1.3212840557098389, -4.185945987701416, -7.887927532196045],
        [-2.7393617630004883, -3.38352108001709, -7.907908916473389],
        [-2.513808012008667, -3.8128108978271484, -7.786564826965332],
        [-2.9415123462677, -4.479848861694336, -7.2640061378479],
        [-2.1740071773529053, -4.244156360626221, -7.664734840393066],
        [-2.266023635864258, -4.603333473205566, -7.427098274230957],
        [4.536585330963135, 5.31317138671875, 5.71637487411499],
        [5.152344703674316, 5.2560882568359375, 5.226300239562988],
        [5.41641092300415, 4.695339679718018, 5.486676216125488],
        [5.331355571746826, 4.4845194816589355, 5.740477085113525],
        [5.4464802742004395, 4.694637298583984, 5.457432746887207],
        [4.820127487182617, 4.834166049957275, 5.906314373016357],
    ]
    p = torch.tensor(
        [
            [
                -0.024810798466205597,
                1.1139256954193115,
                15.248929977416992,
                -15.380133628845215,
                -7.932465553283691,
                -14.325637817382812,
            ],
            [
                -0.1589946448802948,
                1.1027995347976685,
                -6.316956043243408,
                -17.482229232788086,
                6.41753625869751,
                6.904547214508057,
            ],
        ]
    )
    prototype, *_ = direct.fixture(17, 1.25, atoms=2)
    charts = [
        chart_presets.points(
            s,
            geometry=geometry_presets.sphere(
                2, radius=radius, representation="intrinsic"
            ),
        )
        for s in (input_sites, output_sites)
    ]
    return CSTLinear(
        *charts, atoms=p, kernel=prototype.kernel.spec, backend="factored"
    ).to(device)


@cuda
@pytest.mark.parametrize("kind", KINDS)
def test_precision_actual_subfloor_and_two_site_regression_all_six(kind):
    model = low_norm_model()
    x = torch.tensor(
        [[0.7, -0.3, 0.2], [-0.2, 0.5, 0.8], [0.4, 0.6, -0.9]],
        device="cuda",
        requires_grad=True,
    )
    dy = torch.linspace(-0.7, 0.9, 42, device="cuda").reshape(3, 14)
    y = bind(model, kind)(x)
    direct.gate(
        (y, *torch.autograd.grad(y, (x, model.atoms.p), dy)),
        direct.oracle_vjp(model, x, dy),
    )


def test_precision_archived_low_norm_fixture_has_physical_full_chart_support():
    """The regression is a physical two-axis witness, not a selected-site oracle."""
    from benchmarks.cuda.linear.sphere_baseline import oracle_factors

    model = low_norm_model("cpu")
    factors = oracle_factors(
        model, model.atoms.p.detach().double(), scale_amplitude=False
    )
    assert [(side > 0).sum(0).tolist() for side in factors] == [[1, 2], [8, 6]]
    assert 0 < float(factors[0][:, 0].norm()) < 1
    torch.testing.assert_close(
        factors[0][:, 1].norm(), torch.ones((), dtype=torch.float64)
    )
    x = torch.tensor([[0.7, -0.3, 0.2], [-0.2, 0.5, 0.8]])
    dy = torch.linspace(-0.7, 0.9, 28).reshape(2, 14)
    truth = direct.oracle_vjp(model, x, dy)
    assert all(torch.isfinite(t).all() for t in truth)
    assert truth[2][0, 2:4].abs().max() > 1e-3
    assert truth[2][1, 2:4].abs().max() > 1e-3


@cuda
@pytest.mark.parametrize("branch", ["below", "equal", "above"])
def test_precision_physical_profile_vjp_exact_norm_floor_branch(branch):
    """Probe the shared physical profile primitive against independent autograd.

    The floor equals the immutable norm snapshot exactly for the equality case;
    equality must retain the normalization derivative, as torch.clamp_min does.
    """
    import triton
    import triton.language as tl

    from torchcst._backends.cuda.algorithms.linear.sphere_polar.precision_kernels import (
        _block,
    )

    @triton.jit
    def probe(S, Q, J, P, I, N, C, M, Out, FLOOR: tl.constexpr):
        _idx, _valid, phi, g0, g1 = _block(
            S, Q, J, P, I, N, C, M, 0, 0, 3, 64, 16, FLOOR
        )
        t = tl.arange(0, 16)
        tl.store(Out + t, phi)
        tl.store(Out + 16 + t, g0)
        tl.store(Out + 32 + t, g1)

    gap = torch.tensor([0.03, 0.021, 0.015], dtype=torch.float64)
    sites = torch.zeros(3, 3, dtype=torch.float64)
    sites[:, 0] = (1 - gap).sqrt()
    center = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    jac = torch.tensor([[1.0, 0.2], [-0.3, 0.8], [0.4, -0.2]], dtype=torch.float64)
    delta = sites - center
    positive = (1 - delta.square().sum(1)).clamp_min(0)
    raw = positive.pow(3)
    norm = raw.norm().detach()
    floor = float(norm) * {"below": 2.0, "equal": 1.0, "above": 0.5}[branch]
    denominator = raw.norm().clamp_min(floor)
    profile = raw / denominator
    derivatives = torch.stack(
        [
            torch.autograd.grad(value, center, retain_graph=True)[0] @ jac
            for value in profile
        ]
    )
    moment = (
        raw[:, None] * 6 * positive[:, None].square() * delta / denominator.square()
    ).sum(0)
    index = torch.zeros(1, 64, dtype=torch.int16)
    index[0, :3] = torch.arange(3, dtype=torch.int16)
    output = torch.empty(48, device="cuda")
    tensors = (
        sites,
        center.detach()[None],
        jac[None],
        torch.ones(1, dtype=torch.float64),
        index,
        norm[None],
        torch.tensor([3], dtype=torch.int32),
        moment.detach()[None],
    )
    probe[(1,)](*[t.cuda() for t in tensors], output, floor, enable_fp_fusion=False)
    actual = output.reshape(3, 16)
    expected = (
        torch.cat((profile.detach()[None], derivatives.detach().T), dim=0)
        .float()
        .cuda()
    )
    torch.testing.assert_close(actual[:, :3], expected, atol=4e-4, rtol=4e-4)
    assert not torch.count_nonzero(actual[:, 3:])
    if branch == "equal":
        assert float(norm) == floor


def mixed_precision_model(device="cuda"):
    """One normal and three sensitive rows, including true >64-site overflow."""
    from torchcst import (
        BandwidthBounds,
        CSTLinear,
        TriweightSpec,
        chart_presets,
        geometry_presets,
        presets,
    )

    radius = 4.0
    far = float(torch.tensor(radius * (math.pi - 0.02)))
    gap = 2**-9
    rim = far / radius - math.acos(1 - (1 - gap) / (2 * radius**2))
    sites = [[radius, 0.0, 0.0]] * 80 + [
        [radius * math.cos(rim), radius * math.sin(rim), 0.0]
    ] * 80
    chart = chart_presets.points(
        sites,
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
    p = torch.tensor(
        [
            [0.3, 0.8, 0.0, 0.0, 0.0, 0.0],
            [-0.3, 0.8, far, 0.0, far, 0.0],
            [0.0, 0.8, 0.0, 0.0, far, 0.0],
            [0.3, 0.8, radius * math.pi / 2, 0.0, radius * math.pi / 2, 0.0],
        ]
    )
    return CSTLinear(chart, chart, atoms=p, kernel=kernel, backend="factored").to(
        device
    )


@cuda
@pytest.mark.parametrize("kind", KINDS)
def test_precision_mixed_sensitive_normal_ownership_complete_overflow_all_six(kind):
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.precision_executor import (
        forward_snapshots,
    )

    model = mixed_precision_model()
    gen = torch.Generator().manual_seed(149)
    x = torch.randn(3, 160, generator=gen).cuda().requires_grad_()
    dy = torch.randn(3, 160, generator=gen).cuda()
    diagnostics = {}
    inspected, _saved, _floors = forward_snapshots(
        x,
        model.atoms.p,
        model.kernel,
        model.cst_charts(),
        TYPES[kind][1](),
        kind=kind,
        diagnostics=diagnostics,
    )
    assert diagnostics["sensitive"].tolist() == [False, True, True, True]
    assert not torch.any(
        diagnostics["remaining_normal_fallback"] & diagnostics["sensitive"]
    )
    for side, count in zip(
        diagnostics["sides"], diagnostics["original_complete_counts"], strict=True
    ):
        assert count.tolist() == [80, 80, 80, 0]
        assert side[7].tolist() == [80, 0, 0, 0]
    for refined in diagnostics["refined_sides"]:
        assert refined[6].tolist() == [0, 80, 80, 0]
        assert int(refined[6][1]) > TYPES[kind][1]().support_capacity
    y = bind(model, kind)(x)
    truth = direct.oracle_vjp(model, x, dy)
    direct.gate(
        (inspected, y, *torch.autograd.grad(y, (x, model.atoms.p), dy)),
        (truth[0], *truth),
    )


@cuda
@pytest.mark.parametrize("kind", KINDS)
def test_precision_complete_overflow_signed_and_regular_zero_amplitude(
    kind, monkeypatch
):
    reuse(monkeypatch, kind)
    direct.test_one_group_mixes_all_complete_overflow_paths_and_signed_amplitudes(
        (4, True), True, True
    )


@cuda
@pytest.mark.parametrize("kind", KINDS)
def test_precision_two_forwards_retain_live_geometry_scalars_and_widths(
    kind, monkeypatch
):
    model = low_norm_model()
    x = torch.tensor(
        [[0.7, -0.3, 0.2], [-0.2, 0.5, 0.8]], device="cuda", requires_grad=True
    )
    dy = torch.linspace(-0.7, 0.9, 28, device="cuda").reshape(2, 14)
    call = bind(model, kind)
    truth = direct.oracle_vjp(model, x, dy)
    y = call(x)
    first = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
    direct.gate((y, *first), truth)
    with torch.no_grad():
        model.atoms.p[:, 2:].add_(0.00001)
        model.kernel.scalar("amplitude_max").mul_(0.8)
        model.kernel.scalar("sigma_max_input").mul_(0.999)
        for chart in model.cst_charts():
            chart.coordinates.copy_(chart.coordinates.roll(1, 0))
            chart.geometry.radius.mul_(0.999)
    newer = call(x)
    direct.gate(
        (newer, *torch.autograd.grad(newer, (x, model.atoms.p), dy)),
        direct.oracle_vjp(model, x, dy),
    )
    after = torch.autograd.grad(y, (x, model.atoms.p), dy)
    direct.gate((after[0],), (first[0],))
    torch.testing.assert_close(after[1], first[1], atol=0, rtol=0)
    direct.gate((y, *after), truth)


@cuda
@pytest.mark.parametrize("kind", KINDS)
def test_precision_live_graph_geometry_all_gradients(kind, monkeypatch):
    reuse(monkeypatch, kind)
    direct.test_graph_replays_live_widths_geometry_and_full_gradients((4, True))


@cuda
@pytest.mark.parametrize("kind", KINDS)
def test_precision_twenty_public_updates_and_moments(kind, monkeypatch):
    reuse(monkeypatch, kind)
    direct.test_twenty_public_updates_keep_parameters_moments_and_exact_clock((4, True))


@cuda
@pytest.mark.parametrize("kind", KINDS)
def test_precision_twenty_four_complete_graph_updates_preserve_public_trajectory(kind):
    from benchmarks.cuda.linear.scaling_comparison import ProposalUpdate, capture
    from torchcst import CSTOptimizer

    model = low_norm_model()
    generator = torch.Generator().manual_seed(41)
    x = torch.randn(3, 3, generator=generator).cuda().requires_grad_()
    target = torch.randn(3, 14, generator=generator).cuda()
    dy = torch.randn(3, 14, generator=generator).cuda()
    call = bind(model, kind)
    update = ProposalUpdate(model, "sphere", capturable=True)

    def step():
        update.opt.zero_grad(set_to_none=True)
        x.grad = None
        y = call(x)
        loss = (y - target).square().mean()
        loss.backward()
        update()
        return y, loss

    graph, _ = capture(step)
    assert int(update.opt.state[model.atoms.p]["step"]) == 3
    reference = copy.deepcopy(model)
    base = torch.optim.AdamW(
        reference.parameters(), lr=1e-4, weight_decay=0.01, fused=True
    )
    base.load_state_dict(copy.deepcopy(update.opt.state_dict()))
    for group in base.param_groups:
        group["capturable"] = False
    public = CSTOptimizer(base, model=reference)
    xx = x.detach().clone().requires_grad_()
    for index in range(21):
        public.zero_grad(set_to_none=True)
        xx.grad = None
        (reference(xx) - target).square().mean().backward()
        public.step()
        graph.replay()
        torch.cuda.synchronize()
        for name, aa, bb, tol in [
            ("p", model.atoms.p, reference.atoms.p, 2e-6),
            (
                "exp_avg",
                update.opt.state[model.atoms.p]["exp_avg"],
                base.state[reference.atoms.p]["exp_avg"],
                4e-4,
            ),
            (
                "exp_avg_sq",
                update.opt.state[model.atoms.p]["exp_avg_sq"],
                base.state[reference.atoms.p]["exp_avg_sq"],
                4e-4,
            ),
        ]:
            assert torch.isfinite(aa).all() and torch.isfinite(bb).all(), name
            assert direct.error(aa, bb)["max_abs"] <= tol, name
        assert int(update.opt.state[model.atoms.p]["step"]) == index + 4
        assert int(base.state[reference.atoms.p]["step"]) == index + 4
    assert int(update.opt.state[model.atoms.p]["step"]) == 24
    y = call(x)
    direct.gate(
        (y, *torch.autograd.grad(y, (x, model.atoms.p), dy)),
        direct.oracle_vjp(model, x, dy),
    )
