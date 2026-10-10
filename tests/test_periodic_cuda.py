"""Independent whole-site flat periodic D2 CUDA and public-update contracts."""

import copy
import os
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    CSTOptimizer,
    FixedSelector,
    GaussianSpec,
    LinearInputs,
    NormalizationSpec,
    TriweightSpec,
    chart_presets,
    presets,
)
from torchcst._backends.cuda.algorithms.linear.periodic_product.algorithm import (
    PeriodicFactorAlgorithm,
    PeriodicMatrixAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.periodic_product.recipe import (
    PeriodicRecipe,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DeviceInfo, ExecutionPlan

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
ROUTES = ("matrix", "factor")


@pytest.fixture(autouse=True)
def ieee():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def selector(route, **settings):
    registry = Registry()
    algorithms = (PeriodicMatrixAlgorithm(), PeriodicFactorAlgorithm())
    for algorithm in algorithms:
        registry.register(algorithm)
    algorithm = algorithms[ROUTES.index(route)]
    plan = ExecutionPlan(algorithm.id, algorithm.revision, PeriodicRecipe(**settings))
    return FixedSelector(plan, registry=registry), plan, registry


def parameters(count, periods=(1.5, 2.75), origin=(-0.25, 0.125)):
    generator = torch.Generator().manual_seed(719)
    p = torch.rand(count, 4, generator=generator)
    p[:, 0] = (p[:, 0] - 0.5) * 0.6
    p[:, 1] = 1.1 + p[:, 1] * 0.35
    p[:, 2:] = p[:, 2:] * torch.tensor(periods) + torch.tensor(origin)
    if count:
        # Zero amplitude still has an amplitude cotangent; centres cross seams.
        p[0] = torch.tensor(
            [0.0, 1.3, origin[0] - 0.037, origin[1] + periods[1] - 0.029]
        )
    if count > 1:
        p[1, 2:] += torch.tensor([3 * periods[0], -2 * periods[1]])
    return p


def model(
    p,
    route=None,
    *,
    shape=(33, 65),
    periods=(1.5, 2.75),
    origin=(-0.25, 0.125),
    sigma=0.7,
    floor=1e-6,
    live=False,
    device="cpu",
    chart=None,
    **settings,
):
    bounds = (
        BandwidthBounds(minimum=0.1, birth=0.45, maximum=1.2, upper_floor=0.2)
        if live
        else BandwidthBounds(
            minimum=sigma, birth=sigma, maximum=sigma, upper_floor=sigma
        )
    )
    return CSTLinear(
        chart=chart
        if chart is not None
        else chart_presets.periodic_grid(shape, periods=periods, origin=origin),
        atoms=p,
        kernel=presets.polar_periodic_profile_product(
            profiles=(TriweightSpec(), TriweightSpec()),
            amplitude_max=1.0,
            bounds=bounds,
            w_c=0.5,
            alpha_init=0.3,
            radial_regularization=0.2,
            dormant_expansion_rate=0.02,
            normalization=NormalizationSpec(
                kind="discrete_l2", domain="operator_sites", floor=floor
            ),
        ),
        selector=None if route is None else selector(route, **settings)[0],
        backend="factored" if route is None else "auto",
        device=device,
    )


def polar(state, p):
    """Hand-written physical map; task VJP treats current width as fixed."""
    q = p[:, :2].square().sum(-1)
    amp = state.amplitude_max * p[:, 0] / q.clamp_min(torch.finfo(p.dtype).tiny).sqrt()
    alpha = ((q - 1) / 3).clamp(0, 1)
    z = (amp / state.w_c).square()
    lo = state.sigma_min_input + (state.sigma_birth_input - state.sigma_min_input) / (
        1 + state.lower_kappa * z
    )
    hi = state.sigma_min_input + (
        state.sigma_max_input - state.sigma_min_input
    ) * state.kappa / (state.kappa + z.pow(state.upper_decay_power))
    hi = torch.maximum(torch.maximum(hi, state.upper_floor_input), lo)
    sigma = (
        torch.exp((1 - alpha) * lo.log() + alpha * hi.log())
        .clamp(min=lo, max=hi)
        .detach()
    )
    return amp, sigma


def oracle_atoms(layer, p):
    """Enumerate every Cartesian atom entry, with one complete-atom floor.

    Do not use runtime profiles, wrapping, preparation or factored normalization.
    At an exact antipode choose period-residue: its derivative is the declared
    negative-half branch, rather than minimum's averaged tie derivative.
    """
    state, chart = layer.kernel, layer.chart
    amp, sigma = polar(state, p)
    axes = []
    for d, size in enumerate(chart.shape):
        period = chart.geometry.periods[d]
        site = (
            chart.origin[d]
            + torch.arange(size, dtype=p.dtype, device=p.device) * period / size
        )
        residue = torch.remainder(site[None, :] - p[:, 2 + d, None], period)
        distance = torch.where(residue < period / 2, residue, period - residue)
        axes.append((1 - (distance / sigma[:, None]).square()).clamp_min(0).pow(3))
    raw = axes[0][:, :, None] * axes[1][:, None, :]
    norm = torch.linalg.vector_norm(raw, dim=(1, 2)).clamp_min(
        state.spec.normalization.floor
    )
    return raw * (amp / norm)[:, None, None]


def oracle(layer, x, dy, p=None):
    reference = copy.deepcopy(layer).cpu().double()
    point = (layer.atoms.p if p is None else p).detach().cpu().double().requires_grad_()
    xx = x.detach().cpu().double().requires_grad_()
    yy = xx @ oracle_atoms(reference, point).sum(0).T
    gradients = torch.autograd.grad(yy, (xx, point), dy.detach().cpu().double())
    return yy, *gradients


def gate(actual, expected, tolerance=4e-4):
    for name, value, reference in zip(
        ("Y", "dX", "all dP"), actual, expected, strict=True
    ):
        value, reference = (
            value.detach().cpu().double(),
            reference.detach().cpu().double(),
        )
        assert value.shape == reference.shape, name
        assert torch.isfinite(value).all() and torch.isfinite(reference).all(), name
        if not value.numel():
            continue
        difference = value - reference
        assert float(difference.abs().max()) <= tolerance, name
        assert (
            float(difference.norm() / reference.norm().clamp_min(1e-30)) <= tolerance
        ), name


@pytest.mark.parametrize(
    "route,gemm", [("matrix", "torch"), ("matrix", "triton"), ("factor", "torch")]
)
def test_metadata_roundtrip_live_domain_support_and_rejections(route, gemm):
    _, plan, registry = selector(route, gemm=gemm)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    layer = model(parameters(17), route, gemm=gemm)
    context = layer.build_context(LinearInputs(torch.zeros(3, 65)))
    algorithm = registry.get(plan.algorithm_id, revision=plan.algorithm_revision)
    assert not algorithm.supports(context, plan.recipe).supported
    context = replace(context, device=DeviceInfo("cuda", 0))
    for batch in (1, 64):
        for mode in ("eager", "cuda_graph"):
            accepted = replace(
                context,
                input_shape=(batch, 65),
                input_strides=(130, 2),
                execution_mode=mode,
            )
            assert algorithm.supports(accepted, plan.recipe).supported
    assert algorithm.supports(replace(context, atom_count=0), plan.recipe).supported
    for fault in (
        replace(context, dtype=torch.float64),
        replace(context, input_shape=(65, 65)),
        replace(context, precision=replace(context.precision, allow_tf32=True)),
        replace(context, precision=replace(context.precision, autocast=True)),
        replace(context, parameter_dim=5),
        replace(context, atom_count=2**22 + 1),
        replace(context, deterministic=True),
    ):
        assert not algorithm.supports(fault, plan.recipe).supported
    mixed = replace(
        layer.kernel.spec,
        profiles=(
            replace(layer.kernel.spec.profiles[0], profile=GaussianSpec()),
            layer.kernel.spec.profiles[1],
        ),
    )
    assert not algorithm.supports(
        replace(context, operator=replace(context.operator, kernel=mixed)), plan.recipe
    ).supported
    # An incompatible revision is rejected before eligibility by OperatorSpec.
    with pytest.raises(ValueError):
        replace(context.operator, kernel=replace(context.operator.kernel, revision=1))
    for size, accepted in ((1, False), (2, True), (8192, True), (8193, False)):
        boundary = model(parameters(1), shape=(size, 65))
        boundary_context = boundary.build_context(LinearInputs(torch.zeros(1, 65)))
        boundary_context = replace(boundary_context, device=DeviceInfo("cuda", 0))
        assert algorithm.supports(boundary_context, plan.recipe).supported == accepted


@pytest.mark.parametrize(
    "settings",
    [
        {"gemm": "tf32"},
        {"prep_group": True},
        {"prep_group": 2},
        {"prep_sites": 4},
        {"atom_group": 2},
        {"patch_sites": 4},
    ],
)
def test_recipe_strict_fields(settings):
    with pytest.raises((ValueError, TypeError)):
        PeriodicRecipe(**settings)


def test_factor_rejects_gemm_recipe_instead_of_silently_ignoring_it():
    with pytest.raises(ValueError, match="GEMM"):
        selector("factor", gemm="triton")


def test_declarations_do_not_import_execution_or_triton():
    code = """
import sys
from torchcst._backends.cuda.algorithms.linear.periodic_product.algorithm import PeriodicMatrixAlgorithm, PeriodicFactorAlgorithm
from torchcst._backends.cuda.algorithms.linear.periodic_product.recipe import PeriodicRecipe
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
r=Registry()
for a in (PeriodicMatrixAlgorithm(), PeriodicFactorAlgorithm()):
    r.register(a)
    p=ExecutionPlan(a.id,a.revision,PeriodicRecipe())
    assert r.loads_plan(r.dumps_plan(p))==p
assert 'triton' not in sys.modules
assert not any(n.endswith(('periodic_product.executor','periodic_product.kernels')) for n in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )


@pytest.mark.parametrize(
    "sigma,center,floor", [(0.01, 0.5, 1e-6), (0.5, 0.25, 1e-6), (3.0, 0.25, 10.0)]
)
def test_cpu_full_cartesian_oracle_empty_singleton_and_joint_floor(
    sigma, center, floor
):
    layer = model(
        torch.tensor([[0.2, 1.2, center, center]], dtype=torch.float64),
        shape=(2, 2),
        periods=(2.0, 2.0),
        origin=(0.0, 0.0),
        sigma=sigma,
        floor=floor,
    )
    point = layer.atoms.p.detach().clone().requires_grad_()
    expected = oracle_atoms(layer, point)
    actual = layer.materialized_atoms()
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    cotangent = torch.tensor([[[0.3, -0.7], [0.2, 0.4]]], dtype=torch.float64)
    torch.testing.assert_close(
        torch.autograd.grad(actual, layer.atoms.p, cotangent)[0],
        torch.autograd.grad(expected, point, cotangent)[0],
        rtol=2e-10,
        atol=2e-11,
    )
    if sigma == 0.01:
        assert not expected.count_nonzero()
    if sigma == 0.5:
        assert expected.count_nonzero() == 1


@GPU
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"prep_group": 8, "prep_sites": 32, "atom_group": 4, "patch_sites": 16},
        {"prep_group": 1, "prep_sites": 8, "atom_group": 1, "patch_sites": 32},
    ],
)
@pytest.mark.parametrize(
    "route,batch,count,sigma,gemm",
    [
        (route, *case)
        for route in ROUTES
        for case in [
            (1, 17, 0.7, "torch"),
            (64, 9, 0.7, "torch"),
            (3, 129, 0.7, "torch"),
            (3, 17, 4.0, "torch"),
        ]
    ]
    + [("matrix", 3, 17, 0.7, "triton")],
)
def test_full_values_strides_seams_broad_and_all_four_cotangents(
    route, settings, batch, count, sigma, gemm
):
    layer = model(
        parameters(count), route, sigma=sigma, device="cuda", gemm=gemm, **settings
    )
    generator = torch.Generator().manual_seed(431)
    x = (
        torch.randn(batch, 130, generator=generator)
        .cuda()[:, ::2]
        .detach()
        .requires_grad_()
    )
    dy = torch.randn(33, batch, generator=generator).cuda().T
    y = layer(x)
    gate((y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)), oracle(layer, x, dy))


@GPU
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("case", ["empty", "singleton", "below", "equal", "above"])
def test_whole_atom_floor_and_singleton_support_derivatives(route, case):
    # Both profiles at the only active site equal (3/4)^3 exactly. Their
    # product, (3/4)^6, gives an exactly representable equality probe.
    norm = 0.421875**2
    floor = {"below": norm * 2, "equal": norm, "above": norm / 2}.get(case, 1e-6)
    layer = model(
        torch.tensor([[0.2, 1.2, 0.25, 0.25]]),
        route,
        shape=(2, 2),
        periods=(2.0, 2.0),
        origin=(0.0, 0.0),
        sigma=0.01 if case == "empty" else 0.5,
        floor=floor,
        device="cuda",
    )
    x = torch.tensor([[0.3, -0.5]], device="cuda", requires_grad=True)
    dy = torch.tensor([[0.2, -0.7]], device="cuda")
    y = layer(x)
    gradients = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    gate((y, *gradients), oracle(layer, x, dy))
    if case == "empty":
        assert not y.count_nonzero() and not gradients[1].count_nonzero()
    if case in ("singleton", "equal", "above"):
        torch.testing.assert_close(
            gradients[1][:, 2:],
            torch.zeros_like(gradients[1][:, 2:]),
            rtol=0,
            atol=2e-6,
        )
    if case == "below":
        assert torch.count_nonzero(gradients[1][:, 2:]) == 2


@GPU
@pytest.mark.parametrize("route", ROUTES)
def test_negative_half_cut_locus_vjp_and_period_translations(route):
    layer = model(
        torch.tensor([[0.2, 1.2, 0.0, 0.0]]),
        route,
        shape=(2, 4),
        periods=(2.0, 4.0),
        origin=(0.0, 0.0),
        sigma=3.0,
        device="cuda",
    )
    x = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device="cuda", requires_grad=True)
    dy = torch.tensor([[0.0, 1.0]], device="cuda")
    y = layer(x)
    first = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    expected = oracle(layer, x, dy)
    gate((y, *first), expected)
    assert float(expected[2][0, 2]) < 0 and float(expected[2][0, 3]) < 0
    with torch.no_grad():
        layer.atoms.p[:, 2:] += layer.atoms.p.new_tensor([6.0, -8.0])
    moved = layer(x)
    gate((moved, *torch.autograd.grad(moved, (x, layer.atoms.p), dy)), expected)


@GPU
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("count", [0, 17])
@pytest.mark.parametrize("need_x,need_p", [(True, False), (False, True), (True, True)])
def test_zero_atoms_and_requested_gradient_branches(route, count, need_x, need_p):
    layer = model(parameters(count), route, device="cuda")
    layer.atoms.p.requires_grad_(need_p)
    x = torch.randn(2, 65, device="cuda", requires_grad=need_x)
    dy = torch.randn(2, 33, device="cuda")
    y = layer(x)
    expected = oracle(layer, x, dy)
    requested = ([x] if need_x else []) + ([layer.atoms.p] if need_p else [])
    gradients = iter(torch.autograd.grad(y, requested, dy))
    dx = next(gradients) if need_x else expected[1]
    dp = next(gradients) if need_p else expected[2]
    gate((y, dx, dp), expected)


@GPU
@pytest.mark.parametrize("route", ROUTES)
def test_two_outstanding_forwards_own_retained_parameter_scalar_and_lattice_snapshots(
    route,
):
    layer = model(parameters(17), route, live=True, device="cuda")
    x1 = torch.randn(3, 65, device="cuda", requires_grad=True)
    x2 = torch.randn(2, 65, device="cuda", requires_grad=True)
    dy1, dy2 = torch.randn(3, 33, device="cuda"), torch.randn(2, 33, device="cuda")
    original = oracle(layer, x1, dy1)
    y1 = layer(x1)
    first = torch.autograd.grad(y1, (x1, layer.atoms.p), dy1, retain_graph=True)
    gate((y1, *first), original)
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.05)
        layer.atoms.p[:, 2:].add_(0.03)
        layer.kernel.amplitude_max.mul_(0.8)
        for name in ("sigma_max_input", "sigma_max_output"):
            layer.kernel.scalar(name).mul_(0.9)
        layer.chart.origin.add_(0.07)
        layer.chart.geometry.periods.mul_(1.1)
    y2 = layer(x2)
    gate(
        (y2, *torch.autograd.grad(y2, (x2, layer.atoms.p), dy2)), oracle(layer, x2, dy2)
    )
    retained = torch.autograd.grad(y1, (x1, layer.atoms.p), dy1)
    gate((y1, *retained), original)


@GPU
def test_factor_does_not_allocate_or_save_full_axis_factors_or_weights(monkeypatch):
    layer = model(parameters(129), "factor", device="cuda")
    x = torch.randn(3, 65, device="cuda", requires_grad=True)
    allocations, saved = [], []
    for name in ("empty", "zeros", "ones"):
        factory = getattr(torch, name)

        def record(*args, _factory=factory, **kwargs):
            result = _factory(*args, **kwargs)
            allocations.append(tuple(result.shape))
            return result

        monkeypatch.setattr(torch, name, record)

    def save(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        layer(x).sum().backward()
    forbidden = {
        (129, 65),
        (65, 129),
        (129, 33),
        (33, 129),
        (33, 65),
        (65, 33),
        (129, 33, 65),
    }
    assert not forbidden.intersection(allocations)
    assert not forbidden.intersection(saved)


@GPU
@pytest.mark.parametrize("route", ROUTES)
def test_twenty_eager_public_adamw_updates_match_reference_and_live_width(route):
    candidate = model(parameters(9), route, shape=(17, 19), live=True, device="cuda")
    reference = model(
        candidate.atoms.p.detach().clone(), shape=(17, 19), live=True, device="cuda"
    )
    opts = [
        CSTOptimizer(torch.optim.AdamW(m.parameters(), lr=1e-4, foreach=False), model=m)
        for m in (candidate, reference)
    ]
    xs = [torch.randn(3, 19, device="cuda") for _ in range(1)]
    xs = [xs[0].detach().clone().requires_grad_() for _ in range(2)]
    dy = torch.randn(3, 17, device="cuda")
    before = polar(
        copy.deepcopy(candidate.kernel).cpu().double(),
        candidate.atoms.p.detach().cpu().double(),
    )[1]
    for update in range(20):
        for m, x, opt in zip((candidate, reference), xs, opts, strict=True):
            opt.zero_grad(set_to_none=True)
            x.grad = None
            (m(x) * dy).sum().backward()
            opt.step()
        torch.testing.assert_close(
            candidate.atoms.p, reference.atoms.p, rtol=0, atol=2e-6
        )
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                opts[0].state[candidate.atoms.p][key],
                opts[1].state[reference.atoms.p][key],
                rtol=0,
                atol=4e-4,
            )
        assert int(opts[0].state[candidate.atoms.p]["step"].item()) == update + 1
    after = polar(
        copy.deepcopy(candidate.kernel).cpu().double(),
        candidate.atoms.p.detach().cpu().double(),
    )[1]
    assert not torch.equal(before, after)
    y = candidate(xs[0])
    gate(
        (y, *torch.autograd.grad(y, (xs[0], candidate.atoms.p), dy)),
        oracle(candidate, xs[0], dy),
    )


@GPU
@pytest.mark.parametrize("route", ROUTES)
def test_twenty_graph_replays_same_cotangent_public_clock_moments_and_live_sigma(
    route, *, public_fused=False
):
    from torchcst._backends.torch.algorithms.polar_update.executor import graph_update

    p = parameters(9, periods=(1.5, 2.75), origin=(0.0, 0.0))
    candidate = model(
        p, route, shape=(17, 19), origin=(0.0, 0.0), live=True, device="cuda"
    )
    reference = model(
        p.clone(), shape=(17, 19), origin=(0.0, 0.0), live=True, device="cuda"
    )
    opt = torch.optim.AdamW(
        candidate.parameters(), lr=0.01, weight_decay=0.01, fused=True, capturable=True
    )
    public = CSTOptimizer(
        torch.optim.AdamW(
            reference.parameters(),
            lr=0.01,
            weight_decay=0.01,
            foreach=False,
            fused=public_fused,
        ),
        model=reference,
    )
    x = torch.randn(3, 19, device="cuda", requires_grad=True)
    dy = torch.randn(3, 17, device="cuda")
    constant = (
        torch.tensor([[0.3, -0.2, -0.9, 0.7]], device="cuda")
        .expand_as(candidate.atoms.p)
        .clone()
    )
    before = polar(
        copy.deepcopy(candidate.kernel).cpu().double(),
        candidate.atoms.p.detach().cpu().double(),
    )[1]

    def step():
        opt.zero_grad(set_to_none=False)
        if x.grad is not None:
            x.grad.zero_()
        y = candidate(x)
        (y * dy).sum().backward()
        task_dx, task_dp = x.grad.clone(), candidate.atoms.p.grad.clone()
        old = candidate.atoms.p.detach().clone()
        candidate.atoms.p.grad.copy_(constant)
        opt.step()
        with torch.no_grad():
            proposed = graph_update(
                candidate.kernel, old, candidate.atoms.p - old, step_size=0.01
            )
            candidate.atoms.p.copy_(
                torch.cat(
                    (
                        proposed[:, :2],
                        torch.remainder(
                            proposed[:, 2:], candidate.chart.geometry.periods
                        ),
                    ),
                    -1,
                )
            )
        return y, old, task_dx, task_dp

    def eager_update():
        reference.atoms.p.grad = constant.clone()
        public.step()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        eager_update()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y, old, dx, dp = step()
    for replay in range(20):
        graph.replay()
        torch.cuda.synchronize()
        eager_update()
        gate((y, dx, dp), oracle(candidate, x, dy, p=old))
        torch.testing.assert_close(
            candidate.atoms.p, reference.atoms.p, rtol=0, atol=2e-6
        )
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                opt.state[candidate.atoms.p][key],
                public.state[reference.atoms.p][key],
                rtol=0,
                atol=2e-6,
            )
        # Recording contributes no update: two warmups plus twenty replays.
        actual_clock = int(opt.state[candidate.atoms.p]["step"].item())
        public_clock = int(public.state[reference.atoms.p]["step"].item())
        assert actual_clock == public_clock == replay + 3
    after = polar(
        copy.deepcopy(candidate.kernel).cpu().double(),
        candidate.atoms.p.detach().cpu().double(),
    )[1]
    assert not torch.equal(before, after)
