"""Production PolarAmpWidth equivalence for local H and full-domain slicing."""

import math

import pytest
import torch

from benchmarks.cuda.linear.fixtures import local_product_reference as reference
from benchmarks.cuda.linear.fixtures import local_product_state as state
from torchcst._backends.cuda.algorithms.local_product.contract import Domain
from torchcst._backends.cuda.algorithms.local_product.preparation import (
    decode,
    validate_state,
)
from torchcst._backends.cuda.algorithms.local_product.recipe import Recipe
from torchcst._backends.torch.kernels import execution


def fixture(value, domain, *, device="cpu", dtype=torch.float64, atoms=19):
    torch.manual_seed(31)
    amplitude = torch.linspace(-0.12, 0.12, atoms, device=device, dtype=dtype)
    radius = torch.linspace(1, 2, atoms, device=device, dtype=dtype)
    p = torch.stack((amplitude, (1 - amplitude.square()).sqrt()), 1) * radius[:, None]
    center = torch.rand(atoms, 2, device=device, dtype=dtype)
    center[:, 0] = (
        domain.input_origin + center[:, 0] * (domain.input_size - 1) * domain.spacing
    )
    center[:, 1] = (
        domain.output_origin + center[:, 1] * (domain.output_size - 1) * domain.spacing
    )
    p = torch.cat((p, center), 1)
    # Exact singleton, two-site, empty, norm-floor, and zero-amplitude support.
    if atoms >= 5:
        p[:5, :2] = p.new_tensor(
            [
                [0.1, math.sqrt(0.99)],
                [-0.1, math.sqrt(0.99)],
                [0.1, math.sqrt(0.99)],
                [0.1, math.sqrt(0.99)],
                [0.0, 1.0],
            ]
        )
        p[0, 2:] = p.new_tensor(
            [
                domain.input_origin + 4 * domain.spacing,
                domain.output_origin + 5 * domain.spacing,
            ]
        )
        p[1, 2:] = p.new_tensor(
            [
                domain.input_origin + 4.5 * domain.spacing,
                domain.output_origin + 5.5 * domain.spacing,
            ]
        )
        p[2, 2] = domain.input_origin - 50 * domain.spacing
        p[3, 2:] = p.new_tensor(
            [
                domain.input_origin - 0.999 * domain.spacing,
                domain.output_origin - 0.999 * domain.spacing,
            ]
        )
    return p.requires_grad_()


def scalar_oracle(x, p, value, domain):
    q = decode(value, p)
    j = torch.arange(domain.input_size, device=p.device, dtype=p.dtype)
    i = torch.arange(domain.output_size, device=p.device, dtype=p.dtype)
    y = x.new_zeros((len(x), domain.output_count))
    for amp, inv, ci, co in q:
        v = (
            (1 - (domain.input_origin + j * domain.spacing - ci).square() * inv)
            .clamp_min(0)
            .pow(3)
        )
        u = (
            (1 - (domain.output_origin + i * domain.spacing - co).square() * inv)
            .clamp_min(0)
            .pow(3)
        )
        v = v / torch.linalg.vector_norm(v).clamp_min(1e-6)
        u = u / torch.linalg.vector_norm(u).clamp_min(1e-6)
        v = v[domain.input_start : domain.input_start + domain.input_count]
        u = u[domain.output_start : domain.output_start + domain.output_count]
        y = y + amp * (x * v[None]).sum(1)[:, None] * u[None]
    return y


def test_reference_slice_uses_full_domain_norms():
    d = Domain(
        32,
        32,
        spacing=0.5,
        input_start=5,
        output_start=3,
        input_count=9,
        output_count=11,
    )
    s = state(minimum=0.5, birth=0.5, maximum=8, w_c=0.2).double()
    p = fixture(s, d)
    x = torch.randn(7, d.input_count, dtype=p.dtype, requires_grad=True)
    charts = d.charts(dtype=p.dtype)
    actual, truth = reference(x, p, s, charts, d), scalar_oracle(x, p, s, d)
    dy = torch.randn_like(actual)
    ag, eg = (
        torch.autograd.grad(actual, (x, p), dy),
        torch.autograd.grad(truth, (x, p), dy),
    )
    for a, e in [(actual, truth), *zip(ag, eg)]:
        torch.testing.assert_close(a, e, atol=2e-11, rtol=2e-11)
    # Width is activity state: the task gradient is angular, not radial.
    assert (ag[1][:, :2] * p[:, :2]).sum(1).abs().max() < 1e-10


def test_recipe_and_shared_contract():
    validate_state(state())
    for upper in [(16.0,), (2.0, 16.0), (1.0, 2.0, 4.0, 16.0)]:
        Recipe(rho_upper=upper)
    with pytest.raises(ValueError):
        Recipe(rho_upper=(2.0, 1.0))
    with pytest.raises(ValueError):
        Domain(32, 32, input_start=25, input_count=16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("fused_polar", [False, True])
@pytest.mark.parametrize("support_only", [False, True])
@pytest.mark.parametrize(
    "batch,spacing,sliced", [(1, 1.0, False), (7, 0.5, True), (32, 1.0, True)]
)
def test_local_h_y_dx_polar_gradient_update(
    saved, fused_polar, support_only, batch, spacing, sliced
):
    from torchcst._backends.cuda.algorithms.local_product.executor import local_h

    d = Domain(
        32,
        32,
        spacing=spacing,
        input_start=3 if sliced else 0,
        output_start=5 if sliced else 0,
        input_count=17 if sliced else 32,
        output_count=19 if sliced else 32,
    )
    s = state(minimum=spacing, birth=spacing, maximum=16 * spacing, w_c=0.2).cuda()
    validate_state(s)
    p = fixture(s, d, device="cuda", dtype=torch.float32)
    x = torch.randn(batch, d.input_count, device="cuda", requires_grad=True)
    dy = torch.randn(batch, d.output_count, device="cuda")
    actual = local_h(
        x,
        p,
        s,
        d,
        saved=saved,
        fused_polar=fused_polar,
        sparse=support_only,
        support_only=support_only,
        recipe=Recipe(pack=not fused_polar),
    )
    ax, ap = torch.autograd.grad(actual, (x, p), dy)
    pp, xx = p.detach().double().requires_grad_(), x.detach().double().requires_grad_()
    ss = (
        state(minimum=spacing, birth=spacing, maximum=16 * spacing, w_c=0.2)
        .double()
        .cuda()
    )
    truth = scalar_oracle(xx, pp, ss, d)
    ex, ep = torch.autograd.grad(truth, (xx, pp), dy.double())
    for a, e in [(actual, truth), (ax, ex), (ap, ep)]:
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)
    # Preserve the actual projected polar activity update, not plain log-width SGD.
    charts = d.charts(device="cuda")
    updated = execution.apply_parameter_update(
        s, *charts, p.detach(), -1e-4 * ap, step_size=1e-4
    )
    expected = execution.apply_parameter_update(
        ss,
        *d.charts(device="cuda", dtype=torch.float64),
        pp.detach(),
        -1e-4 * ep,
        step_size=1e-4,
    )
    torch.testing.assert_close(updated.double(), expected, atol=4e-4, rtol=4e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("fused_polar", [False, True])
@pytest.mark.parametrize("support_only", [False, True])
def test_graph_replay_refreshes_polar_decode_and_packing(
    saved, fused_polar, support_only
):
    from torchcst._backends.cuda.algorithms.local_product.executor import local_h

    d = Domain(32, 32)
    s = state(minimum=1, birth=1, w_c=0.2).cuda()
    p = fixture(s, d, device="cuda", dtype=torch.float32)
    x = torch.randn(16, 32, device="cuda", requires_grad=True)

    def step():
        y = local_h(
            x,
            p,
            s,
            d,
            saved=saved,
            fused_polar=fused_polar,
            sparse=support_only,
            support_only=support_only,
            recipe=Recipe(pack=not fused_polar),
        )
        return y, *torch.autograd.grad(y, (x, p), torch.ones_like(y))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = step()
    with torch.no_grad():
        p[0, :2] *= 2
        p[0, 2:] = p.new_tensor([19.3, 23.2])
        s.scalar("amplitude_max").fill_(1.7)
    graph.replay()
    pp, xx = p.detach().double().requires_grad_(), x.detach().double().requires_grad_()
    ss = state(minimum=1, birth=1, w_c=0.2).double().cuda()
    ss.scalar("amplitude_max").copy_(s.scalar("amplitude_max"))
    y = scalar_oracle(xx, pp, ss, d)
    expected = (y, *torch.autograd.grad(y, (xx, pp), torch.ones_like(y)))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("saved", [False, True])
def test_fused_polar_nontrivial_envelopes_at_max_batch(saved):
    from torchcst._backends.cuda.algorithms.local_product.executor import local_h

    d = Domain(
        64,
        64,
        spacing=0.5,
        input_start=3,
        input_count=33,
        output_start=5,
        output_count=47,
    )
    s = state(minimum=0.5, birth=1.5, maximum=8, w_c=0.2).cuda()
    p = fixture(s, d, device="cuda", dtype=torch.float32, atoms=41)
    x = torch.randn(64, d.input_count, device="cuda", requires_grad=True)
    dy = torch.randn(64, d.output_count, device="cuda")
    y = local_h(x, p, s, d, saved=saved, fused_polar=True, recipe=Recipe(pack=False))
    grads = torch.autograd.grad(y, (x, p), dy)
    xx, pp = x.detach().double().requires_grad_(), p.detach().double().requires_grad_()
    ss = state(minimum=0.5, birth=1.5, maximum=8, w_c=0.2).double().cuda()
    expected = scalar_oracle(xx, pp, ss, d)
    truth_grads = torch.autograd.grad(expected, (xx, pp), dy.double())
    for a, e in [(y, expected), *zip(grads, truth_grads)]:
        torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "route", ["local", "saved", "hybrid", "hybrid_support", "support", "support_saved"]
)
@pytest.mark.parametrize("batch", [32, 64])
def test_full_128_transform_all_atom_gradients(route, batch):
    from torchcst._backends.cuda.algorithms.local_product.executor import local_h

    d = Domain(128, 128)
    s = state(birth=1).cuda()
    p = fixture(
        s, d, device="cuda", dtype=torch.float32, atoms=819 if batch == 32 else 41
    )
    x = torch.randn(batch, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    recipe = Recipe(pack=False, rho_upper=(4.0, 16.0))
    y = local_h(
        x,
        p,
        s,
        d,
        saved=route in ("saved", "support_saved"),
        hybrid=route in ("hybrid", "hybrid_support"),
        sparse=route in ("hybrid_support", "support", "support_saved"),
        support_only=route in ("support", "support_saved"),
        fused_polar=True,
        recipe=recipe,
    )
    grads = torch.autograd.grad(y, (x, p), dy)
    xx, pp = x.detach().double().requires_grad_(), p.detach().double().requires_grad_()
    expected = scalar_oracle(xx, pp, state(birth=1).double().cuda(), d)
    truth_grads = torch.autograd.grad(expected, (xx, pp), dy.double())
    for a, e in [(y, expected), *zip(grads, truth_grads)]:
        torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("limit", [2.0, 4.0, 8.0])
@pytest.mark.parametrize("sparse", [False, True])
def test_hybrid_graph_switches_both_directions_without_stale_h(limit, sparse):
    from torchcst._backends.cuda.algorithms.local_product.executor import local_h

    d = Domain(128, 128, input_start=3, input_count=73, output_start=5, output_count=97)
    s = state(birth=1).cuda()
    p = fixture(s, d, device="cuda", dtype=torch.float32, atoms=41)
    with torch.no_grad():
        p[:, :2].div_(p[:, :2].norm(dim=1, keepdim=True))
        p[16:32, :2].mul_(2)
        p[33::2, :2].mul_(2)
    recipe = Recipe(pack=False, rho_upper=(limit, 16.0))
    x = torch.randn(7, d.input_count, device="cuda", requires_grad=True)
    dy = torch.randn(7, d.output_count, device="cuda")

    def step():
        y = local_h(
            x, p, s, d, hybrid=True, sparse=sparse, fused_polar=True, recipe=recipe
        )
        return y, *torch.autograd.grad(y, (x, p), dy)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = step()
    for flip in (True, False):
        with torch.no_grad():
            p[:, :2].div_(p[:, :2].norm(dim=1, keepdim=True))
            p[: 16 if flip else 0, :2].mul_(2)
            if not flip:
                p[16:32, :2].mul_(2)
            p[33::2, :2].mul_(2)
            p[0, 2:] = p.new_tensor([23.3, 41.2])
        q = decode(s, p).detach()
        wide = q[:, 1] < (d.spacing * limit) ** -2
        assert wide[:16].all() if flip else not wide[:16].any()
        assert not wide[16:32].any() if flip else wide[16:32].all()
        graph.replay()
        xx, pp = (
            x.detach().double().requires_grad_(),
            p.detach().double().requires_grad_(),
        )
        expected = scalar_oracle(xx, pp, state(birth=1).double().cuda(), d)
        truth_grads = torch.autograd.grad(expected, (xx, pp), dy.double())
        for a, e in zip(actual, (expected, *truth_grads)):
            torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)


def test_graph_update_and_dense_baseline_match_production():
    from benchmarks.cuda.linear.fixtures import (
        local_product_dense_factors as dense_factors,
    )
    from torchcst._backends.cuda.algorithms.local_product.polar import graph_update

    d = Domain(32, 32, input_start=4, input_count=17, output_start=2, output_count=19)
    s = state(minimum=1, birth=1, w_c=0.2).double()
    p = fixture(s, d)
    displacement = torch.randn_like(p) * 0.01
    charts = d.charts(dtype=p.dtype)
    actual = graph_update(s, p, displacement, step_size=0.01)
    expected = execution.apply_parameter_update(
        s, *charts, p, displacement, step_size=0.01
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    v, u = dense_factors(p, s, d)
    x = torch.randn(7, d.input_count, dtype=p.dtype)
    torch.testing.assert_close(
        x @ v @ u.T, reference(x, p, s, charts, d), atol=1e-12, rtol=1e-12
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("saved", [False, True])
def test_max_batch_atom32_and_unsorted_preserve_gradients(saved):
    from torchcst._backends.cuda.algorithms.local_product.executor import local_h

    d = Domain(64, 64, input_start=7, output_start=3, input_count=33, output_count=47)
    s = state(minimum=1, birth=1, w_c=0.2).cuda()
    p = fixture(s, d, device="cuda", dtype=torch.float32, atoms=41)
    x = torch.randn(64, d.input_count, device="cuda", requires_grad=True)
    dy = torch.randn(64, d.output_count, device="cuda")
    actual = local_h(x, p, s, d, saved=saved, recipe=Recipe(atom_block=32, pack=False))
    ax, ap = torch.autograd.grad(actual, (x, p), dy)
    pp, xx = p.detach().double().requires_grad_(), x.detach().double().requires_grad_()
    truth = scalar_oracle(xx, pp, state(minimum=1, birth=1, w_c=0.2).double().cuda(), d)
    ex, ep = torch.autograd.grad(truth, (xx, pp), dy.double())
    for a, e in [(actual, truth), (ax, ex), (ap, ep)]:
        torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)


def test_actual_support_count_floor_and_slice():
    from torchcst._backends.cuda.algorithms.local_product.support import (
        analyze_side,
        summarize,
    )

    d = Domain(16, 16, input_start=4, input_count=4)
    # shared sigma=spacing: aligned one site vs. two between sites; empty/floor.
    q = torch.tensor(
        [
            [0.1, 1.0, 4.0, 5.0],
            [0.1, 1.0, 4.5, 5.5],
            [0.1, 1.0, -50.0, 5.0],
            [0.1, 1.0, -0.999, -0.999],
        ],
        dtype=torch.float64,
    )
    vi, uo = analyze_side(q, d, "input"), analyze_side(q, d, "output")
    assert vi.full_count.tolist() == [1, 2, 0, 1]
    assert vi.local_count.tolist() == [1, 2, 0, 0]
    assert vi.singleton_live.tolist() == [True, False, False, False]
    assert uo.singleton_live.tolist() == [True, False, True, False]
    assert vi.start.tolist() == [4, 4, 4, 4]
    assert vi.stop.tolist() == [5, 6, 4, 4]
    for upper in [(1,), (1, 2), (2, 8, 16)]:
        report = summarize(q, d, count_upper=upper)
        assert sum(report["bucket_atoms"]) == 4
        assert report["onehot_both_live_atoms"] == 1
        assert report["inactive_in_local_transform_atoms"] == 2


def test_benchmark_polar_adamw_policy_matches_cst_optimizer():
    from types import SimpleNamespace

    from benchmarks.cuda.linear.local_product import optimizer_step
    from torchcst import CSTLinear, CSTOptimizer

    d = Domain(16, 16)
    s = state(birth=1).double()
    p = fixture(s, d).detach()
    model = CSTLinear(
        *d.charts(dtype=torch.float64),
        atoms=p,
        kernel=s.spec,
        dtype=torch.float64,
        backend="factored",
    )
    actual = SimpleNamespace(p=torch.nn.Parameter(p.clone()), local_state=s)
    proposal = torch.optim.AdamW([actual.p], lr=0.0001, weight_decay=0.01)
    base = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    oracle = CSTOptimizer(base, model=model)
    gen = torch.Generator().manual_seed(95)
    for _ in range(4):
        grad = torch.randn(p.shape, generator=gen, dtype=p.dtype)
        actual.p.grad, model.atoms.p.grad = grad.clone(), grad.clone()
        optimizer_step(actual, proposal, step_size=0.0001)
        oracle.step()
        torch.testing.assert_close(actual.p, model.atoms.p, atol=1e-12, rtol=1e-12)
        for key, tensor in proposal.state[actual.p].items():
            torch.testing.assert_close(
                tensor, base.state[model.atoms.p][key], atol=1e-12, rtol=1e-12
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "plan_id",
    [
        "local-unpacked",
        "local-saved",
        "local-support",
        "local-polar",
        "local-polar-saved",
        "local-hybrid",
        "local-hybrid-support",
        "local-polar-support",
        "local-polar-support-saved",
        "hybrid-packed-mid4",
        "hybrid-persistent-mid4",
    ],
)
def test_graph_training_updates_width_and_matches_public_optimizer(
    plan_id, size_override=None, polar_update="torch", updates=4
):
    from benchmarks.cuda.linear.local_product import (
        fixture_operator,
        initialize,
        optimizer_step,
    )
    from benchmarks.cuda.linear.manifest import DEFAULT_PLANS, load_run
    from benchmarks.cuda.linear.run import PlanLinear
    from torchcst import CSTLinear, CSTOptimizer

    is_persistent = plan_id == "hybrid-persistent-mid4"
    is_packed = plan_id in ("hybrid-packed-mid4", "hybrid-persistent-mid4")
    size = size_override or (
        128 if plan_id.startswith("local-hybrid") or is_packed else 32
    )
    batch = 32 if size_override is not None or size == 128 else 16
    catalog = DEFAULT_PLANS.with_name(
        "plans-local-persistent.json"
        if is_persistent
        else "plans-local-packed.json"
        if is_packed
        else "plans-local-rho.json"
        if plan_id.startswith("local-polar-support")
        else "plans-local-hybrid-support.json"
        if plan_id.startswith("local-hybrid")
        else "plans-local-polar.json"
        if plan_id.startswith("local-polar")
        else "plans-local-support.json"
    )
    case = load_run(
        DEFAULT_PLANS.parent
        / (
            f"cases/local-size-{size}-middle.json"
            if size_override is not None
            else "cases/local-128-persistent-middle.json"
            if is_persistent
            else "cases/local-128-packed-middle.json"
            if is_packed
            else f"cases/local-{size}-mixed{'-hybrid' if size == 128 else ''}.json"
        ),
        catalog,
    )
    from benchmarks.cuda.linear.manifest import decode_catalog, read_json

    entries = decode_catalog(read_json(catalog)[0])
    plan = next(entry.plan for entry in entries if entry.id == plan_id)
    model = PlanLinear(initialize(case.case).cuda(), fixture_operator(case.case), plan)
    opt = torch.optim.AdamW(
        [model.p], lr=0.0001, weight_decay=0.01, fused=True, capturable=True
    )
    gen = torch.Generator().manual_seed(193)
    x = torch.randn(batch, size, generator=gen).cuda().requires_grad_()
    target = torch.randn(batch, size, generator=gen).cuda()

    def step():
        opt.zero_grad(set_to_none=True)
        x.grad = None
        y = model(x)
        ((y * target).sum() / x.numel()).backward()
        optimizer_step(model, opt, step_size=0.0001, polar_update=polar_update)
        return y

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_y = step()
    torch.cuda.synchronize()
    before = decode(model.local_state, model.p).detach()[:, 1].rsqrt().clone()
    # Start the public, eager optimizer from the same post-capture state/moments.
    d = Domain(size, size)
    truth = CSTLinear(
        *d.charts(device="cuda"),
        atoms=model.p.detach().clone(),
        kernel=model.local_state.spec,
        backend="factored",
    )
    # Singleton centre derivatives are mathematically zero; FP32 factored
    # normalization leaves ~1e-9 residuals amplified by Adam epsilon. The new
    # matched-size campaign uses the public FP64 oracle without relaxing checks.
    if size_override is not None:
        truth = truth.double()
    base = torch.optim.AdamW(
        truth.parameters(), lr=0.0001, weight_decay=0.01, fused=True
    )
    import copy

    base.load_state_dict(copy.deepcopy(opt.state_dict()))
    for group in base.param_groups:
        group["capturable"] = False
    oracle = CSTOptimizer(base, model=truth)
    tx = x.detach().clone().to(truth.atoms.p.dtype).requires_grad_()
    for _ in range(updates):
        oracle.zero_grad()
        tx.grad = None
        expected_y = truth(tx)
        ((expected_y * target).sum() / tx.numel()).backward()
        oracle.step()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            actual_y.to(expected_y.dtype), expected_y, atol=4e-4, rtol=4e-4
        )
        torch.testing.assert_close(
            x.grad.to(tx.grad.dtype), tx.grad, atol=4e-4, rtol=4e-4
        )
        torch.testing.assert_close(
            model.p.grad.to(truth.atoms.p.dtype),
            truth.atoms.p.grad,
            atol=4e-4,
            rtol=4e-4,
        )
        torch.testing.assert_close(
            model.p.to(truth.atoms.p.dtype), truth.atoms.p, atol=3e-6, rtol=3e-6
        )
        for key, tensor in opt.state[model.p].items():
            torch.testing.assert_close(
                tensor.to(base.state[truth.atoms.p][key].dtype),
                base.state[truth.atoms.p][key],
                atol=3e-6,
                rtol=3e-6,
            )
    after = decode(model.local_state, model.p).detach()[:, 1].rsqrt()
    assert torch.count_nonzero(after != before) > len(after) // 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("limit", [1, 2, 4, 8])
@pytest.mark.parametrize("spacing", [1.0, 0.5])
def test_support_route_handles_narrow_wide_floor_and_slices(limit, spacing):
    from torchcst._backends.cuda.algorithms.local_product.executor import local_h

    d = Domain(
        64,
        64,
        spacing=spacing,
        input_start=3,
        input_count=33,
        output_start=5,
        output_count=47,
    )
    s = state(minimum=spacing, birth=spacing, maximum=16 * spacing).cuda()
    p = fixture(s, d, device="cuda", dtype=torch.float32, atoms=41)
    with torch.no_grad():
        # Most atoms start at the minimum. Include a broad group plus a purely
        # narrow partial group so both device branches execute in the same call.
        radius = p[:, :2].square().sum(1).sqrt()
        p[:, :2].div_(radius[:, None])
        p[8, :2].mul_(2)
    x = torch.randn(64, d.input_count, device="cuda", requires_grad=True)
    dy = torch.randn(64, d.output_count, device="cuda")
    recipe = Recipe(pack=False, support_limit=limit)
    y = local_h(x, p, s, d, sparse=True, recipe=recipe)
    grads = torch.autograd.grad(y, (x, p), dy)
    xx, pp = x.detach().double().requires_grad_(), p.detach().double().requires_grad_()
    ss = state(minimum=spacing, birth=spacing, maximum=16 * spacing).double().cuda()
    expected = scalar_oracle(xx, pp, ss, d)
    truth_grads = torch.autograd.grad(expected, (xx, pp), dy.double())
    for a, e in [(y, expected), *zip(grads, truth_grads)]:
        torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_support_graph_refreshes_bounds_across_threshold():
    from torchcst._backends.cuda.algorithms.local_product.executor import local_h

    d = Domain(32, 32)
    s = state(birth=1).cuda()
    p = fixture(s, d, device="cuda", dtype=torch.float32, atoms=19)
    with torch.no_grad():
        p[:, :2].div_(p[:, :2].norm(dim=1, keepdim=True))
    x = torch.randn(7, 32, device="cuda", requires_grad=True)
    dy = torch.randn(7, 32, device="cuda")
    recipe = Recipe(pack=False, support_limit=4)

    def step():
        y = local_h(x, p, s, d, sparse=True, recipe=recipe)
        return y, *torch.autograd.grad(y, (x, p), dy)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = step()
    with torch.no_grad():
        p[:16, :2].mul_(2)
        p[0, 2:] = p.new_tensor([17.3, 21.2])
    graph.replay()
    xx, pp = x.detach().double().requires_grad_(), p.detach().double().requires_grad_()
    ss = state(birth=1).double().cuda()
    expected = scalar_oracle(xx, pp, ss, d)
    expected_grads = torch.autograd.grad(expected, (xx, pp), dy.double())
    for a, e in zip(actual, (expected, *expected_grads)):
        torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)
