"""Grouped atom tails, exact Strip support and coalesced canonical VJPs."""

import copy

import pytest
import torch

from benchmarks.cuda.linear.global_profile_product import oracle_vjp as product_oracle
from benchmarks.cuda.linear.manifest import REGISTRY
from benchmarks.cuda.linear.strip_profile_product import oracle_vjp as strip_oracle
from torchcst import (
    BandwidthBounds,
    CSTLinear,
    Dispatcher,
    LinearInputs,
    NormalizationSpec,
    TriweightSpec,
    chart_presets,
    pattern_presets,
    presets,
)
from torchcst._backends.cuda.algorithms.linear.profile_product_global.recipe_v2 import (
    GroupedProductRecipe,
    GroupedStripRecipe,
)
from torchcst._backends.schema import ExecutionPlan

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


def model(
    p, family, n=65, out=33, width=None, floor=1e-6, pitch=20, origin=0, device="cpu"
):
    axes = (
        pattern_presets.line(out, low=origin, high=origin + out - 1),
        pattern_presets.line(n, low=origin, high=origin + n - 1),
    )
    chart = (
        chart_presets.product((out, n), axes)
        if family == "global"
        else chart_presets.strip(
            (out, n), (out, 16), axes=axes, axis=1, tile_pitch=pitch
        )
    )
    bounds = BandwidthBounds(
        minimum=0.25 if width is None else width,
        birth=1 if width is None else width,
        maximum=16 if width is None else width,
        upper_floor=1 if width is None else width,
    )
    return CSTLinear(
        chart=chart,
        atoms=p,
        kernel=presets.polar_profile_product(
            profiles=(TriweightSpec(), TriweightSpec()),
            amplitude_max=1,
            bounds=bounds,
            w_c=1e6,
            normalization=NormalizationSpec(
                kind="discrete_l2", domain="operator_sites", floor=floor
            ),
            dormant_expansion_rate=0.02,
        ),
        device=device,
    )


def recipe(family, **kw):
    return (GroupedProductRecipe if family == "global" else GroupedStripRecipe)(**kw)


def run(layer, x, family, **kw):
    id = (
        "research_profile_product_global"
        if family == "global"
        else "research_strip_profile_product"
    )
    return Dispatcher(registry=REGISTRY).run(
        layer, LinearInputs(x), plan=ExecutionPlan(id, "v2", recipe(family, **kw))
    )


def oracle(layer, x, dy, family):
    fn = product_oracle if family == "global" else strip_oracle
    return fn(
        copy.deepcopy(layer.kernel).double(),
        layer.atoms.p,
        x,
        dy,
        layer.declaration().charts[0],
    )


@pytest.mark.parametrize("family", ["global", "strip"])
def test_v1_and_v2_recipes_roundtrip_without_rewriting_legacy(family):
    id = (
        "research_profile_product_global"
        if family == "global"
        else "research_strip_profile_product"
    )
    plan = ExecutionPlan(id, "v2", recipe(family))
    assert REGISTRY.loads_plan(REGISTRY.dumps_plan(plan)) == plan
    assert REGISTRY.get(id, revision="v1").recipe_type is not type(plan.recipe)
    with pytest.raises(ValueError):
        recipe(family, prep_group=4)
    with pytest.raises(ValueError):
        recipe(family, atom_group=8)
    with pytest.raises(ValueError):
        recipe(family, sorting="approximate")


@GPU
@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("preparation", ["full", "support"])
@pytest.mark.parametrize(
    "n,out,b,a",
    [
        (65, 33, 1, 1),
        (128, 64, 7, 19),
        (512, 64, 32, 33),
        (1024, 128, 64, 129),
        (128, 128, 7, 4097),
    ],
)
def test_grouped_all_sites_and_canonical_gradients(family, preparation, n, out, b, a):
    torch.manual_seed(41)
    p = torch.randn(a, 4) * 0.1 + torch.tensor([0.3, 1.5, 0, 0])
    p[:, 2] = torch.rand(a) * (out - 2) + 0.37
    p[:, 3] = torch.rand(a) * (n - 2) + 0.37
    layer = model(p, family, n=n, out=out, device="cuda")
    x = torch.randn(b, n * 2, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(out, b, device="cuda").T
    y, dx, dp = oracle(layer, x, dy, family)
    actual = run(layer, x, family, preparation=preparation)
    gx, gp = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    for value, truth in [(actual, y), (gx, dx), (gp, dp)]:
        torch.testing.assert_close(value.double(), truth, rtol=4e-4, atol=2e-5)


@GPU
@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize(
    "case", ["empty", "singleton", "tiny", "floor", "wide", "precision"]
)
def test_grouped_global_floor_and_full_support_fallback(family, case):
    n, out, width = (1024, 64, 400) if case == "wide" else (65, 33, 1)
    co, ci = {
        "empty": (-20, -20),
        "singleton": (0, 0),
        "tiny": (-0.999, -0.999),
        "floor": (-0.39, -0.39),
        "wide": (22.4, 390),
        "precision": (1000000.37, 1000032.7),
    }[case]
    layer = model(
        torch.tensor([[0.3, 1.5, co, ci]]),
        family,
        n=n,
        out=out,
        width=width,
        floor=0.5 if case == "floor" else 1e-6,
        origin=1000000 if case == "precision" else 0,
        device="cuda",
    )
    x = torch.randn(7, n, device="cuda", requires_grad=True)
    dy = torch.randn(7, out, device="cuda")
    y, dx, dp = oracle(layer, x, dy, family)
    actual = run(layer, x, family)
    gx, gp = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    for value, truth in [(actual, y), (gx, dx), (gp, dp)]:
        torch.testing.assert_close(value.double(), truth, rtol=8e-4, atol=2e-5)
    if case == "singleton":
        assert torch.count_nonzero(gp[:, 2:]) == 0


@GPU
@pytest.mark.parametrize("pitch", [8, 16, 20, 40])
def test_strip_overlap_gaps_and_partial_tile_have_exact_full_norm(pitch):
    layer = model(
        torch.tensor([[0.3, 1.5, 2.4, 20.37], [-0.4, 1.4, 22.3, 41.7]]),
        "strip",
        pitch=pitch,
        width=4,
        device="cuda",
    )
    x = torch.randn(7, 65, device="cuda", requires_grad=True)
    dy = torch.randn(7, 33, device="cuda")
    y, dx, dp = oracle(layer, x, dy, "strip")
    actual = run(layer, x, "strip")
    grads = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    for value, truth in [(actual, y), (grads[0], dx), (grads[1], dp)]:
        torch.testing.assert_close(value.double(), truth, rtol=4e-4, atol=2e-5)


@GPU
@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize(
    "prep_group,atom_group,sorting",
    [(1, 1, "legacy"), (8, 1, "legacy"), (1, 1, "torch")],
)
def test_isolated_control_recipes(family, prep_group, atom_group, sorting):
    layer = model(
        torch.tensor([[0.3, 1.5, 2.4, 24], [-0.4, 1.4, 6.3, 32.7]]),
        family,
        device="cuda",
    )
    x = torch.randn(7, 65, device="cuda", requires_grad=True)
    dy = torch.randn(7, 33, device="cuda")
    y, dx, dp = oracle(layer, x, dy, family)
    actual = run(
        layer,
        x,
        family,
        preparation="full" if prep_group == 1 else "support",
        prep_group=prep_group,
        atom_group=atom_group,
        sorting=sorting,
    )
    grads = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    for value, truth in [(actual, y), (grads[0], dx), (grads[1], dp)]:
        torch.testing.assert_close(value.double(), truth, rtol=4e-4, atol=2e-5)


@GPU
@pytest.mark.parametrize("family", ["global", "strip"])
def test_old_forward_snapshots_keep_pitch_amplitude_and_parameters(family):
    layer = model(
        torch.tensor([[0.3, 1.5, 2.4, 24], [-0.4, 1.4, 6.3, 32.7]]),
        family,
        device="cuda",
    )
    reference = copy.deepcopy(layer)
    x = torch.randn(7, 65, device="cuda", requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    dy = torch.randn(7, 33, device="cuda")
    truth = reference.operator.apply(xx, algorithm="factored")
    actual = run(layer, x, family)
    with torch.no_grad():
        layer.atoms.p.add_(0.17)
        layer.kernel.amplitude_max.fill_(0.7)
        if family == "strip":
            layer.chart.tile_pitch.fill_(8)
    run(layer, x.detach(), family)
    for value, expected in zip(
        torch.autograd.grad(actual, (x, layer.atoms.p), dy),
        torch.autograd.grad(truth, (xx, reference.atoms.p), dy),
    ):
        torch.testing.assert_close(value, expected, rtol=4e-4, atol=2e-5)


@GPU
@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("grad_x,grad_p", [(True, False), (False, True), (True, True)])
def test_zero_atoms_and_gradient_branches(family, grad_x, grad_p):
    for p in [torch.empty(0, 4), torch.tensor([[0.3, 1.5, 2.4, 24]])]:
        layer = model(p, family, device="cuda")
        layer.atoms.p.requires_grad_(grad_p)
        x = torch.randn(7, 65, device="cuda", requires_grad=grad_x)
        actual = run(layer, x, family)
        targets = tuple(t for t in [x, layer.atoms.p] if t.requires_grad)
        grads = torch.autograd.grad(actual.sum(), targets)
        assert all(torch.isfinite(t).all() for t in grads)
        if not len(p):
            assert not actual.count_nonzero()
            assert all(not t.count_nonzero() for t in grads)


@GPU
@pytest.mark.parametrize("family", ["global", "strip"])
def test_twenty_captured_updates_match_reference_moments_and_live_widths(family):
    from benchmarks.cuda.linear.profile_product import decode
    from benchmarks.cuda.polar_update import optimizer_step
    from torchcst import AtomUpdateBinding, CSTOptimizer

    layer = model(
        torch.tensor([[0.3, 1.5, 2.4, 24], [-0.4, 1.4, 6.3, 32.7]]),
        family,
        device="cuda",
    )
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3, fused=True, capturable=True)
    binding = AtomUpdateBinding(layer.operator)
    x = torch.randn(7, 65, device="cuda")
    dy = torch.randn(7, 33, device="cuda")
    initial = decode(layer.kernel, layer.atoms.p)[:, 1].detach().clone()

    def step():
        opt.zero_grad(set_to_none=True)
        y = run(layer, x, family)
        (y * dy).sum().backward()
        optimizer_step(binding, opt, step_size=1e-3, polar_update="fused")
        return y

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = step()
    torch.cuda.synchronize()
    reference = copy.deepcopy(layer)
    base = torch.optim.AdamW(reference.parameters(), lr=1e-3, fused=True)
    base.load_state_dict(copy.deepcopy(opt.state_dict()))
    for group in base.param_groups:
        group["capturable"] = False
    ropt = CSTOptimizer(base, model=reference)
    for _ in range(20):
        ropt.zero_grad()
        expected = reference.operator.apply(x, algorithm="factored")
        (expected * dy).sum().backward()
        ropt.step()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(
            layer.atoms.p, reference.atoms.p, rtol=2e-4, atol=2e-5
        )
        for key in ["exp_avg", "exp_avg_sq", "step"]:
            torch.testing.assert_close(
                opt.state[layer.atoms.p][key],
                base.state[reference.atoms.p][key],
                rtol=4e-4,
                atol=2e-5,
            )
    assert torch.any(decode(layer.kernel, layer.atoms.p)[:, 1] != initial)


@GPU
def test_captured_strip_reads_changed_pitch_and_overlap_fallback():
    layer = model(
        torch.tensor([[0.3, 1.5, 2.4, 24], [-0.4, 1.4, 6.3, 32.7]]),
        "strip",
        width=4,
        device="cuda",
    )
    x = torch.randn(7, 65, device="cuda", requires_grad=True)
    dy = torch.randn(7, 33, device="cuda")

    def probe():
        y = run(layer, x, "strip")
        return y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            probe()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = probe()
    for pitch in (8, 16, 40, 20):
        with torch.no_grad():
            layer.chart.tile_pitch.fill_(pitch)
        truth = layer.operator.apply(x, algorithm="factored")
        grads = torch.autograd.grad(truth, (x, layer.atoms.p), dy)
        graph.replay()
        torch.cuda.synchronize()
        for value, expected in zip(actual, (truth, *grads)):
            torch.testing.assert_close(value, expected, rtol=4e-4, atol=2e-5)
