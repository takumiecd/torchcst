"""Independent physical queries, bounded blocks, live state and captured steps."""

import copy
from dataclasses import replace

import pytest
import torch
from test_torus_profile_product import model, physical_atoms
from test_torus_profile_product_graph_update import PLAN as UPDATE
from test_torus_profile_product_graph_update import dispatcher

from torchcst import (
    AtomUpdateBinding,
    AtomUpdateInputs,
    CSTOptimizer,
    Dispatcher,
    LinearInputs,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
from torchcst._backends.torch.algorithms.linear.torus_profile_product.algorithm import (
    TorusChunkAlgorithm,
    TorusChunkRecipe,
)

RECIPES = [
    TorusChunkRecipe(16),
    TorusChunkRecipe(16, save_h=False),
    TorusChunkRecipe(16, "w", False),
]


def run(layer, x, recipe):
    registry = Registry()
    registry.register(TorusChunkAlgorithm())
    plan = ExecutionPlan(TorusChunkAlgorithm().id, "v1", recipe)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    return Dispatcher(registry=registry).run(layer, LinearInputs(x), plan=plan)


def fixture(*, device="cpu", **kwargs):
    layer = model(device=device, **kwargs)
    # A partial last block, multiple active centres, and existing empty atoms.
    from torchcst import CSTLinear

    p = layer.atoms.p.detach().repeat(5, 1)[:19].clone()
    p[4:, 2] += 0.03
    return CSTLinear(
        chart=copy.deepcopy(layer.chart),
        atoms=p,
        kernel=layer.kernel.spec,
        device=device,
        dtype=p.dtype,
        backend="factored",
    )


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize(
    "kind,reverse", [("product", False), ("strip", False), ("strip", True)]
)
@pytest.mark.parametrize("floor", [1e-6, 0.5])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_all_values_and_every_parameter_against_embedded_matrix(
    recipe, kind, reverse, floor, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("actual CUDA required")
    torch.manual_seed(41)
    layer = fixture(device=device, kind=kind, reverse=reverse, floor=floor)
    p = layer.atoms.p.detach().clone().requires_grad_()
    x = torch.randn(
        3, layer.in_features, device=device, dtype=torch.float64, requires_grad=True
    )
    rx = x.detach().clone().requires_grad_()
    dy = torch.randn(3, layer.out_features, device=device, dtype=torch.float64)
    truth = rx @ physical_atoms(layer, p).sum(0).T
    actual = run(layer, x, recipe)
    ga = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    gt = torch.autograd.grad(truth, (rx, p), dy)
    torch.testing.assert_close(actual, truth, rtol=1e-10, atol=1e-12)
    for a, b in zip(ga, gt, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_old_forward_snapshots_mutated_parameters_geometry_pitch_and_amplitude(
    recipe, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("actual CUDA required")
    layer = fixture(device=device)
    reference = copy.deepcopy(layer)
    x = torch.randn(
        2, layer.in_features, device=device, dtype=torch.float64, requires_grad=True
    )
    rx = x.detach().clone().requires_grad_()
    dy = torch.randn(2, layer.out_features, device=device, dtype=torch.float64)
    y = run(layer, x, recipe)
    truth = reference(rx)
    with torch.no_grad():
        layer.atoms.p.add_(0.03)
        layer.chart.tile_pitch.add_(0.1)
        layer.chart.geometry.major_radius.add_(0.1)
        layer.chart.geometry.minor_radius.add_(0.05)
        layer.kernel.scalar("amplitude_max").mul_(0.8)
    ga = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    gt = torch.autograd.grad(truth, (rx, reference.atoms.p), dy)
    for a, b in zip(ga, gt, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(
        run(layer, x.detach(), recipe), layer(x.detach()), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("grads", [(False, True), (True, False), (True, True)])
def test_empty_and_independent_gradient_requirements(recipe, grads):
    layer = model()
    from torchcst import CSTLinear

    empty = CSTLinear(
        chart=layer.chart,
        atoms=layer.atoms.p[:0].detach(),
        kernel=layer.kernel.spec,
        dtype=torch.float64,
    )
    empty.atoms.p.requires_grad_(grads[1])
    x = torch.randn(2, empty.in_features, dtype=torch.float64, requires_grad=grads[0])
    y = run(empty, x, recipe)
    targets = [q for q in (x, empty.atoms.p) if q.requires_grad]
    values = torch.autograd.grad(y.sum(), targets)
    assert y.count_nonzero() == 0
    assert all(v.count_nonzero() == 0 for v in values)
    full = fixture()
    full.atoms.p.requires_grad_(grads[1])
    xx = x.detach().requires_grad_(grads[0])
    actual = run(full, xx, recipe)
    ref = copy.deepcopy(full)
    rx = xx.detach().clone().requires_grad_(grads[0])
    aa = torch.autograd.grad(
        actual.sum(), [q for q in (xx, full.atoms.p) if q.requires_grad]
    )
    bb = torch.autograd.grad(
        ref(rx).sum(), [q for q in (rx, ref.atoms.p) if q.requires_grad]
    )
    for a, b in zip(aa, bb, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)


def test_strict_recipe_metadata_and_geometry_support():
    for options in [
        {"atom_chunk": True},
        {"atom_chunk": 32},
        {"save_h": 1},
        {"contraction": "w"},
    ]:
        with pytest.raises(ValueError):
            TorusChunkRecipe(**options)
    layer = model()
    ctx = layer.build_context(
        LinearInputs(torch.randn(3, layer.in_features, dtype=torch.float64))
    )
    alg = TorusChunkAlgorithm()
    assert alg.supports(ctx, RECIPES[0]).supported
    assert not alg.supports(replace(ctx, dtype=torch.float16), RECIPES[0]).supported
    ambient = model(representation="ambient")
    context = ambient.build_context(
        LinearInputs(torch.randn(3, ambient.in_features, dtype=torch.float64))
    )
    assert not alg.supports(context, RECIPES[0]).supported


@pytest.mark.parametrize("recipe", RECIPES)
def test_forward_retains_selected_intermediate_and_no_full_factor_arrays(recipe):
    layer = fixture()
    x = torch.randn(3, layer.in_features, dtype=torch.float64, requires_grad=True)
    shapes = []

    def pack(t):
        shapes.append(tuple(t.shape))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = run(layer, x, recipe)
    assert (layer.in_features, len(layer.atoms.p)) not in shapes
    assert (layer.out_features, len(layer.atoms.p)) not in shapes
    assert ((len(x), len(layer.atoms.p)) in shapes) == recipe.save_h
    assert ((layer.out_features, layer.in_features) in shapes) == (
        recipe.contraction == "w"
    )
    y.sum().backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("reverse", [False, True])
def test_twenty_captured_steps_gradients_moments_widths_and_live_geometry(
    recipe, reverse
):
    from torchcst._backends.torch.parameterizations.profile_product import (
        bandwidth_sigma,
    )

    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = fixture(device="cuda", reverse=reverse).float()
    binding = AtomUpdateBinding(layer.operator)
    d = dispatcher()
    opt = torch.optim.AdamW(
        layer.parameters(), lr=1e-3, weight_decay=0.02, fused=True, capturable=True
    )
    x = torch.randn(3, layer.in_features, device="cuda", requires_grad=True)
    dy = torch.randn(3, layer.out_features, device="cuda")
    initial = bandwidth_sigma(layer.kernel, layer.chart, layer.atoms.p).detach().clone()

    def step():
        opt.zero_grad(set_to_none=True)
        x.grad = None
        y = run(layer, x, recipe)
        (y * dy).sum().backward()
        previous = layer.atoms.p.detach().clone()
        opt.step()
        d.run(binding, AtomUpdateInputs(previous, 1e-3), plan=UPDATE)
        return y

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y = step()
    torch.cuda.synchronize()
    ref = copy.deepcopy(layer)
    base = torch.optim.AdamW(ref.parameters(), lr=1e-3, weight_decay=0.02, fused=True)
    base.load_state_dict(copy.deepcopy(opt.state_dict()))
    for group in base.param_groups:
        group["capturable"] = False
    ropt = CSTOptimizer(base, model=ref)
    rx = x.detach().clone().requires_grad_()
    # Replaying must use live radii/pitch; this is not a frozen geometry cache.
    for m in (layer, ref):
        m.chart.geometry.minor_radius.add_(0.01)
        m.chart.geometry.major_radius.add_(0.02)
        m.chart.tile_pitch.add_(0.01)
    for _ in range(20):
        ropt.zero_grad(set_to_none=True)
        rx.grad = None
        truth = ref(rx)
        (truth * dy).sum().backward()
        ropt.step()
        graph.replay()
        torch.cuda.synchronize()
        for a, b in [
            (y, truth),
            (x.grad, rx.grad),
            (layer.atoms.p.grad, ref.atoms.p.grad),
            (layer.atoms.p, ref.atoms.p),
        ]:
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                opt.state[layer.atoms.p][key],
                base.state[ref.atoms.p][key],
                rtol=2e-5,
                atol=2e-5,
            )
    assert (bandwidth_sigma(layer.kernel, layer.chart, layer.atoms.p) != initial).any()
