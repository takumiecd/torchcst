"""Physical oracle, supported metadata, retained tensors and captured updates."""

import copy
import subprocess
import sys
from dataclasses import replace

import pytest
import torch
from test_torus_profile_product import model
from test_torus_profile_product_graph_update import PLAN as UPDATE
from test_torus_profile_product_graph_update import dispatcher

from benchmarks.cuda.linear.torus_profile_product import _strict_check, oracle_vjp
from torchcst import (
    AtomUpdateBinding,
    AtomUpdateInputs,
    CSTLinear,
    CSTOptimizer,
    Dispatcher,
    LinearInputs,
    TriweightSpec,
)
from torchcst._backends.cuda.algorithms.linear.torus_profile_product.onchip.algorithm import (
    OnchipRecipe,
    TorusOnchipAlgorithm,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DeviceInfo, ExecutionPlan, PrecisionPolicy

RECIPE = OnchipRecipe()
PLAN = ExecutionPlan(TorusOnchipAlgorithm().id, "v3", RECIPE)


def fixture(**kwargs):
    layer = model(**kwargs)
    spec = replace(
        layer.kernel.spec,
        profiles=tuple(
            replace(b, profile=TriweightSpec()) for b in layer.kernel.spec.profiles
        ),
    )
    return CSTLinear(
        chart=copy.deepcopy(layer.chart),
        kernel=spec,
        atoms=layer.atoms.p.detach().clone(),
        device=layer.atoms.p.device,
        dtype=layer.atoms.p.dtype,
        backend="factored",
    )


def run(layer, x):
    registry = Registry()
    registry.register(TorusOnchipAlgorithm())
    assert registry.loads_plan(registry.dumps_plan(PLAN)) == PLAN
    return Dispatcher(registry=registry).run(layer, LinearInputs(x), plan=PLAN)


def test_metadata_rejects_unsupported_geometry_shape_and_precision():
    layer = fixture().float()
    alg = TorusOnchipAlgorithm()
    ctx = layer.build_context(LinearInputs(torch.randn(3, layer.in_features)))
    cuda = replace(ctx, device=DeviceInfo("cuda", 0, "NVIDIA L4", (8, 9), 58))
    assert alg.supports(cuda, RECIPE).supported
    for change in (
        {"device": DeviceInfo("cpu")},
        {"dtype": torch.float64},
        {"deterministic": True},
        {"precision": PrecisionPolicy(autocast=True)},
        {"input_shape": (0, 12)},
        {"input_shape": (1, 3, 12), "input_strides": (36, 12, 1)},
    ):
        assert not alg.supports(replace(cuda, **change), RECIPE).supported
    for kwargs in (
        {"reverse": True},
        {"kind": "product"},
        {"representation": "ambient"},
    ):
        other = fixture(**kwargs).float()
        cc = other.build_context(LinearInputs(torch.randn(3, other.in_features)))
        assert not alg.supports(replace(cc, device=cuda.device), RECIPE).supported
    gaussian = model().float()
    gc = gaussian.build_context(LinearInputs(torch.randn(3, gaussian.in_features)))
    assert not alg.supports(replace(gc, device=cuda.device), RECIPE).supported
    for trig in (True, 2, "unknown"):
        with pytest.raises(ValueError):
            OnchipRecipe(trig=trig)
    for settings in ((True, 4), (32, 4), (64, True), (64, 8)):
        with pytest.raises(ValueError):
            OnchipRecipe(*settings)


def test_metadata_and_catalog_do_not_import_execution_code():
    code = (
        "import sys; from benchmarks.cuda.linear.manifest import REGISTRY; "
        "assert REGISTRY.get('research_cuda_torus_profile_product_onchip', revision='v3'); "
        "assert 'triton' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


@CUDA
@pytest.mark.parametrize("floor", [1e-6, 0.5, 100.0])
@pytest.mark.parametrize("batch", [1, 3, 32, 64])
def test_y_dx_all_p_against_independent_embedded_fibres(floor, batch):
    torch.manual_seed(41)
    layer = fixture(device="cuda", floor=floor).float()
    x = torch.randn(batch, layer.in_features, device="cuda", requires_grad=True)
    dy = torch.randn(batch, layer.out_features, device="cuda")
    truth, tx, tp = oracle_vjp(layer.kernel.double(), layer.atoms.p, x, dy, layer.chart)
    layer.kernel.float()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append(tuple(t.shape)) or t, lambda t: t
    ):
        y = run(layer, x)
    gx, gp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for actual, expected in ((y, truth), (gx, tx), (gp, tp)):
        _strict_check(actual, expected, tol=4e-4)
    assert (batch, len(layer.atoms.p)) not in saved
    assert (layer.out_features, len(layer.atoms.p)) not in saved
    assert (layer.in_features, len(layer.atoms.p)) not in saved
    assert (layer.out_features, layer.in_features) not in saved


@CUDA
@pytest.mark.parametrize("grads", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("empty", [False, True])
def test_independent_gradient_requirements_and_empty(grads, empty):
    torch.manual_seed(41)
    layer = fixture(device="cuda").float()
    if empty:
        layer = CSTLinear(
            chart=layer.chart,
            kernel=layer.kernel.spec,
            atoms=layer.atoms.p[:0].detach(),
            device="cuda",
        )
    ref = copy.deepcopy(layer)
    layer.atoms.p.requires_grad_(grads[1])
    ref.atoms.p.requires_grad_(grads[1])
    x = torch.randn(3, layer.in_features, device="cuda", requires_grad=grads[0])
    rx = x.detach().clone().requires_grad_(grads[0])
    y, truth = run(layer, x), ref(rx)
    ga = torch.autograd.grad(
        y.sum(), [q for q in (x, layer.atoms.p) if q.requires_grad]
    )
    gt = torch.autograd.grad(
        truth.sum(), [q for q in (rx, ref.atoms.p) if q.requires_grad]
    )
    torch.testing.assert_close(y, truth, atol=4e-4, rtol=4e-4)
    for a, b in zip(ga, gt, strict=True):
        torch.testing.assert_close(a, b, atol=4e-4, rtol=4e-4)


@CUDA
def test_previous_forward_snapshots_live_mutated_operands():
    layer = fixture(device="cuda").float()
    ref = copy.deepcopy(layer)
    x = torch.randn(3, layer.in_features, device="cuda", requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    dy = torch.randn(3, layer.out_features, device="cuda")
    y, truth = run(layer, x), ref(rx)
    with torch.no_grad():
        layer.atoms.p.add_(0.03)
        layer.chart.tile_pitch.add_(0.1)
        layer.chart.geometry.major_radius.add_(0.1)
        layer.chart.geometry.minor_radius.add_(0.05)
        layer.kernel.scalar("amplitude_max").mul_(0.8)
    ga = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    gt = torch.autograd.grad(truth, (rx, ref.atoms.p), dy)
    for a, b in zip(ga, gt, strict=True):
        _strict_check(a, b, tol=4e-4)
    torch.testing.assert_close(
        run(layer, x.detach()), layer(x.detach()), atol=4e-4, rtol=4e-4
    )


@CUDA
def test_twenty_captured_steps_with_live_geometry_widths_and_moments():
    from torchcst._backends.torch.parameterizations.profile_product import (
        bandwidth_sigma,
    )

    torch.manual_seed(41)
    layer = fixture(device="cuda").float()
    opt = torch.optim.AdamW(
        layer.parameters(), lr=1e-3, weight_decay=0.02, fused=True, capturable=True
    )
    binding, d = AtomUpdateBinding(layer.operator), dispatcher()
    x = torch.randn(3, layer.in_features, device="cuda", requires_grad=True)
    dy = torch.randn(3, layer.out_features, device="cuda")
    initial = bandwidth_sigma(layer.kernel, layer.chart, layer.atoms.p).detach().clone()

    def step():
        opt.zero_grad(set_to_none=True)
        x.grad = None
        y = run(layer, x)
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
    for m in (layer, ref):
        m.chart.geometry.major_radius.add_(0.02)
        m.chart.geometry.minor_radius.add_(0.01)
        m.chart.tile_pitch.add_(0.01)
    for _ in range(20):
        ropt.zero_grad(set_to_none=True)
        rx.grad = None
        truth = ref(rx)
        (truth * dy).sum().backward()
        ropt.step()
        graph.replay()
        torch.cuda.synchronize()
        for a, b in (
            (y, truth),
            (x.grad, rx.grad),
            (layer.atoms.p.grad, ref.atoms.p.grad),
            (layer.atoms.p, ref.atoms.p),
        ):
            _strict_check(a, b, tol=4e-4)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                opt.state[layer.atoms.p][key],
                base.state[ref.atoms.p][key],
                atol=2e-5,
                rtol=2e-5,
            )
    assert (bandwidth_sigma(layer.kernel, layer.chart, layer.atoms.p) != initial).any()


@CUDA
@pytest.mark.parametrize("n", [1024, 2048])
@pytest.mark.parametrize("sigma", [3, 8])
def test_full_axes_thin_periodic_support_and_partial_atoms(n, sigma):
    from benchmarks.cuda.linear.manifest import load_run
    from benchmarks.cuda.linear.torus_profile_product import (
        fixture_operator,
        initialize,
    )

    case = load_run(
        f"benchmarks/cuda/linear/cases/torus-profile-product-onchip-{n}-sigma{sigma}.json",
        "benchmarks/cuda/linear/plans-torus-profile-product-onchip.json",
    ).case
    op = fixture_operator(case)
    layer = CSTLinear(
        chart=op.charts[0], kernel=op.kernel, atoms=initialize(case)[:23], device="cuda"
    )
    gen = torch.Generator().manual_seed(41)
    x = torch.randn(32, n, generator=gen).cuda().requires_grad_()
    dy = torch.randn(32, n, generator=gen).cuda()
    truth, tx, tp = oracle_vjp(
        copy.deepcopy(layer.kernel).double(), layer.atoms.p, x, dy, layer.chart
    )
    y = run(layer, x)
    gx, gp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    assert (tp[:, 2:].abs() > 1e-8).all()
    for a, b in ((y, truth), (gx, tx), (gp, tp)):
        _strict_check(a, b, tol=4e-4)
