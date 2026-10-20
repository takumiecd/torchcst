"""Node-owner integration: independent full Polar all4P and frame lifetime gates."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from benchmarks.cuda.linear import bspline as bs
from torchcst import CSTLinear
from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.algorithm import (
    BSplineBernsteinNodeAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.recipe import (
    BSplineBernsteinNodeRecipe,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DeviceInfo, ExecutionPlan
from torchcst.operators.context import LinearContext

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
MODES = [
    {},
    {"control_float32": True},
    {"control_float32": True, "owner_float32": True},
    {"control_float32": True, "owner_float32": True, "query_float32": True},
]


def test_node_recipe_roundtrip_and_capacity_support():
    registry = Registry()
    algorithm = BSplineBernsteinNodeAlgorithm()
    registry.register(algorithm)
    for flags in MODES:
        recipe = BSplineBernsteinNodeRecipe(**flags)
        plan = ExecutionPlan(algorithm.id, algorithm.revision, recipe)
        assert registry.loads_plan(registry.dumps_plan(plan)) == plan
        assert recipe.snapshot_source and not recipe.norm_float32
        for n, expected in ((64, True), (1024, True), (63, False), (2048, False)):
            case = SimpleNamespace(size=n, atoms=137, seed=67, widths=None)
            context = LinearContext(
                bs.fixture_operator(case),
                (33, n),
                (n, 1),
                torch.float32,
                137,
                DeviceInfo("cuda", 0),
                4,
            )
            assert algorithm.supports(context, recipe).supported is expected
    for bad in (
        {"owner_float32": True},
        {"query_float32": True},
        {"control_float32": 1},
        {"owner_rows": 64},
        {"batch_chunk": True},
        {"strategy": "sigma_stream"},
    ):
        with pytest.raises(ValueError):
            BSplineBernsteinNodeRecipe(**bad)


def _model(n=64, a=137, floor=1e-6, **options):
    case = SimpleNamespace(size=n, atoms=a, seed=67137, widths=None)
    op = bs.fixture_operator(case)
    kernel = replace(
        op.kernel, normalization=replace(op.kernel.normalization, floor=floor)
    )
    registry = Registry()
    algorithm = BSplineBernsteinNodeAlgorithm()
    registry.register(algorithm)
    plan = ExecutionPlan(
        algorithm.id, algorithm.revision, BSplineBernsteinNodeRecipe(**options)
    )
    layer = CSTLinear(
        chart=op.charts[0],
        atoms=bs.initialize(case),
        kernel=kernel,
        selector=FixedSelector(plan, registry=registry),
    )
    layer.chart.double()
    layer.chart.spacing.fill_(0.1)
    return layer.cuda()


def _close(actual, truth):
    assert actual.shape == truth.shape
    if not actual.numel():
        return
    diff = actual.double() - truth.double()
    assert torch.isfinite(actual).all() and torch.isfinite(truth).all()
    assert diff.abs().max() < 4e-4, float(diff.abs().max())
    assert diff.norm() / truth.double().norm().clamp_min(1e-300) < 4e-4


def _gate(layer, x, dy):
    rx = x.detach().double().requires_grad_()
    rp = layer.atoms.p.detach().double().requires_grad_()
    truth = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    y = layer(x)
    actual = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    expected = torch.autograd.grad(truth, (rx, rp), dy.double())
    for a, b in zip((y, *actual), (truth, *expected)):
        _close(a, b)
    repeated = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for a, b in zip(repeated, actual):
        _close(a, b)


@GPU
def test_node_ids_cover_and_owner_csr_no_dropped_edges():
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        bernstein_node_kernels as nk,
    )
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        bernstein_owner_kernels as ok,
    )
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.bernstein_executor import (
        _owner,
    )

    torch.manual_seed(67137)
    a, n = 137, 1024
    recipe = BSplineBernsteinNodeRecipe()
    origin = torch.tensor([-0.27, 0.31], device="cuda", dtype=torch.float64)
    spacing = torch.tensor([0.1, 0.13], device="cuda", dtype=torch.float64)
    meta = torch.zeros(a, 12, device="cuda", dtype=torch.float64)
    meta[:, 1] = torch.exp(torch.rand(a, device="cuda", dtype=torch.float64) * 5 - 1.5)
    meta[:, 2:4] = (
        origin
        + torch.rand(a, 2, device="cuda", dtype=torch.float64) * (n - 1) * spacing
    )
    meta[:3, 2:4] = -1000
    meta[:, 10:] = spacing.reciprocal()
    cuts = torch.empty(2, 3, 2, a, device="cuda", dtype=torch.int32)
    nk.prepare_cuts[(tr.cdiv(a, 128),)](
        meta, cuts, origin, spacing, a, n, n, num_warps=4, enable_fp_fusion=False
    )
    ids = torch.empty(2, a, 34, device="cuda", dtype=torch.int16)
    counts = torch.empty(2, 3, a, device="cuda", dtype=torch.uint8)
    hist = torch.zeros(2 * n - 1, device="cuda", dtype=torch.int32)
    ok.prepare_ids[(tr.cdiv(a, 128),)](
        meta,
        ids,
        counts,
        hist,
        origin,
        spacing,
        a,
        n,
        n,
        34,
        num_warps=4,
        enable_fp_fusion=False,
        debug=True,
    )
    codes = (ids.int() & 65535).cpu()
    cc = counts.cpu()
    limits = cuts.cpu()
    for dim in (0, 1):
        for atom in range(a):
            start = 0
            for piece in range(3):
                count = int(cc[dim, piece, atom])
                selected = codes[dim, atom, start : start + count]
                left, right = limits[dim, piece, :, atom].tolist()
                cursor = left
                for code in selected.tolist():
                    assert code >> 14 == piece
                    lo = code & 1023
                    log = (code >> 10) & 15
                    width = 1 << log
                    assert lo == cursor and lo % width == 0 and lo + width <= right
                    cursor += width
                assert cursor == right
                start += count
            assert start <= 34 and torch.all(codes[dim, atom, start:] == 65535)
        offsets, owners = _owner(
            ids, a, n, dim, recipe, "unused", hist if dim == 0 else None
        )
        off = offsets.cpu()
        owner = owners.cpu()
        flat = codes[dim].flatten()
        found = []
        for node in range(2 * n - 1):
            edges = owner[off[node] : off[node + 1]].tolist()
            for e in edges:
                code = int(flat[e])
                log = (code >> 10) & 15
                left = code & 1023
                assert code != 65535 and (n >> log) - 1 + (left >> log) == node
            found.extend(edges)
        assert sorted(found) == torch.nonzero(flat != 65535).flatten().tolist()
    assert ids.numel() * ids.element_size() == 2 * a * 34 * 2


@GPU
@pytest.mark.parametrize("options", MODES, ids=["node64", "node32", "part32", "full32"])
@pytest.mark.parametrize(
    "n,batch,floor", [(64, 3, 1e-6), (64, 33, 1e-6), (64, 64, 1000.0), (1024, 33, 1e-6)]
)
def test_nodes_dynamic_all4p(options, n, batch, floor):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(67137)
    layer = _model(n=n, floor=floor, **options)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.001)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(batch, 2 * n, device="cuda")[:, ::2].requires_grad_()
        _gate(layer, x, torch.randn(batch, n, device="cuda"))


@GPU
@pytest.mark.parametrize("options", MODES, ids=["node64", "node32", "part32", "full32"])
@pytest.mark.parametrize("radius_scale", [1.15, 1.4])
def test_nodes_coincident513_and_tiny_floor(options, radius_scale):
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = _model(a=513, **options)
    with torch.no_grad():
        layer.atoms.p[:, 0] = 0.6 * radius_scale
        layer.atoms.p[:, 1] = 0.8 * radius_scale
        layer.atoms.p[:, 2:] = 3.15
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.001)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(33, 64, device="cuda", requires_grad=True)
        _gate(layer, x, torch.randn_like(x))
    layer = _model(a=17, **options)
    with torch.no_grad():
        layer.atoms.p[0, 0] = 0.6
        layer.atoms.p[0, 1] = 0.8
        sigma = bs.decode(layer.kernel, layer.atoms.p)[0, 1].rsqrt()
        layer.atoms.p[0, 2] = -2.449489742783178 * sigma + 1e-6
        layer.atoms.p[1, 2:] = -100
        layer.atoms.p[2, 0] = 0
        layer.atoms.p[3, 0] *= -1
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.001)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(33, 64, device="cuda", requires_grad=True)
        _gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("options", [MODES[0], MODES[-1]], ids=["node64", "full32"])
def test_nodes_graph_individual_and_empty(options):
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = _model(a=33, **options)
    x = torch.randn(33, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    values = []

    def run():
        y = layer(x)
        grads = torch.autograd.grad(y, (x, layer.atoms.p), dy)
        values[:] = [y, *grads]

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        with torch.no_grad():
            layer.atoms.p[:, :2].mul_(1.001)
            layer.atoms.p[:, 2:].add_(0.0037)
            x.mul_(1.003)
            dy.mul_(0.997)
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        truth = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        gradients = torch.autograd.grad(truth, (rx, rp), dy.double())
        for a, b in zip(values, (truth, *gradients)):
            _close(a, b)
    for need_x, need_p in ((True, False), (False, True)):
        layer.atoms.p.requires_grad_(need_p)
        xx = x.detach().requires_grad_(need_x)
        rx = xx.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        truth = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        expected = torch.autograd.grad(truth, rx if need_x else rp, dy.double())[0]
        actual = torch.autograd.grad(layer(xx), xx if need_x else layer.atoms.p, dy)[0]
        _close(actual, expected)
    empty = _model(a=0, **options)
    xx = x.detach().requires_grad_()
    yy = empty(xx)
    gx, gp = torch.autograd.grad(yy.sum(), (xx, empty.atoms.p))
    assert torch.count_nonzero(yy) == torch.count_nonzero(gx) == 0 and gp.shape == (
        0,
        4,
    )


@GPU
@pytest.mark.parametrize("options", [MODES[0], MODES[-1]], ids=["node64", "full32"])
def test_nodes_retained_snapshot_and_concurrent_backward(options):
    from concurrent.futures import ThreadPoolExecutor

    torch.backends.cuda.matmul.allow_tf32 = False
    layer = _model(a=33, **options)
    x = torch.randn(33, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = layer(x)
    expected = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.003)
        layer.atoms.p[:, 2:].add_(0.037)
    # A later forward must not replace the retained frame's IDs or bank.
    _gate(layer, torch.randn_like(x, requires_grad=True), torch.randn_like(dy))
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())

    def work(stream):
        with torch.cuda.stream(stream):
            return torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(work, streams))
    for stream in streams:
        stream.synchronize()
    for gradients in results:
        for a, b in zip(gradients, expected):
            _close(a, b)
