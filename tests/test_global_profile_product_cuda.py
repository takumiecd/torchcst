"""Whole Product norms, support preparation, large layouts and evolving VJPs."""

import copy
from dataclasses import asdict, replace

import pytest
import torch

from benchmarks.cuda.linear.global_profile_product import oracle_factors, oracle_vjp
from benchmarks.cuda.linear.manifest import REGISTRY, decode_snapshot, load_run
from benchmarks.cuda.linear.profile_product import oracle_atoms
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
from torchcst._backends.cuda.algorithms.linear.profile_product_global import (
    GlobalProductAlgorithm,
    GlobalProductRecipe,
)
from torchcst._backends.cuda.algorithms.linear.profile_product_global.algorithm import (
    product_spec,
)
from torchcst._backends.schema import DeviceInfo, ExecutionPlan

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


def model(p, n=65, out=33, width=None, floor=1e-6, device="cpu"):
    bounds = BandwidthBounds(
        minimum=0.25 if width is None else width,
        birth=1.0 if width is None else width,
        maximum=16.0 if width is None else width,
        upper_floor=1.0 if width is None else width,
    )
    return CSTLinear(
        chart=chart_presets.product(
            (out, n),
            (
                pattern_presets.line(out, low=-3, high=out - 4),
                pattern_presets.line(n, low=7, high=n + 6),
            ),
        ),
        atoms=p,
        kernel=presets.polar_profile_product(
            profiles=(TriweightSpec(), TriweightSpec()),
            amplitude_max=1.0,
            bounds=bounds,
            w_c=1e6,
            normalization=NormalizationSpec(
                kind="discrete_l2", domain="operator_sites", floor=floor
            ),
            dormant_expansion_rate=0.02,
        ),
        device=device,
    )


def run(layer, x, preparation="support", atom_block=32):
    return Dispatcher(registry=REGISTRY).run(
        layer,
        LinearInputs(x),
        plan=ExecutionPlan(
            "research_profile_product_global",
            "v1",
            GlobalProductRecipe(preparation=preparation, atom_block=atom_block),
        ),
    )


@pytest.mark.parametrize("preparation", ["full", "support"])
def test_metadata_roundtrip_and_support(preparation):
    recipe = GlobalProductRecipe(preparation=preparation)
    plan = ExecutionPlan("research_profile_product_global", "v1", recipe)
    assert REGISTRY.loads_plan(REGISTRY.dumps_plan(plan)) == plan
    layer = model(torch.tensor([[0.3, 1.5, 2.4, 24.0]]), n=1024, out=1024)
    context = layer.build_context(LinearInputs(torch.randn(7, 1024)))
    algorithm = GlobalProductAlgorithm()
    assert not algorithm.supports(context, recipe).supported
    context = replace(context, device=DeviceInfo("cuda", 0))
    assert algorithm.supports(context, recipe).supported
    for bad in (
        replace(context, atom_count=65537),
        replace(context, dtype=torch.float64),
        replace(context, input_shape=(65, 1024)),
    ):
        assert not algorithm.supports(bad, recipe).supported
    with pytest.raises(ValueError):
        product_spec(model(layer.atoms.p, n=1025).declaration())
    with pytest.raises(ValueError):
        GlobalProductRecipe(preparation="clipped")


@pytest.mark.parametrize("n", [256, 512, 1024])
@pytest.mark.parametrize("rho", [3, 8])
def test_global_fixture_roundtrip(n, rho):
    value = load_run(
        f"benchmarks/cuda/linear/cases/profile-product-global-{n}-rho{rho}.json",
        "benchmarks/cuda/linear/plans-profile-product-global.json",
    )
    assert decode_snapshot(value.snapshot()) == value
    from benchmarks.cuda.linear.protocol import measurement_operator

    assert measurement_operator(asdict(value.case)).charts[0].shape == (n, n)


@pytest.mark.parametrize("floor,width", [(1e-6, 4), (0.5, 1), (1e-6, 400)])
def test_factored_oracle_matches_independent_full_raw_matrices(floor, width):
    layer = model(
        torch.tensor([[0.3, 1.5, -3.39, 6.61], [-0.4, 1.4, 6.3, 32.7]]),
        n=33,
        out=16,
        width=width,
        floor=floor,
    )
    p = layer.atoms.p.detach().double().requires_grad_()
    value = copy.deepcopy(layer.kernel).double()
    chart = layer.declaration().charts[0]
    u, v, scale = oracle_factors(value, p, chart)
    matrix = scale[:, None, None] * u[:, :, None] * v[:, None, :]
    raw = oracle_atoms(
        value,
        p,
        1,
        output_sites=torch.arange(16, dtype=torch.float64) - 3,
        input_sites=torch.arange(33, dtype=torch.float64) + 7,
    )
    torch.testing.assert_close(matrix, raw, rtol=1e-12, atol=1e-12)
    dy = torch.randn_like(raw)
    (a,) = torch.autograd.grad(matrix, p, dy, retain_graph=True)
    (b,) = torch.autograd.grad(raw, p, dy)
    torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)


@GPU
@pytest.mark.parametrize("preparation", ["full", "support"])
@pytest.mark.parametrize(
    "n,out,rows,atoms",
    [
        (64, 32, 1, 19),
        (65, 33, 7, 19),
        (512, 512, 32, 19),
        (1024, 1024, 64, 19),
        (128, 128, 7, 4097),
    ],
)
def test_full_sites_y_dx_and_canonical_gradients(preparation, n, out, rows, atoms):
    torch.manual_seed(41)
    p = torch.randn(atoms, 4) * 0.1 + torch.tensor([0.3, 1.5, 0.0, 0.0])
    p[:, 2] = -2.63 + torch.rand(atoms) * (out - 2)
    p[:, 3] = 7.37 + torch.rand(atoms) * (n - 2)
    layer = model(p, n=n, out=out, device="cuda")
    x = torch.randn(rows, n * 2, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(out, rows, device="cuda").T
    y, dx, dp = oracle_vjp(
        copy.deepcopy(layer.kernel).double(),
        layer.atoms.p,
        x,
        dy,
        layer.declaration().charts[0],
    )
    actual = run(layer, x, preparation)
    ga = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    for a, b in ((actual, y), (ga[0], dx), (ga[1], dp)):
        torch.testing.assert_close(a.double(), b, rtol=4e-4, atol=2e-5)
    assert (dp[:, 2:].abs() > 1e-8).all()


@GPU
@pytest.mark.parametrize("case", ["empty", "singleton", "tiny", "floor", "wide"])
def test_support_floor_certificates_and_unbounded_width(case):
    n, out, width = (1024, 512, 400) if case == "wide" else (33, 16, 1)
    co, ci = {
        "empty": (-20, -20),
        "singleton": (-3, 39),
        "tiny": (-3.999, 6.001),
        "floor": (-3.39, 6.61),
        "wide": (22.4, 390),
    }[case]
    layer = model(
        torch.tensor([[0.3, 1.5, co, ci]]),
        n=n,
        out=out,
        width=width,
        floor=0.5 if case == "floor" else 1e-6,
        device="cuda",
    )
    x = torch.randn(7, n, device="cuda", requires_grad=True)
    dy = torch.randn(7, out, device="cuda")
    y, dx, dp = oracle_vjp(
        copy.deepcopy(layer.kernel).double(),
        layer.atoms.p,
        x,
        dy,
        layer.declaration().charts[0],
    )
    actual = run(layer, x, atom_block=16)
    ga = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    for a, b in ((actual, y), (ga[0], dx), (ga[1], dp)):
        torch.testing.assert_close(a.double(), b, rtol=8e-4, atol=2e-5)


@GPU
def test_chunk_ranges_never_drop_an_early_wide_interval():
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.profile_product_global import (
        kernels as k,
    )

    a, owners = 4097, 64
    views = torch.zeros(2, 13, a, device="cuda")
    lo = torch.arange(a, device="cuda") * 1024 // a
    hi = torch.minimum(lo + 4, torch.full_like(lo, 1024))
    hi[0] = 1024
    for direction in (0, 1):
        views[direction, 9] = lo
        views[direction, 10] = hi
        views[direction, 11] = lo
        views[direction, 12] = hi
    chunks = tr.cdiv(a, 1024)
    highs = torch.empty(2, chunks, device="cuda", dtype=torch.int32)
    ranges = torch.empty(2, owners, 2, device="cuda", dtype=torch.int32)
    k.block_highs[(chunks, 2)](views, highs, a, chunks, 1024, num_warps=4)
    k.chunk_ranges[(2, owners)](
        views,
        highs,
        ranges,
        a,
        owners,
        chunks,
        tr.next_power_of_2(chunks),
        1024,
        num_warps=4,
    )
    assert (ranges[:, :, 0] == 0).all()
    ids = torch.arange(a, device="cuda")
    for owner in range(owners):
        valid = (lo < (owner + 1) * 16) & (hi > owner * 16)
        for direction in (0, 1):
            included = (ids >= ranges[direction, owner, 0]) & (
                ids < ranges[direction, owner, 1]
            )
            assert included[valid].all()


@GPU
def test_product_old_vjp_uses_its_forward_snapshot():
    layer = model(torch.tensor([[0.3, 1.5, 2.4, 24.0]]), device="cuda")
    ref = copy.deepcopy(layer)
    x = torch.randn(7, 65, device="cuda", requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    dy = torch.randn(7, 33, device="cuda")
    actual = run(layer, x)
    truth = ref.operator.apply(xx, algorithm="factored")
    with torch.no_grad():
        layer.atoms.p[:, 2:].add_(0.4)
        layer.kernel.amplitude_max.mul_(0.7)
    run(layer, x.detach())
    for a, b in zip(
        torch.autograd.grad(actual, (x, layer.atoms.p), dy),
        torch.autograd.grad(truth, (xx, ref.atoms.p), dy),
    ):
        torch.testing.assert_close(a, b, rtol=4e-4, atol=2e-5)


@GPU
@pytest.mark.parametrize("preparation", ["full", "support"])
def test_product_captured_updates_moments_and_live_widths(preparation):
    from benchmarks.cuda.linear.profile_product import decode
    from benchmarks.cuda.polar_update import optimizer_step
    from torchcst import AtomUpdateBinding, CSTOptimizer

    layer = model(
        torch.tensor([[0.3, 1.5, 2.4, 24.0], [-0.4, 1.4, 6.3, 32.7]]), device="cuda"
    )
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3, fused=True, capturable=True)
    binding = AtomUpdateBinding(layer.operator)
    x = torch.randn(7, 65, device="cuda")
    dy = torch.randn(7, 33, device="cuda")
    initial = decode(layer.kernel, layer.atoms.p)[:, 1].detach().clone()

    def step():
        opt.zero_grad(set_to_none=True)
        y = run(layer, x, preparation)
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
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                opt.state[layer.atoms.p][key],
                base.state[reference.atoms.p][key],
                rtol=4e-4,
                atol=2e-5,
            )
    assert torch.any(decode(layer.kernel, layer.atoms.p)[:, 1] != initial)
