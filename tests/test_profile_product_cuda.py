"""Metadata contracts and actual-GPU product normalization/source VJP checks."""

import copy
from dataclasses import replace

import pytest
import torch

from benchmarks.cuda.linear.manifest import REGISTRY, load_run
from benchmarks.cuda.linear.profile_product import oracle_atoms
from torchcst import (
    BandwidthBounds,
    CSTLinear,
    CSTOptimizer,
    Dispatcher,
    LinearInputs,
    NormalizationSpec,
    TriweightSpec,
    chart_presets,
    pattern_presets,
    presets,
)
from torchcst._backends.cuda.algorithms.linear.profile_product.algorithm import (
    ProductAlgorithm,
    domain,
)
from torchcst._backends.cuda.algorithms.linear.profile_product.recipe import (
    ProductRecipe,
)
from torchcst._backends.schema import ExecutionPlan

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


def spec(*, width=None, floor=1e-6):
    bounds = BandwidthBounds(
        minimum=0.25 if width is None else width,
        birth=1.0 if width is None else width,
        maximum=16.0 if width is None else width,
        upper_floor=1.0 if width is None else width,
    )
    return presets.polar_profile_product(
        profiles=(TriweightSpec(), TriweightSpec()),
        amplitude_max=1.0,
        bounds=bounds,
        w_c=1e6,
        normalization=NormalizationSpec(
            kind="discrete_l2", domain="operator_sites", floor=floor
        ),
        dormant_expansion_rate=0.02,
    )


def model(p, *, width=None, floor=1e-6, n=16, device="cpu"):
    return CSTLinear(
        chart=chart_presets.product(
            (n, n),
            (pattern_presets.line(n, low=0, high=n - 1),) * 2,
        ),
        atoms=p,
        kernel=spec(width=width, floor=floor),
        device=device,
    )


def run(layer, x, route, *, atom_block=16):
    plan = ExecutionPlan(
        "research_profile_product",
        "v1",
        ProductRecipe(execution_route=route, atom_block=atom_block),
    )
    return Dispatcher(registry=REGISTRY).run(layer, LinearInputs(x), plan=plan)


@pytest.mark.parametrize("route", ["torch", "local", "saved", "ordered", "reuse"])
@pytest.mark.parametrize("block", [16, 32])
def test_metadata_and_json(route, block):
    plan = ExecutionPlan(
        "research_profile_product",
        "v1",
        ProductRecipe(execution_route=route, atom_block=block),
    )
    assert plan == REGISTRY.loads_plan(REGISTRY.dumps_plan(plan))
    layer = model(torch.tensor([[0.2, 1.4, 2.1, 4.3]]))
    d = domain(layer.declaration())
    assert (d.input_size, d.output_size) == (16, 16)
    context = layer.build_context(LinearInputs(torch.randn(3, 16)))
    assert not ProductAlgorithm().supports(context, plan.recipe).supported


def test_rejects_unsupported_geometry_profiles_and_recipe():
    layer = model(torch.tensor([[0.2, 1.4, 2.1, 4.3]]))
    declaration = layer.declaration()
    from torchcst import GaussianSpec

    with pytest.raises(ValueError):
        domain(
            replace(
                declaration,
                kernel=replace(
                    declaration.kernel,
                    profiles=(
                        replace(declaration.kernel.profiles[0], profile=GaussianSpec()),
                        declaration.kernel.profiles[1],
                    ),
                ),
            )
        )
    with pytest.raises(ValueError):
        ProductRecipe(pack=True)
    with pytest.raises(ValueError):
        ProductRecipe(execution_route="unknown")


@pytest.mark.parametrize("n", [64, 128])
@pytest.mark.parametrize("profile", ["rho1_25", "rho3", "rho8"])
def test_runner_fixture_roundtrip(n, profile):
    path = f"benchmarks/cuda/linear/cases/profile-product-{n}-{profile}.json"
    run = load_run(path, "benchmarks/cuda/linear/plans-profile-product.json")
    from benchmarks.cuda.linear.manifest import decode_snapshot

    assert decode_snapshot(run.snapshot()) == run
    from dataclasses import asdict

    from benchmarks.cuda.linear.protocol import measurement_operator

    assert measurement_operator(asdict(run.case)) == domain_operator(run.case)


def domain_operator(case):
    from benchmarks.cuda.linear.profile_product import fixture_operator

    return fixture_operator(case)


@GPU
@pytest.mark.parametrize("route", ["local", "saved", "ordered", "reuse"])
@pytest.mark.parametrize(
    "n,rows,block", [(16, 1, 16), (32, 7, 32), (64, 32, 16), (128, 64, 32)]
)
def test_full_site_oracle_y_dx_all_source(route, n, rows, block):
    torch.manual_seed(41)
    p = torch.randn(19, 4)
    p[:, :2] = p[:, :2] * 0.1 + p.new_tensor([0.3, 1.5])
    p[:, 2:] = torch.rand(19, 2) * (n - 1)
    layer = model(p, n=n, device="cuda")
    x = torch.randn(rows, n, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    tx = x.detach().double().requires_grad_()
    tp = layer.atoms.p.detach().double().requires_grad_()
    state = copy.deepcopy(layer.kernel).double()
    truth = tx @ oracle_atoms(state, tp, n).sum(0).T
    actual = run(layer, x, route, atom_block=block)
    gradients = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    expected = torch.autograd.grad(truth, (tx, tp), dy.double())
    torch.testing.assert_close(actual.double(), truth, rtol=4e-4, atol=2e-5)
    for a, b in zip(gradients, expected):
        torch.testing.assert_close(a.double(), b, rtol=4e-4, atol=2e-5)
    assert torch.all(expected[1][:, 2:].abs() > 1e-8)


@GPU
@pytest.mark.parametrize("route", ["local", "saved", "ordered", "reuse"])
@pytest.mark.parametrize("case", ["empty", "singleton", "tiny", "global_floor"])
def test_global_floor_and_support_gradients(route, case):
    floor = 0.5 if case == "global_floor" else 1e-6
    center = {
        "empty": 10.0,
        "singleton": 0.0,
        "tiny": -0.999,
        "global_floor": (1 - 0.6 ** (1 / 3)) ** 0.5,
    }[case]
    # Width one: neighbors exactly on the zero boundary for the singleton.
    layer = model(
        torch.tensor([[0.3, 1.2, center, center]]),
        width=1.0,
        n=2,
        floor=floor,
        device="cuda",
    )
    x = torch.tensor([[0.4, -0.3]], device="cuda", requires_grad=True)
    y = run(layer, x, route)
    tx, tp = (
        x.detach().double().requires_grad_(),
        layer.atoms.p.detach().double().requires_grad_(),
    )
    state = copy.deepcopy(layer.kernel).double()
    truth = tx @ oracle_atoms(state, tp, 2).sum(0).T
    a = torch.autograd.grad(y.sum(), (x, layer.atoms.p))
    b = torch.autograd.grad(truth.sum(), (tx, tp))
    torch.testing.assert_close(y.double(), truth, rtol=8e-4, atol=2e-6)
    for ga, gb in zip(a, b):
        assert torch.isfinite(ga).all()
        torch.testing.assert_close(ga.double(), gb, rtol=8e-4, atol=2e-5)
    if case == "global_floor":
        assert b[1][:, 2:].abs().min() > 0.01


@GPU
@pytest.mark.parametrize("route", ["local", "saved", "ordered", "reuse"])
def test_saved_snapshot_after_source_and_scalar_changes(route):
    layer = model(torch.tensor([[0.3, 1.5, 2.4, 4.2]]), device="cuda")
    ref = copy.deepcopy(layer)
    x = torch.randn(7, 16, device="cuda", requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    dy = torch.randn_like(x)
    y = run(layer, x, route)
    truth = ref.operator.apply(xx, algorithm="factored")
    with torch.no_grad():
        layer.atoms.p[:, 2:].add_(0.4)
        layer.kernel.amplitude_max.mul_(0.7)
    _ = run(layer, x.detach(), route)
    a = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    b = torch.autograd.grad(truth, (xx, ref.atoms.p), dy)
    for ga, gb in zip(a, b):
        torch.testing.assert_close(ga, gb, rtol=4e-4, atol=2e-5)


@GPU
@pytest.mark.parametrize("route", ["local", "saved", "ordered", "reuse"])
@pytest.mark.parametrize("update", ["torch", "fused"])
def test_captured_updates_parameters_moments_and_evolving_width(route, update):
    torch.manual_seed(13)
    layer = model(
        torch.tensor([[0.3, 1.5, 2.4, 4.2], [-0.4, 1.4, 6.3, 8.7]]), device="cuda"
    )
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3, fused=True, capturable=True)
    from benchmarks.cuda.polar_update import optimizer_step
    from torchcst import AtomUpdateBinding

    binding = AtomUpdateBinding(layer.operator)
    x = torch.randn(7, 16, device="cuda")
    target = torch.randn_like(x)
    from benchmarks.cuda.linear.profile_product import decode

    initial = decode(layer.kernel, layer.atoms.p).detach().clone()[:, 1]

    def step():
        opt.zero_grad(set_to_none=True)
        y = run(layer, x, route)
        (y * target).sum().backward()
        optimizer_step(binding, opt, step_size=1e-3, polar_update=update)
        return y

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        actual_y = step()
    torch.cuda.synchronize()
    reference = copy.deepcopy(layer)
    base = torch.optim.AdamW(reference.parameters(), lr=1e-3, fused=True)
    base.load_state_dict(copy.deepcopy(opt.state_dict()))
    for group in base.param_groups:
        group["capturable"] = False
    reference_opt = CSTOptimizer(base, model=reference)
    for _ in range(20):
        reference_opt.zero_grad()
        expected_y = reference.operator.apply(x, algorithm="factored")
        (expected_y * target).sum().backward()
        reference_opt.step()
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual_y, expected_y, rtol=2e-4, atol=2e-5)
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


@GPU
@pytest.mark.parametrize("route", ["local", "saved", "ordered", "reuse"])
def test_rectangular_shifted_sites_full_matrix_oracle(route):
    torch.manual_seed(57)
    p = torch.randn(19, 4)
    p[:, :2] = p[:, :2] * 0.1 + p.new_tensor([0.3, 1.5])
    p[:, 2] = -3 + torch.rand(19) * 8
    p[:, 3] = 7 + torch.rand(19) * 15
    layer = CSTLinear(
        chart=chart_presets.product(
            (17, 31),
            (
                pattern_presets.line(17, low=-3, high=5),
                pattern_presets.line(31, low=7, high=22),
            ),
        ),
        atoms=p,
        kernel=spec(width=3),
        device="cuda",
    )
    x = torch.randn(9, 31, device="cuda", requires_grad=True)
    dy = torch.randn(9, 17, device="cuda")
    tx = x.detach().double().requires_grad_()
    tp = layer.atoms.p.detach().double().requires_grad_()
    out_sites = torch.linspace(-3, 5, 17, device="cuda", dtype=torch.float64)
    in_sites = torch.linspace(7, 22, 31, device="cuda", dtype=torch.float64)
    u = (1 - (out_sites - tp[:, 2, None]).square() / 9).clamp_min(0).pow(3)
    v = (1 - (in_sites - tp[:, 3, None]).square() / 9).clamp_min(0).pow(3)
    raw = u[:, :, None] * v[:, None, :]
    norm = raw.flatten(1).norm(dim=1).clamp_min(1e-6)
    amp = tp[:, 0] / tp[:, :2].norm(dim=1)
    truth = tx @ (raw * (amp / norm)[:, None, None]).sum(0).T
    actual = run(layer, x, route)
    ga = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    gb = torch.autograd.grad(truth, (tx, tp), dy.double())
    torch.testing.assert_close(actual.double(), truth, rtol=4e-4, atol=2e-5)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a.double(), b, rtol=4e-4, atol=2e-5)
    assert torch.all(gb[1][:, 2:].abs() > 1e-8)


def test_reuse_matches_tuned_small_core_launch_and_layout():
    from benchmarks.cuda.linear.local_product import LocalRecipe

    previous = LocalRecipe(
        route="persistent_supportprep_band_recompute_vjp_ordered_reuse_histtightinline_paramatom16_param4_param2",
        atom_block=32,
        pack=False,
    )
    candidate = ProductRecipe(execution_route="reuse", atom_block=32)
    for field in (
        "atom_block",
        "batch_block",
        "output_block",
        "parameter_atom_block",
        "parameter_batch_block",
        "parameter_warps",
        "contraction_warps",
        "owner_splits",
        "parameter_splits",
        "order_by_position",
        "compact_order_key",
        "parallel_owner_ranges",
        "parallel_order_copy",
        "histogram_owner_ranges",
        "fused_histogram_owner_ranges",
        "tight_histogram_owner_ranges",
        "preparation_warps",
        "band_dispatch",
        "vector_support",
        "recompute_h",
        "recompute_param_h",
    ):
        assert getattr(candidate, field) == getattr(previous, field), field
