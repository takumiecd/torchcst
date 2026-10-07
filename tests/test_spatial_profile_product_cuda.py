"""Spatial support permutation, complete canonical VJPs and saved Graph updates."""

import pytest
import test_grouped_profile_product_cuda as scenarios
import torch

from benchmarks.cuda.linear.manifest import REGISTRY, decode_snapshot, load_run
from torchcst import Dispatcher, LinearInputs
from torchcst._backends.cuda.algorithms.linear.profile_product_matrix.recipe_v5 import (
    SpatialMatrixProductRecipe,
    SpatialMatrixStripRecipe,
)
from torchcst._backends.schema import ExecutionPlan


def plan(family, large=False, **kw):
    prefix = (
        "research_profile_product"
        if family == "global"
        else "research_strip_profile_product"
    )
    cls = SpatialMatrixProductRecipe if family == "global" else SpatialMatrixStripRecipe
    return ExecutionPlan(
        f"{prefix}{'_large' if large else ''}_matrix",
        "v3" if large else "v5",
        cls(**kw),
    )


def run(layer, x, family, large=False, **kw):
    return Dispatcher(registry=REGISTRY).run(
        layer, LinearInputs(x), plan=plan(family, large, **kw)
    )


@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("tile", [16, 32])
def test_roundtrip_and_previous_revision_rejects_spatial(family, large, tile):
    value = plan(family, large, spatial_tile=tile)
    assert REGISTRY.loads_plan(REGISTRY.dumps_plan(value)) == value
    data = REGISTRY.dump_plan(value)
    data["algorithm_revision"] = "v2" if large else "v4"
    with pytest.raises(ValueError):
        REGISTRY.load_plan(data)


@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("invalid", [True, 0, 1, 8, 64, 16.0, None])
def test_spatial_rejects_other_settings(family, invalid):
    with pytest.raises(ValueError):
        plan(family, spatial_tile=invalid)


@pytest.mark.parametrize("family", ["global", "strip"])
def test_spatial_rejects_torch_engine(family):
    with pytest.raises(ValueError):
        plan(family, gemm="torch")


@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("large,size", [(False, 1024), (True, 2048), (True, 8192)])
@pytest.mark.parametrize("rho", [3, 8])
def test_spatial_case_snapshots_and_live_optimizer_binding(family, large, size, rho):
    from dataclasses import asdict

    from benchmarks.cuda.linear.protocol import measurement_operator
    from benchmarks.cuda.linear.run import PlanLinear

    tag = "large-" if large else ""
    value = load_run(
        f"benchmarks/cuda/linear/cases/profile-product-{tag}{family}-{size}-rho{rho}-spatial.json",
        f"benchmarks/cuda/linear/plans-profile-product-{tag}{family}-spatial.json",
    )
    assert decode_snapshot(value.snapshot()) == value
    layer = PlanLinear(
        torch.tensor([[0.3, 1.5, 2.4, 24]]),
        measurement_operator(asdict(value.case)),
        value.entry("spatial16").plan,
    )
    assert layer.update_binding is not None
    assert layer.local_state is layer.live_operator.kernel


@pytest.fixture(params=[(False, 16), (False, 32), (True, 16)])
def spatial_route(request, monkeypatch):
    large, tile = request.param
    original = scenarios.model

    def model(p, family, **kw):
        if large:
            kw.setdefault("n", 2049 if family == "global" else 8191)
            kw.setdefault("out", 2051 if family == "global" else 65)
        return original(p, family, **kw)

    def execute(layer, x, family, **kw):
        return run(
            layer,
            x,
            family,
            large,
            spatial_tile=tile,
            split_k=8 if family == "strip" else 1,
            **kw,
        )

    monkeypatch.setattr(scenarios, "model", model)
    monkeypatch.setattr(scenarios, "run", execute)


@pytest.mark.usefixtures("spatial_route")
class TestSpatialOperator:
    test_all_sites = staticmethod(
        scenarios.test_grouped_all_sites_and_canonical_gradients
    )
    test_floor = staticmethod(
        scenarios.test_grouped_global_floor_and_full_support_fallback
    )
    test_pitch = staticmethod(
        scenarios.test_strip_pitch_boundaries_gaps_and_partial_tile_have_exact_full_norm
    )
    test_snapshots = staticmethod(
        scenarios.test_old_forward_snapshots_keep_pitch_amplitude_and_parameters
    )
    test_gradients = staticmethod(scenarios.test_zero_atoms_and_gradient_branches)
    test_updates = staticmethod(
        scenarios.test_twenty_captured_updates_match_reference_moments_and_live_widths
    )
    test_live_pitch = staticmethod(
        scenarios.test_captured_strip_reads_changed_pitch_and_spacing_fallback
    )


@scenarios.GPU
@pytest.mark.parametrize("family", ["global", "strip"])
def test_large_spatial_complete_chart_with_tails(family):
    torch.manual_seed(41)
    n, out = (2049, 2051) if family == "global" else (8191, 65)
    p = torch.tensor([[0.3, 1.5, 2.4, 24], [-0.4, 1.4, 6.3, 32.7]], device="cuda")
    layer = scenarios.model(p, family, n=n, out=out, device="cuda")
    x = torch.randn(7, n * 2, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(out, 7, device="cuda").T
    truth = scenarios.oracle(layer, x, dy, family)
    y = run(layer, x, family, True, spatial_tile=32)
    dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for value, reference in zip((y, dx, dp), truth, strict=True):
        torch.testing.assert_close(value.double(), reference, rtol=4e-4, atol=2e-5)


@scenarios.GPU
@pytest.mark.parametrize("ni,no", [(1, 1), (65, 33), (8192, 8192)])
@pytest.mark.parametrize("atoms", [1, 17, 257])
@pytest.mark.parametrize("tile", [16, 32])
def test_ordered_payload_is_exact_bijection_with_wide_keys(ni, no, atoms, tile):
    from torchcst._backends.cuda.algorithms.linear.profile_product_matrix.spatial_kernels import (
        order_support,
        support_keys,
    )

    torch.manual_seed(41)
    packed = torch.randn(13, atoms, device="cuda")
    lows = torch.randint(-9, ni + 7, (atoms,)).tolist()
    outs = torch.randint(-9, no + 7, (atoms,)).tolist()
    packed[9] = torch.tensor(lows, device="cuda")
    packed[11] = torch.tensor(outs, device="cuda")
    # Independent lexicographic semantic oracle; canonical ID breaks ties.
    truth = sorted(
        range(atoms),
        key=lambda a: (
            min(max(outs[a], 0), no - 1) // tile,
            min(max(lows[a], 0), ni - 1) // tile,
            a,
        ),
    )
    ordered, order = order_support(packed, ni, no, tile)
    assert order.cpu().tolist() == truth
    torch.testing.assert_close(ordered, packed[:, truth], rtol=0, atol=0)
    assert sorted(order.cpu().tolist()) == list(range(atoms))
    keys = torch.empty(atoms, dtype=torch.int64, device="cuda")
    support_keys[((atoms + 255) // 256,)](packed, keys, atoms, ni, no, tile, 256)
    if ni > 1 and atoms > 1:
        assert bool((keys > 2**31).any())


@scenarios.GPU
def test_empty_order_has_no_launch_or_atom_remapping():
    from torchcst._backends.cuda.algorithms.linear.profile_product_matrix.spatial_kernels import (
        order_support,
    )

    p = torch.empty((13, 0), device="cuda")
    ordered, order = order_support(p, 65, 33, 16)
    assert ordered is p and order.shape == (0,) and order.dtype == torch.int32


@scenarios.GPU
@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("tile", [16, 32])
def test_captured_center_permutation_keeps_old_forward_and_canonical_vjps(family, tile):
    """Move owners across tiles without changing Parameter/moment row identities."""
    torch.manual_seed(41)
    p = torch.tensor(
        [
            [0.3, 1.5, 4.4, 100.4],
            [-0.4, 1.4, 36.4, 4.4],
            [0.2, 1.3, 20.4, 68.4],
            [0.5, 1.6, 52.4, 36.4],
        ],
        device="cuda",
    )
    layer = scenarios.model(p, family, n=129, out=65, device="cuda")
    x = torch.randn(7, 129, device="cuda", requires_grad=True)
    dy = torch.randn(7, 65, device="cuda")
    kwargs = {"spatial_tile": tile, "split_k": 8 if family == "strip" else 1}
    initial = scenarios.oracle(layer, x, dy, family)
    retained = run(layer, x, family, **kwargs)

    def probe():
        y = run(layer, x, family, **kwargs)
        # Keep the algorithm's saved physical-to-canonical map as a live
        # diagnostic buffer. Its values must rebuild during Graph replay.
        order = y.grad_fn.saved_tensors[-1]
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
        return y, dx, dp, order

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            probe()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = probe()
    torch.cuda.synchronize()
    first_order = actual[3].cpu().tolist()
    with torch.no_grad():
        layer.atoms.p[:, 2:].copy_(layer.atoms.p[:, 2:].flip(0))
    moved = scenarios.oracle(layer, x, dy, family)
    graph.replay()
    torch.cuda.synchronize()
    assert actual[3].cpu().tolist() != first_order
    assert sorted(actual[3].cpu().tolist()) == list(range(len(p)))
    for value, truth in zip(actual[:3], moved, strict=True):
        torch.testing.assert_close(value.double(), truth, rtol=4e-4, atol=2e-5)
    old_grads = torch.autograd.grad(retained, (x, layer.atoms.p), dy)
    for value, truth in zip((retained, *old_grads), initial, strict=True):
        torch.testing.assert_close(value.double(), truth, rtol=4e-4, atol=2e-5)
