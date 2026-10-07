"""Explicit size boundaries and complete-site gradients on larger charts."""

from dataclasses import replace

import pytest
import test_grouped_profile_product_cuda as scenarios
import torch

from benchmarks.cuda.linear.manifest import REGISTRY, decode_snapshot, load_run
from torchcst import Dispatcher, LinearInputs
from torchcst._backends.cuda.algorithms.linear.profile_product_global.recipe_v3 import (
    PreparationProductRecipe,
    PreparationStripRecipe,
)
from torchcst._backends.cuda.algorithms.linear.profile_product_matrix.recipe_v3 import (
    GroupedMatrixProductRecipe,
    GroupedMatrixStripRecipe,
)
from torchcst._backends.schema import DeviceInfo, ExecutionPlan


def plan(family, route="native", **kwargs):
    prefix = (
        "research_profile_product"
        if family == "global"
        else "research_strip_profile_product"
    )
    if route == "prepared":
        cls = PreparationProductRecipe if family == "global" else PreparationStripRecipe
        suffix = "prepared"
    else:
        cls = (
            GroupedMatrixProductRecipe
            if family == "global"
            else GroupedMatrixStripRecipe
        )
        kwargs["gemm"] = "triton" if route == "native" else "torch"
        suffix = "matrix"
    return ExecutionPlan(f"{prefix}_large_{suffix}", "v1", cls(**kwargs))


def run(layer, x, family, route="native", **kwargs):
    return Dispatcher(registry=REGISTRY).run(
        layer, LinearInputs(x), plan=plan(family, route, **kwargs)
    )


@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("route", ["prepared", "native", "torch"])
def test_explicit_large_boundaries_and_old_plans_remain_small(family, route):
    p = torch.tensor([[0.3, 1.5, 2.4, 24]])
    layer = scenarios.model(p, family, n=8192, out=8192 if family == "global" else 65)
    context = replace(
        layer.build_context(LinearInputs(torch.randn(7, 8192))),
        device=DeviceInfo("cuda", 0),
    )
    selected = plan(family, route)
    assert REGISTRY.loads_plan(REGISTRY.dumps_plan(selected)) == selected
    algorithm = REGISTRY.get(selected.algorithm_id, revision="v1")
    assert algorithm.supports(context, selected.recipe).supported
    assert not algorithm.supports(
        replace(context, atom_count=4194305), selected.recipe
    ).supported
    old_id = selected.algorithm_id.replace(
        "_large_prepared", "_global" if family == "global" else ""
    ).replace("_large_matrix", "_matrix")
    old = REGISTRY.get(old_id, revision="v3")
    assert not old.supports(context, selected.recipe).supported
    too_large = scenarios.model(p, family, n=8193, out=33)
    bad = replace(
        too_large.build_context(LinearInputs(torch.randn(7, 8193))),
        device=DeviceInfo("cuda", 0),
    )
    assert not algorithm.supports(bad, selected.recipe).supported


@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("n", [2048, 8192])
@pytest.mark.parametrize("rho", [3, 8])
def test_large_case_snapshots_and_live_update_binding(family, n, rho):
    from dataclasses import asdict

    from benchmarks.cuda.linear.protocol import measurement_operator
    from benchmarks.cuda.linear.run import PlanLinear

    value = load_run(
        f"benchmarks/cuda/linear/cases/profile-product-large-{family}-{n}-rho{rho}.json",
        f"benchmarks/cuda/linear/plans-profile-product-large-{family}.json",
    )
    assert decode_snapshot(value.snapshot()) == value
    operator = measurement_operator(asdict(value.case))
    layer = PlanLinear(
        torch.tensor([[0.3, 1.5, 2.4, 24]]), operator, value.entry("native-g8-p8").plan
    )
    assert layer.update_binding is not None
    assert layer.local_state is layer.live_operator.kernel


@scenarios.GPU
@pytest.mark.parametrize(
    "family,n,out,b",
    [("global", 2049, 2051, 7), ("global", 8192, 8192, 1), ("strip", 8191, 65, 7)],
)
@pytest.mark.parametrize("route", ["prepared", "native", "torch"])
@pytest.mark.parametrize("preparation", ["full", "support"])
def test_large_complete_sites_and_all_atom_gradients(
    family, n, out, b, route, preparation
):
    torch.manual_seed(41)
    a = 19
    p = torch.randn(a, 4) * 0.1 + torch.tensor([0.3, 1.5, 0, 0])
    p[:, 2] = torch.rand(a) * (out - 2) + 0.37
    p[:, 3] = torch.rand(a) * (n - 2) + 0.37
    if family == "strip":
        logical = p[:, 3].floor()
        p[:, 3] = (
            logical.remainder(16) + logical.div(16, rounding_mode="floor") * 20 + 0.37
        )
    layer = scenarios.model(p, family, n=n, out=out, device="cuda")
    x = torch.randn(b, n * 2, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(out, b, device="cuda").T
    y, dx, dp = scenarios.oracle(layer, x, dy, family)
    actual = run(layer, x, family, route, preparation=preparation)
    gx, gp = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    for value, truth in [(actual, y), (gx, dx), (gp, dp)]:
        torch.testing.assert_close(value.double(), truth, rtol=4e-4, atol=2e-5)


@pytest.fixture(params=["prepared", "native", "torch"])
def large_route(request, monkeypatch):
    original = scenarios.model

    def model(p, family, **kwargs):
        kwargs.setdefault("n", 2048 if family == "global" else 8191)
        kwargs.setdefault("out", 2048 if family == "global" else 65)
        return original(p, family, **kwargs)

    def execute(layer, x, family, **kwargs):
        return run(layer, x, family, request.param, **kwargs)

    monkeypatch.setattr(scenarios, "model", model)
    monkeypatch.setattr(scenarios, "run", execute)


@pytest.mark.usefixtures("large_route")
class TestLargeSavedState:
    test_updates = staticmethod(
        scenarios.test_twenty_captured_updates_match_reference_moments_and_live_widths
    )
    test_snapshots = staticmethod(
        scenarios.test_old_forward_snapshots_keep_pitch_amplitude_and_parameters
    )
    test_live_pitch = staticmethod(
        scenarios.test_captured_strip_reads_changed_pitch_and_spacing_fallback
    )


@scenarios.GPU
def test_large_sorted_views_preserve_keys_above_int32():
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.profile_product_global.kernels import (
        copy_views,
        make_keys,
    )

    a = 300001
    indices = torch.arange(a, device="cuda")
    packed = indices.float().expand(13, a).clone()
    packed[11] = torch.where(indices.remainder(2) == 0, 8191, 0)
    packed[9] = torch.where(indices.remainder(3) == 0, 8192, 4096)
    keys = torch.empty((2, a), device="cuda", dtype=torch.int64)
    make_keys[(tr.cdiv(a, 256), 2)](packed, keys, a, 256, True, num_warps=4)
    expected = torch.stack([packed[11].long(), packed[9].long()]) * (a + 1) + indices
    assert expected.max() > 2**31 - 1
    torch.testing.assert_close(keys, expected, rtol=0, atol=0)
    keys = keys.sort(dim=1).values
    views = torch.empty((2, 13, a), device="cuda")
    order = torch.empty((2, a), device="cuda", dtype=torch.int32)
    inverse = torch.empty(a, device="cuda", dtype=torch.int32)
    copy_views[(tr.cdiv(a, 256), 2)](
        packed, keys, views, order, inverse, a, 256, num_warps=4
    )
    truth_order = expected.argsort(dim=1)
    torch.testing.assert_close(order.long(), truth_order, rtol=0, atol=0)
    for direction in (0, 1):
        torch.testing.assert_close(
            views[direction], packed[:, truth_order[direction]], rtol=0, atol=0
        )
    torch.testing.assert_close(inverse[truth_order[0]].long(), indices, rtol=0, atol=0)
