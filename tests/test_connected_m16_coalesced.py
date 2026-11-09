import pytest
import torch

from torchcst._backends.cuda.algorithms.linear.connected_cell_mass_coalesced.algorithm import (
    CoalescedRecipe,
    ConnectedCoalescedAlgorithm,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("rows", [1, 7, 65, 256])
def test_cuda_independent_vjp_retained_and_multiple_forwards(rows):
    from benchmarks.cuda.linear.connected_cell_mass import (
        error,
        operator,
        oracle,
        parameters,
    )
    from benchmarks.cuda.linear.manifest import REGISTRY
    from benchmarks.cuda.linear.run import PlanLinear
    from torchcst._backends.schema import ExecutionPlan

    torch.manual_seed(92)
    p = parameters(128, 10, 34)
    p[:, 1] = torch.tensor([0.5, 0.72, 0.8, 0.9, 1.44, 1.8, 7, 16, 31.9, 50])
    p[:, 2] = 7.5
    p[:, 3] = 13.5
    plan = ExecutionPlan(ConnectedCoalescedAlgorithm().id, "v1", CoalescedRecipe())
    REGISTRY.validate_plan(plan)
    model = PlanLinear(p.cuda(), operator(128, 16, True), plan)
    x = torch.randn(rows, 256, device="cuda")[:, ::2].requires_grad_()
    dy = torch.randn(rows, 256, device="cuda")[:, ::2]
    y = model(x)
    later = model(x * 1.1)
    gx, gp = torch.autograd.grad(y, (x, model.p), dy, retain_graph=True)
    rx = x.detach().cpu().double().requires_grad_()
    rp = p.double().requires_grad_()
    ry = oracle(rx, rp, 16, True)
    ex, ep = torch.autograd.grad(ry, (rx, rp), dy.cpu().double())
    for value, expected in ((y.cpu(), ry), (gx.cpu(), ex), (gp.cpu(), ep)):
        e = error(value, expected)
        assert e["max_abs"] < 3e-4 and e["relative_l2"] < 3e-4, e
    gx2, gp2 = torch.autograd.grad(y, (x, model.p), dy)
    torch.testing.assert_close(gx2, gx, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(gp2, gp, atol=2e-6, rtol=2e-5)
    later.sum().backward()
    stale = model(x)
    with torch.no_grad():
        model.p.add_(0.001)
    with pytest.raises(RuntimeError, match="modified|version"):
        stale.sum().backward()


def test_coalesced_catalog_roundtrip():
    from dataclasses import replace

    from benchmarks.cuda.linear.manifest import REGISTRY, load_run

    run = load_run(
        "benchmarks/cuda/linear/cases/connected-radius-2048-r256-c8-coalesced.json",
        "benchmarks/cuda/linear/plans-connected-radius-coalesced.json",
    )
    for entry in run.plans:
        assert REGISTRY.loads_plan(REGISTRY.dumps_plan(entry.plan)) == entry.plan
    with pytest.raises(ValueError):
        ConnectedCoalescedAlgorithm().validate_recipe(
            replace(CoalescedRecipe(), modes=8)
        )
