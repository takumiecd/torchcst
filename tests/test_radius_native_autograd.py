"""The optional host graph changes crossings, never CUDA math or saved caches."""

from dataclasses import replace

import pytest
import torch
from test_radius_cell_mass_cuda import GPU, atoms, axis_plan
from test_radius_cell_mass_hybrid import check_vjp, registry

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import BLOCK_LOCAL


def recipe(schedule="block_local_cached_parallel", **kwargs):
    return replace(
        BLOCK_LOCAL.recipe, schedule=schedule, native_autograd=True, **kwargs
    )


def test_native_host_graph_is_explicit_and_rejects_other_schedules():
    reg = registry()
    plan = replace(BLOCK_LOCAL, recipe=recipe())
    assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for changes in (
        {"native_autograd": 1},
        {"schedule": "sequential"},
        {"schedule": "overlap"},
    ):
        with pytest.raises(ValueError, match="native_autograd"):
            reg.validate_plan(replace(plan, recipe=replace(plan.recipe, **changes)))


@GPU
@pytest.mark.parametrize(
    "schedule", ["block_local", "block_local_parallel", "block_local_cached_parallel"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "sites,modes,start,tile,merge,method",
    [
        (128, 8, 2, 32, 2, "merge"),
        (27, 3, 1, 13, 3, "merge"),
        (16, 5, 5, 7, 2, "direct"),
    ],
)
def test_native_host_all_vjps_against_independent_oracle(
    schedule, dtype, sites, modes, start, tile, merge, method
):
    axis = axis_plan(sites, modes=modes, merge=merge)
    plan = HierarchyPlan(input=axis, output=axis)
    cfg = recipe(
        schedule,
        input_direct_levels=start,
        output_direct_levels=start,
        output_tile_sites=tile,
        moment_method=method,
    )
    gen = torch.Generator().manual_seed(985)
    check_vjp(
        plan,
        cfg,
        torch.randn(3, sites, dtype=dtype, generator=gen),
        atoms(plan, dtype),
        torch.randn(3, sites, dtype=dtype, generator=gen),
    )


@GPU
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (True, False), (False, True)]
)
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_native_host_matches_python_bits_retained_vjp_and_saved_hooks(
    needs_x, needs_p, offload, empty
):
    axis = axis_plan(32, modes=3)
    plan = HierarchyPlan(input=axis, output=axis)
    x = torch.randn(2, 32, dtype=torch.float64, device="cuda", requires_grad=needs_x)
    p = torch.empty(0, 4, dtype=x.dtype) if empty else atoms(plan, x.dtype)
    p = p.cuda().requires_grad_(needs_p)
    dy = torch.randn(2, 32, dtype=x.dtype, device="cuda")
    selected = tuple(t for t in (x, p) if t.requires_grad)
    seen = []

    def pack(t):
        seen.append((t.shape, t.device))
        return (t.detach().cpu().clone(), t.device) if offload else t

    def unpack(value):
        return value[0].to(value[1]) if offload else value

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = ex.radius_linear(plan, x, p, recipe=recipe())
    assert "RadiusLinear" in y.grad_fn.name()
    # X, current atoms, completed moments/H; routing keys only when dX needs them.
    assert len(seen) == (5 if needs_x else 4)
    actual = torch.autograd.grad(y, selected, dy, retain_graph=True)
    y_ref = ex.radius_linear(
        plan, x, p, recipe=replace(recipe(), native_autograd=False)
    )
    expected = torch.autograd.grad(y_ref, selected, dy)
    for a, b in zip((y, *actual), (y_ref, *expected)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    repeated = torch.autograd.grad(y, selected, dy)
    for a, b in zip(actual, repeated):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="second time|freed"):
        torch.autograd.grad(y, selected, dy)


@GPU
def test_native_host_saved_version_check_and_second_derivative_rejection():
    axis = axis_plan(16, modes=3)
    plan = HierarchyPlan(input=axis, output=axis)
    x = torch.randn(2, 16, dtype=torch.float64, device="cuda", requires_grad=True)
    p = atoms(plan, x.dtype).cuda().requires_grad_()
    y = ex.radius_linear(plan, x, p, recipe=recipe())
    with torch.no_grad():
        p.add_(0.0)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        y.sum().backward()
    y = ex.radius_linear(plan, x, p, recipe=recipe())
    gradient = torch.autograd.grad(y.square().sum(), x, create_graph=True)[0]
    assert gradient.requires_grad
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), x)
