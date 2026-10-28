"""Masked C expansion gates; finite inactive C is zero by the producer contract."""

import runpy
from pathlib import Path

import pytest
import torch

from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical import executor
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.plan import (
    AxisPlan,
    HierarchyPlan,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def plan_for(n):
    return HierarchyPlan(
        input=AxisPlan(site_count=n, min_tile_sites=4, spacing=128 / n),
        output=AxisPlan(site_count=n, min_tile_sites=8, spacing=128 / n),
    )


def masks_for(plan, atoms):
    layout = executor.prepare_layout(plan, atoms, reverse=True)
    return executor._extension().prepare_bank_masks(
        layout.groups,
        layout.offsets,
        plan.input.native_config(),
        plan.output.native_config(),
    )


@pytest.mark.parametrize("n", [128, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", ["all", "sparse", "empty", "zero_c"])
def test_active_expand_raw_c_matches_full(n, dtype, case):
    plan = plan_for(n)
    if case == "all":
        sigma = sorted(
            {
                bank.tile_sites * axis.spacing
                for axis in (plan.input, plan.output)
                for bank in axis.banks
                if not bank.full_chart
            }
            | {40.0}
        )
    else:
        sigma = [] if case == "empty" else [9.0]
    atoms = torch.tensor(
        [[0.0, s, 127.5, -0.5] for s in sigma], device="cuda", dtype=torch.float64
    ).reshape(-1, 4)
    masks = masks_for(plan, atoms)
    values = masks.cpu().tolist()  # Explicit untimed correctness diagnostic.
    like = torch.empty(5, n, device="cuda", dtype=dtype)
    for side, axis in enumerate((plan.input, plan.output)):
        if case == "all":
            assert values[2 * side] == (1 << len(axis.banks)) - 1
        c = torch.randn(
            5, axis.node_count, axis.basis_count, device="cuda", dtype=torch.float64
        )
        for index, bank in enumerate(axis.banks):
            if not values[2 * side] & (1 << index):
                c[:, bank.node_offset : bank.node_offset + bank.tile_count] = 0
        if case == "zero_c":
            c.zero_()
        full = executor.expand_c(c, axis, like)
        masked = executor.expand_c(c, axis, like, bank_masks=masks, mask_side=side)
        torch.testing.assert_close(masked, full, rtol=0, atol=0)
        if case in ("empty", "zero_c"):
            assert torch.count_nonzero(masked).item() == 0


@pytest.mark.parametrize("n", [128, 1024])
@pytest.mark.parametrize("active_z", [False, True])
def test_active_expand_zero_amplitude_independent_pullback(n, active_z):
    plan = plan_for(n)
    atoms = torch.tensor(
        [[0.0, 9.0, 127.5, -0.5]],
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    x = torch.randn(5, n, device="cuda", dtype=torch.float64, requires_grad=True)
    dy = torch.randn_like(x)
    kwargs = {
        "z_chunk_sites": 32 if active_z else 0,
        "window_moments": True,
        "c_splits": 2,
        "fuse_g_grad": True,
    }
    baseline = executor.hierarchical_linear(plan, x, atoms, **kwargs)
    y = executor.hierarchical_linear(
        plan, x, atoms, **kwargs, active_bank_z=active_z, active_bank_expand=True
    )
    dx, dp = torch.autograd.grad(y, (x, atoms), dy, retain_graph=True)
    repeated = torch.autograd.grad(y, (x, atoms), -2 * dy)
    assert torch.count_nonzero(y).item() == 0
    assert torch.count_nonzero(dx).item() == 0
    assert dp[0, 0].abs().item() > 0  # Routed zero amplitude still has a pullback.
    helper = runpy.run_path(
        str(Path(__file__).with_name("test_cell_mass_hierarchical.py"))
    )
    matrix = helper["differentiable_profile_matrix"]
    xr = x.detach().cpu().requires_grad_()
    pr = atoms.detach().cpu().requires_grad_()
    yr = ((xr @ matrix(pr, plan.input, 3).T) * pr[:, 0]) @ matrix(pr, plan.output, 2)
    reference = torch.autograd.grad(yr, (xr, pr), dy.cpu())
    torch.testing.assert_close(y, baseline, rtol=0, atol=0)
    for actual, expected, again in zip((dx, dp), reference, repeated):
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-9, atol=2e-10)
        torch.testing.assert_close(again, -2 * actual, rtol=3e-9, atol=2e-10)


@pytest.mark.parametrize("active_z", [False, True])
def test_active_expand_polar_all_gradients_and_retained(active_z):
    helper = runpy.run_path(str(Path(__file__).with_name("test_cell_mass_polar.py")))
    model = helper["layer"](
        128, count=37, dtype=torch.float32, device="cuda", site_scaled=True
    )
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.algorithm import (
        hierarchy_from_operator,
    )
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.plans import (
        SITE_SCALED,
    )

    x = torch.randn(5, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = executor.hierarchical_linear(
        hierarchy_from_operator(model.declaration()),
        x,
        model.atoms.p,
        active_bank_z=active_z,
        active_bank_expand=True,
        **SITE_SCALED.recipe.arguments(),
    )
    dx, dp = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
    repeated = torch.autograd.grad(y, (x, model.atoms.p), -2 * dy)
    xr = x.detach().cpu().double().requires_grad_()
    pr = model.atoms.p.detach().cpu().double().requires_grad_()
    yr = helper["oracle"](model, xr, pr)
    reference = torch.autograd.grad(yr, (xr, pr), dy.cpu().double())
    torch.testing.assert_close(y.cpu().double(), yr, rtol=3e-6, atol=2e-7)
    for actual, expected, again in zip((dx, dp), reference, repeated):
        torch.testing.assert_close(
            actual.cpu().double(), expected, rtol=3e-6, atol=2e-7
        )
        torch.testing.assert_close(again, -2 * actual, rtol=3e-6, atol=2e-7)
