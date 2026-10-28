"""Small batched active-bank gates; use one GPU invocation, no timings."""

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


@pytest.mark.parametrize("case", ["low", "parent", "full", "mixed"])
@pytest.mark.parametrize("windowed", [False, True])
def test_masked_moments_and_physical_pullback(case, windowed):
    """Include absent ancestors, independent full chart, zero amp and wraps."""
    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4),
        output=AxisPlan(site_count=128, min_tile_sites=8),
    )
    sigma = {
        "low": [0.2],
        "parent": [9.0],
        "full": [40.0],
        "mixed": [0.2, 4.0, 4.000001, 8.0, 8.000001, 40.0],
    }[case]
    source = torch.tensor(
        [
            [(-1.0 if i % 2 else 0.0), s, 128.0 + i / 3, -0.5 + i]
            for i, s in enumerate(sigma)
        ],
        dtype=torch.float64,
    )
    p = source.cuda().requires_grad_()
    x = torch.randn(5, 128, device="cuda", dtype=torch.float64, requires_grad=True)
    dy = torch.randn_like(x)
    layout = executor.prepare_layout(plan, p, reverse=True)
    masks = executor._extension().prepare_bank_masks(
        layout.groups,
        layout.offsets,
        plan.input.native_config(),
        plan.output.native_config(),
    )
    mask_values = masks.cpu().tolist()  # Explicit correctness gate, never timed.
    assert mask_values[0] and mask_values[2]  # Zero amplitude never removes routing.
    if case == "full":
        for side, axis in enumerate((plan.input, plan.output)):
            full_bit = 1 << (len(axis.banks) - 1)
            assert mask_values[2 * side : 2 * side + 2] == [full_bit, full_bit]
    for side, axis in enumerate((plan.input, plan.output)):
        full = executor.build_hierarchy(x.detach(), axis, chunk_sites=32)
        masked = executor.build_hierarchy(
            x.detach(), axis, chunk_sites=32, bank_masks=masks, mask_side=side
        )
        full_r = executor.build_window_moments(full, axis)
        masked_r = executor.build_window_moments(
            masked, axis, bank_masks=masks, mask_side=side
        )
        for index, bank in enumerate(axis.banks):
            nodes = slice(bank.node_offset, bank.node_offset + bank.tile_count)
            for actual, reference, mask in (
                (masked, full, mask_values[2 * side + 1]),
                (masked_r, full_r, mask_values[2 * side]),
            ):
                expected = (
                    reference[:, nodes]
                    if mask & (1 << index)
                    else torch.zeros_like(reference[:, nodes])
                )
                torch.testing.assert_close(actual[:, nodes], expected, rtol=0, atol=0)
    options = dict(
        layout=layout,
        z_chunk_sites=32,
        window_moments=windowed,
        active_bank_z=True,
        c_splits=2,
        fuse_g_grad=True,
    )
    y = executor.hierarchical_linear(plan, x, p, **options)
    dx, dp = torch.autograd.grad(y, (x, p), dy, retain_graph=True)
    again = torch.autograd.grad(y, (x, p), -2 * dy)
    helper = runpy.run_path(
        str(Path(__file__).with_name("test_cell_mass_hierarchical.py"))
    )
    matrix = helper["differentiable_profile_matrix"]
    xr = x.detach().cpu().requires_grad_()
    pr = source.clone().requires_grad_()
    yr = ((xr @ matrix(pr, plan.input, 3).T) * pr[:, 0]) @ matrix(pr, plan.output, 2)
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.cpu())
    for actual, expected in ((y, yr), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-9, atol=2e-10)
    for actual, repeated in zip((dx, dp), again):
        torch.testing.assert_close(repeated, -2 * actual, rtol=3e-9, atol=2e-10)


@pytest.mark.parametrize(
    "sigma,bank", [(False, False), (True, False), (False, True), (True, True)]
)
def test_masked_site_scaled_polar_oracle_and_retained_backward(sigma, bank):
    """Exercise the training-style fused ragged/compensated route as well."""
    helper = runpy.run_path(str(Path(__file__).with_name("test_cell_mass_polar.py")))
    model = helper["layer"](
        128, count=7, dtype=torch.float32, device="cuda", site_scaled=True
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
        active_bank_z=True,
        hc_reuse_sigma_decode=sigma,
        hc_known_input_bank=bank,
        hc_batch_tile=8,
        **SITE_SCALED.recipe.arguments(),
    )
    dx, dp = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
    repeated = torch.autograd.grad(y, (x, model.atoms.p), -2 * dy)
    xr = x.detach().cpu().double().requires_grad_()
    pr = model.atoms.p.detach().cpu().double().requires_grad_()
    yr = helper["oracle"](model, xr, pr)
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.cpu().double())
    for actual, expected in ((y, yr), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(
            actual.cpu().double(), expected, rtol=3e-6, atol=2e-7
        )
    for actual, second in zip((dx, dp), repeated):
        torch.testing.assert_close(second, -2 * actual, rtol=3e-6, atol=2e-7)


def test_empty_layout_mask_returns_defined_zeros():
    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4),
        output=AxisPlan(site_count=128, min_tile_sites=8),
    )
    atoms = torch.empty(0, 4, device="cuda", dtype=torch.float64)
    layout = executor.prepare_layout(plan, atoms, reverse=True)
    masks = executor._extension().prepare_bank_masks(
        layout.groups,
        layout.offsets,
        plan.input.native_config(),
        plan.output.native_config(),
    )
    assert masks.cpu().tolist() == [0, 0, 0, 0]
    x = torch.randn(5, 128, device="cuda", dtype=torch.float64)
    for side, axis in enumerate((plan.input, plan.output)):
        z = executor.build_hierarchy(
            x, axis, chunk_sites=32, bank_masks=masks, mask_side=side
        )
        r = executor.build_window_moments(z, axis, bank_masks=masks, mask_side=side)
        assert torch.count_nonzero(z).item() == 0
        assert torch.count_nonzero(r).item() == 0


def test_saved_active_bank_mask_version_is_checked(monkeypatch):
    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4),
        output=AxisPlan(site_count=128, min_tile_sites=8),
    )
    atoms = torch.tensor(
        [[0.0, 9.0, 127.5, -0.5]],
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    x = torch.randn(5, 128, device="cuda", dtype=torch.float64, requires_grad=True)
    native = executor._extension()
    original = native.prepare_bank_masks
    saved_masks = []

    def capture(*args):
        masks = original(*args)
        saved_masks.append(masks)
        return masks

    monkeypatch.setattr(native, "prepare_bank_masks", capture)
    y = executor.hierarchical_linear(
        plan, x, atoms, z_chunk_sites=32, active_bank_z=True
    )
    assert len(saved_masks) == 1
    # Even a no-op external mutation must invalidate the saved forward metadata.
    saved_masks[0].add_(0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()
