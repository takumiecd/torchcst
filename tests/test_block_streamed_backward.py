"""Mapped weight windows propagate input and atom gradients without full W."""

import pytest
import torch

pytest.importorskip("triton")

from prototypes.block_streamed_backward import (
    mapped_streamed_trainable,
    trainable_boxed_prepare,
)
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import (
    balanced_home_columns,
    station_site_boxes,
)
from torchcst.nn._backends._preparation import execution_plan


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size", (128, 192))
@pytest.mark.parametrize(
    "atom_kernel",
    (
        "baseline",
        "factored",
        "staged",
        "staged_partial",
        "staged_listed",
        "atom_major",
        "interval",
        "fused",
    ),
)
def test_mapped_streamed_backward_matches_torch_reference(size, atom_kernel):
    torch.manual_seed(916)
    layer = BlockStripLinear(
        (size, size), (64, 64), round(size * size * 0.05), device="cuda"
    )
    site = layer.strip
    plan = execution_plan(site)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(site)
    x = torch.randn(3, size, device="cuda", requires_grad=True)
    upstream = torch.randn(3, size, device="cuda")
    y = mapped_streamed_trainable(
        layer,
        x,
        boxes=boxes,
        witness_cols=hints,
        atom_kernel=atom_kernel,
    )
    dx, dp = torch.autograd.grad(y, (x, site.atoms.p), upstream)
    expected = layer.reference(x)
    expected_dx, expected_dp = torch.autograd.grad(
        expected, (x, site.atoms.p), upstream
    )
    torch.testing.assert_close(y, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(dx, expected_dx, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(dp, expected_dp, atol=3e-4, rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "atom_kernel",
    (
        "baseline",
        "factored",
        "staged",
        "staged_partial",
        "staged_listed",
        "atom_major",
        "interval",
        "fused",
    ),
)
def test_mapped_backward_after_atom_movement_and_boundary_support(atom_kernel):
    torch.manual_seed(563)
    layer = BlockStripLinear(
        (128, 128),
        (64, 64),
        128,
        device="cuda",
        tile_pitch=2.6,
        sigma=0.8,
        sigma_min=0.8,
    )
    site = layer.strip
    chart = site.chart
    plan = execution_plan(site)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(site)
    with torch.no_grad():
        p = site.atoms.p
        atom = torch.arange(p.shape[0], device="cuda")
        coordinate = p.new_zeros((p.shape[0], 3))
        coordinate[:, 0] = chart.axes[0].start[0] + (atom % 4) * 2.6
        coordinate[:, 0] += torch.where(atom % 2 == 0, 0.7, 2.05)
        p[:, 2:] = chart.geometry.encode_centers(
            chart.geometry.lift_chart_coordinates(coordinate)
        )
        p[:, 0] = 0.8
        p[::13, 4] += 4.0
    prepared = trainable_boxed_prepare(site, p, boxes=boxes, witness_cols=hints)
    counts = torch.diff(prepared[3])
    assert counts[1:-1:2].sum() > 0
    assert counts[-1] > 0
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    upstream = torch.randn(3, 128, device="cuda")
    y = mapped_streamed_trainable(
        layer,
        x,
        boxes=boxes,
        witness_cols=hints,
        window_rows=64,
        atom_kernel=atom_kernel,
        cache_windows=1 if atom_kernel == "staged_listed" else 0,
        materialize_mode="listed" if atom_kernel == "staged_listed" else "default",
    )
    dx, dp = torch.autograd.grad(y, (x, p), upstream)
    expected = layer.reference(x)
    expected_dx, expected_dp = torch.autograd.grad(expected, (x, p), upstream)
    torch.testing.assert_close(y, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(dx, expected_dx, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(dp, expected_dp, atol=3e-4, rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cached_backward_when_only_atom_gradient_is_required():
    torch.manual_seed(817)
    layer = BlockStripLinear((128, 128), (64, 64), 256, device="cuda")
    site = layer.strip
    plan = execution_plan(site)
    x = torch.randn(3, 128, device="cuda")
    upstream = torch.randn(3, 128, device="cuda")
    y = mapped_streamed_trainable(
        layer,
        x,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(site),
        window_rows=64,
        atom_kernel="staged_listed",
        cache_windows=1,
        materialize_mode="listed",
    )
    (dp,) = torch.autograd.grad(y, (site.atoms.p,), upstream)
    expected = layer.reference(x)
    (expected_dp,) = torch.autograd.grad(expected, (site.atoms.p,), upstream)
    torch.testing.assert_close(y, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(dp, expected_dp, atol=3e-4, rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size", (128, 192))
@pytest.mark.parametrize("gemm_mode", ("tf32x3", "tf32x3_dx"))
@pytest.mark.parametrize("forward_gemm_mode", ("ieee", "tf32x3"))
@pytest.mark.parametrize("materialize_mode", ("default", "listed"))
def test_tf32x3_window_gemm_matches_reference(
    size, gemm_mode, forward_gemm_mode, materialize_mode
):
    torch.manual_seed(825)
    layer = BlockStripLinear(
        (size, size), (64, 64), round(size * size * 0.05), device="cuda"
    )
    site = layer.strip
    plan = execution_plan(site)
    x = torch.randn(32, size, device="cuda", requires_grad=True)
    upstream = torch.randn(32, size, device="cuda")
    actual = mapped_streamed_trainable(
        layer,
        x,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(site),
        window_rows=64,
        cache_windows=1,
        atom_kernel="staged_listed",
        gemm_mode=gemm_mode,
        forward_gemm_mode=forward_gemm_mode,
        materialize_mode=materialize_mode,
    )
    dx, dp = torch.autograd.grad(actual, (x, site.atoms.p), upstream)
    expected = layer.reference(x)
    expected_dx, expected_dp = torch.autograd.grad(
        expected, (x, site.atoms.p), upstream
    )
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(dx, expected_dx, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(dp, expected_dp, atol=3e-4, rtol=3e-4)
