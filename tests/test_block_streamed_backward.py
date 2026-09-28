"""Mapped weight windows propagate input and atom gradients without full W."""

import pytest
import torch

pytest.importorskip("triton")

from prototypes.block_streamed_backward import mapped_streamed_trainable
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import (
    balanced_home_columns,
    station_site_boxes,
)
from torchcst.nn._backends._preparation import execution_plan


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size", (128, 192))
def test_mapped_streamed_backward_matches_torch_reference(size):
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
        layer, x, boxes=boxes, witness_cols=hints, window_rows=64
    )
    dx, dp = torch.autograd.grad(y, (x, site.atoms.p), upstream)
    expected = layer.reference(x)
    expected_dx, expected_dp = torch.autograd.grad(
        expected, (x, site.atoms.p), upstream
    )
    torch.testing.assert_close(y, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(dx, expected_dx, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(dp, expected_dp, atol=3e-4, rtol=3e-4)
