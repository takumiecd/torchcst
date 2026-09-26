"""Accumulation scheduling changes preserve local geometry and updated atoms."""

import pytest
import torch
import torch.nn.functional as F

from prototypes.block_fused_config import FusedConfig
from prototypes.block_strip_linear import BlockStripLinear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "config",
    [
        FusedConfig(128, 16, 16, 8, 4, False),
        FusedConfig(128, 16, 16, 32, 4, False),
        FusedConfig(128, 16, 16, 8, 4, True),
        FusedConfig(128, 32, 16, 16, 4, True),
        FusedConfig(128, 16, 64, 8, 8, True),
    ],
)
def test_fused_schedule_tail_and_seam(config):
    from test_block_strip_linear import test_boundary_seam_and_graph_updates

    torch.manual_seed(397)
    layer = BlockStripLinear((17, 65), (5, 32), 259, device="cuda")
    with torch.no_grad():
        layer.strip.atoms.p[::7, 0] = 0
        x = torch.randn(65, 129, device="cuda").T
        actual = layer(x, backend="triton_fused", fused_config=config)
        expected = F.linear(x, layer.dense_weight())
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        assert layer(x[:0], backend="triton_fused", fused_config=config).shape == (
            0,
            17,
        )
    test_boundary_seam_and_graph_updates("triton_fused", {"fused_config": config})
