"""Accumulation scheduling changes preserve local geometry and updated atoms."""

import math

import pytest
import torch
import torch.nn.functional as F

from prototypes.block_fused_config import FusedConfig, default_fused_config
from prototypes.block_strip_linear import BlockStripLinear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "config",
    [
        FusedConfig(128, 16, 16, 1, 4, True),
        FusedConfig(128, 16, 16, 2, 4, True),
        FusedConfig(128, 16, 16, 8, 4, False),
        FusedConfig(128, 16, 16, 32, 4, False),
        FusedConfig(128, 16, 16, 8, 4, True),
        FusedConfig(128, 32, 16, 16, 4, True),
        FusedConfig(128, 16, 64, 8, 8, True),
        FusedConfig(128, 16, 16, 8, 4, True, True),
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_selected_fused_default_preserves_tails_and_explicit_override():
    torch.manual_seed(401)
    layer = BlockStripLinear((65, 129), (64, 64), 37, device="cuda")
    with torch.no_grad():
        weight = layer.dense_weight()
        for batch in (127, 129):
            x = torch.randn(batch, 129, device="cuda")
            selected = (
                FusedConfig(128, late_reduce=True) if batch >= 128 else FusedConfig()
            )
            actual = layer(x, backend="triton_fused")
            explicit = layer(x, backend="triton_fused", fused_config=selected)
            torch.testing.assert_close(actual, explicit, atol=0, rtol=0)
            torch.testing.assert_close(
                actual, F.linear(x, weight), atol=3e-5, rtol=3e-5
            )
            override = layer(x, backend="triton_fused", batch_tile=64)
            original = layer(x, backend="triton_fused", fused_config=FusedConfig())
            torch.testing.assert_close(override, original, atol=0, rtol=0)


def test_fp_fusion_requires_exact_bool():
    with pytest.raises(ValueError, match="fp_fusion must be bool"):
        FusedConfig(fp_fusion=1)


def test_default_fp_fusion_requires_dense_a100_and_respects_override():
    bm128 = FusedConfig(128, late_reduce=True)
    assert default_fused_config(
        (64, 64), 128, atom_density=0.05, gpu_name="NVIDIA A100-SXM4-40GB"
    ) == FusedConfig(128, late_reduce=True, fp_fusion=True)
    assert (
        default_fused_config((64, 64), 128, atom_density=0.049, gpu_name="NVIDIA A100")
        == bm128
    )
    assert (
        default_fused_config(
            (64, 64), 128, atom_density=0.2, gpu_name="NVIDIA RTX PRO 6000"
        )
        == bm128
    )
    assert default_fused_config(
        (64, 64), 256, 64, atom_density=1.0, gpu_name="A100"
    ) == FusedConfig(batch_rows=64)


@pytest.mark.skipif(
    not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0),
    reason="default FP fusion is selected only on A100",
)
def test_default_fp_fusion_matches_explicit_dense_a100_tile():
    torch.manual_seed(419)
    shape = (64, 64)
    atoms = math.ceil(0.05 * shape[0] * shape[1])
    layer = BlockStripLinear(shape, (64, 64), atoms, device="cuda")
    with torch.no_grad():
        x = torch.randn(128, shape[1], device="cuda")
        selected = FusedConfig(128, late_reduce=True, fp_fusion=True)
        actual = layer(x, backend="triton_fused")
        explicit = layer(x, backend="triton_fused", fused_config=selected)
        torch.testing.assert_close(actual, explicit, atol=0, rtol=0)
        torch.testing.assert_close(
            actual, F.linear(x, layer.dense_weight()), atol=3e-5, rtol=3e-5
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                layer(x, backend="triton_fused")
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(x, backend="triton_fused")
        layer.strip.atoms.p[:, 0] *= 0.7
        layer.strip.atoms.p[:, 2] += 0.2
        graph.replay()
        torch.testing.assert_close(
            captured, F.linear(x, layer.dense_weight()), atol=3e-5, rtol=3e-5
        )
