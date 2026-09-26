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
        FusedConfig(128, 32, 16, 1, 4, True),
        FusedConfig(128, 16, 64, 1, 4, True),
        FusedConfig(128, 16, 32, 1, 4, True, True),
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


def test_merge_buckets_requires_exact_bool_and_ba1_late_reduce():
    with pytest.raises(ValueError, match="merge_buckets must be bool"):
        FusedConfig(merge_buckets=1)
    with pytest.raises(ValueError, match="atoms=1 and late_reduce=True"):
        FusedConfig(128, 16, 16, 8, 4, True, False, True)
    with pytest.raises(ValueError, match="atoms=1 and late_reduce=True"):
        FusedConfig(atoms=1, merge_buckets=True)
    opted = FusedConfig(128, 16, 16, 1, 4, True, False, True)
    assert opted.merge_buckets and not opted.fp_fusion


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "shape, tile",
    [
        ((15, 18), (15, 18)),
        ((17, 17), (16, 17)),
    ],
)
def test_merge_buckets_matches_late_reduce_bitwise(shape, tile):
    from torchcst.nn._backends._preparation import prepare

    torch.manual_seed(431)
    layer = BlockStripLinear(
        shape, tile, 8, device="cuda", tile_pitch=2.6, sigma=0.8, sigma_min=0.8
    )
    old = FusedConfig(16, 16, 16, 1, 4, True)
    merged = FusedConfig(16, 16, 16, 1, 4, True, False, True)
    with torch.no_grad():
        p = layer.strip.atoms.p
        chart = layer.strip.chart
        assert chart.tile_count == (1 if shape == (15, 18) else 2)
        coord = p.new_zeros((8, 3))
        coord[:, 0] = (
            chart.axes[0].start[0] + (torch.arange(8, device="cuda") // 2) * 2.6
        )
        coord[:, 0] += torch.where(torch.arange(8, device="cuda") % 2 == 0, 0.7, 2.05)
        p[:, 2:] = chart.geometry.encode_centers(
            chart.geometry.lift_chart_coordinates(coord)
        )
        p[:, 0] = 0.8
        if chart.tile_count > 1:
            counts = torch.diff(prepare(layer.strip, p, support_layout=True)[3])
            assert counts[1:-1:2].sum() > 0 and counts[-2] > 0
        x = torch.randn(5, shape[1], device="cuda")
        weight = layer.dense_weight()
        actual = layer(x, backend="triton_fused", fused_config=merged)
        reference = layer(x, backend="triton_fused", fused_config=old)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        torch.testing.assert_close(actual, F.linear(x, weight), atol=3e-5, rtol=3e-5)
        assert layer(x[:0], backend="triton_fused", fused_config=merged).shape == (
            0,
            shape[0],
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                layer(x, backend="triton_fused", fused_config=merged)
                layer(x, backend="triton_fused", fused_config=old)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(x, backend="triton_fused", fused_config=merged)
            captured_old = layer(x, backend="triton_fused", fused_config=old)
        p[:, 0] *= 0.7
        p[:, 2] += 0.2
        graph.replay()
        torch.testing.assert_close(captured, captured_old, atol=0, rtol=0)
        torch.testing.assert_close(
            captured, F.linear(x, layer.dense_weight()), atol=3e-5, rtol=3e-5
        )


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
