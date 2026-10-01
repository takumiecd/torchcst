from __future__ import annotations

from torchcst._backends.torch.geometry import execution as _geometry
from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.profiles import execution as _profile

"""Mapped block sites, oracle gradients, and forward GPU correctness."""
import pytest
import torch
import torch.nn.functional as F

from experiments.cuda.linear.block_strip_linear import BlockStripLinear


@pytest.mark.parametrize(
    "shape,tile",
    [((8, 8), (4, 4)), ((9, 11), (4, 4)), ((3, 5), (4, 8)), ((5, 9), (3, 5))],
)
def test_mapped_reference_values_gradients_and_indices(shape, tile):
    torch.manual_seed(7)
    layer = BlockStripLinear(shape, tile, 12, dtype=torch.float64)
    x = torch.randn(3, shape[1], dtype=torch.float64, requires_grad=True)
    expected = F.linear(x, layer.dense_weight())
    got = layer(x)
    torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-12)
    dy = torch.randn_like(got)
    for a, b in zip(
        torch.autograd.grad(got, (x, layer.strip.atoms.p), dy),
        torch.autograd.grad(expected, (x, layer.strip.atoms.p), dy),
    ):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
    virtual = torch.arange(layer.strip.chart.features).reshape(layer.strip.chart.shape)
    indices = torch.arange(shape[0] * shape[1])
    torch.testing.assert_close(
        layer.unroll_weight(virtual).flatten(), layer.logical_to_virtual(indices)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "shape,tile",
    [
        ((3, 5), (4, 8)),
        ((33, 35), (16, 16)),
        ((65, 129), (64, 128)),
        ((128, 128), (128, 128)),
    ],
)
@pytest.mark.parametrize(
    "backend",
    [
        "triton_direct",
        "triton_fused",
        "triton_reuse",
        "triton_shared",
        "triton_atom_dot",
    ],
)
def test_mapped_gpu(shape, tile, backend):
    torch.manual_seed(7)
    layer = BlockStripLinear(shape, tile, 16, device="cuda")
    with torch.no_grad():
        x = torch.randn(shape[1], 5, device="cuda").T
        expected = F.linear(x, layer.dense_weight())
        actual = layer(x, backend=backend)
        torch.testing.assert_close(actual, expected, atol=3e-05, rtol=3e-05)
        assert layer(x[:0], backend=backend).shape == (0, shape[0])
    with pytest.raises(NotImplementedError, match="forward only"):
        layer(x, backend=backend)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "backend",
    [
        "triton_direct",
        "triton_fused",
        "triton_reuse",
        "triton_shared",
        "triton_atom_dot",
    ],
)
def test_boundary_seam_and_graph_updates(backend, execution=None):
    execution = execution or {}
    from torchcst._backends.cuda.algorithms.strip_torus.fused.host import prepare

    layer = BlockStripLinear(
        (32, 32), (16, 16), 8, device="cuda", tile_pitch=2.6, sigma=0.8, sigma_min=0.8
    )
    with torch.no_grad():
        p = layer.strip.atoms.p
        chart = layer.strip.chart
        coord = p.new_zeros((8, 3))
        coord[:, 0] = chart.axes[0].start[0] + torch.arange(8, device="cuda") // 2 * 2.6
        coord[:, 0] += torch.where(torch.arange(8, device="cuda") % 2 == 0, 0.7, 2.05)
        p[:, 2:] = _geometry.encode_centers(
            chart.geometry, _geometry.lift_chart_coordinates(chart.geometry, coord)
        )
        p[:, 0] = 0.8
        offsets = prepare(layer.strip, p, support_layout=True)[3]
        counts = torch.diff(offsets)
        assert counts[1:-1:2].sum() > 0 and counts[-2] > 0
        x = torch.randn(5, 32, device="cuda")
        torch.testing.assert_close(
            layer(x, backend=backend, **execution),
            F.linear(x, layer.dense_weight()),
            atol=3e-05,
            rtol=3e-05,
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                layer(x, backend=backend, **execution)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = layer(x, backend=backend, **execution)
        p[:, 0] *= 0.7
        p[:, 2] += 0.2
        graph.replay()
        torch.testing.assert_close(
            y, F.linear(x, layer.dense_weight()), atol=3e-05, rtol=3e-05
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_diagnostic_modes_and_counts():
    from experiments.cuda.linear.direct_diagnostics import (
        count_rows,
        diagnostic_forward,
    )
    from torchcst._backends.cuda.algorithms.strip_torus.fused.host import prepare

    layer = BlockStripLinear((32, 32), (16, 32), 16, device="cuda")
    with torch.no_grad():
        p, circle, section, offsets = prepare(
            layer.strip, layer.strip.atoms.p, support_layout=True
        )
        bounds = torch.stack((section.amin(0), section.amax(0)))
        x = torch.randn(5, 32, device="cuda")
        y = torch.empty((5, 32), device="cuda")
        weight = layer.dense_weight()
        opts = {"N": 32, "K": 32, "S": 16, "G": 2, "D": 4, "PROFILE": 1, "BK": 128}
        for mode in (0, 1, 2, 3):
            diagnostic_forward[1, 32](
                x,
                p,
                circle,
                section,
                offsets,
                bounds,
                y,
                M=5,
                BM=16,
                MODE=mode,
                **opts,
                num_warps=4,
                enable_fp_fusion=False,
            )
            expected = F.linear(torch.ones_like(x) if mode == 1 else x, weight)
            torch.testing.assert_close(y, expected, atol=3e-05, rtol=3e-05)
        counts = torch.empty((32, 4), device="cuda", dtype=torch.int32)
        count_rows[32,](
            p,
            circle,
            section,
            offsets,
            bounds,
            counts,
            **opts,
            num_warps=4,
            enable_fp_fusion=False,
        )
        center, amp, prec = _kernel.coordinate(
            layer.strip.kernel,
            "tile_parameters",
            layer.strip.chart,
            layer.strip.atoms.p,
        )
        values = _profile.evaluate_with_precision_slice(
            layer.strip.kernel.profiles[0],
            layer.strip.chart,
            center,
            prec,
            slice(0, 1024),
        )
        assert int(counts[:, 2].sum()) == int(((values != 0) & (amp != 0)).sum())
        assert bool((counts[:, 1] <= counts[:, 0]).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_tile", [16, 64, 128])
def test_reuse_tail_and_multiple_column_fragments(batch_tile):
    torch.manual_seed(19)
    layer = BlockStripLinear((33, 197), (16, 192), 19, device="cuda")
    with torch.no_grad():
        layer.strip.atoms.p[::3, 0] = 0
        x = torch.randn(129, 197, device="cuda")
        expected = F.linear(x, layer.dense_weight())
        torch.testing.assert_close(
            layer(x, backend="triton_reuse", batch_tile=batch_tile),
            expected,
            atol=3e-05,
            rtol=3e-05,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "backend,bm,bn,warps",
    [
        ("triton_shared", 16, 2, 4),
        ("triton_shared", 32, 4, 4),
        ("triton_shared", 64, 4, 8),
        ("triton_shared", 16, 8, 4),
        ("triton_atom_dot", 16, 16, 4),
        ("triton_atom_dot", 32, 16, 4),
        ("triton_atom_dot", 32, 32, 4),
        ("triton_atom_dot", 64, 16, 4),
        ("triton_atom_dot", 64, 16, 8),
    ],
)
def test_shared_partial_station_and_batch(backend, bm, bn, warps):
    torch.manual_seed(41)
    layer = BlockStripLinear((17, 197), (5, 192), 19, device="cuda")
    with torch.no_grad():
        layer.strip.atoms.p[::3, 0] = 0
        x = torch.randn(65, 197, device="cuda")
        torch.testing.assert_close(
            layer(x, backend=backend, batch_tile=bm, output_tile=bn, num_warps=warps),
            F.linear(x, layer.dense_weight()),
            atol=3e-05,
            rtol=3e-05,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "width,mode", [(16, "bounds"), (16, "exact"), (32, "bounds"), (32, "exact")]
)
def test_support_culling_partial_and_seam(width, mode):
    torch.manual_seed(29)
    layer = BlockStripLinear((17, 197), (5, 192), 19, device="cuda")
    with torch.no_grad():
        layer.strip.atoms.p[::3, 0] = 0
        x = torch.randn(65, 197, device="cuda")
        torch.testing.assert_close(
            layer(x, backend="triton_atom_dot", column_tile=width, support_cull=mode),
            F.linear(x, layer.dense_weight()),
            atol=3e-05,
            rtol=3e-05,
        )
    test_boundary_seam_and_graph_updates(
        "triton_atom_dot", {"column_tile": width, "support_cull": mode}
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_support_counts_match_canonical_atom_sites():
    from experiments.cuda.linear.block_support_diagnostics import diagnose_support
    from torchcst._backends.cuda.algorithms.strip_torus.fused.host import prepare

    layer = BlockStripLinear((17, 65), (16, 64), 12, device="cuda")
    with torch.no_grad():
        p = layer.strip.atoms.p
        p[::4, 0] = 0
        result = diagnose_support(
            layer, prepare(layer.strip, p, support_layout=True), 65
        )
        center, amp, prec = _kernel.coordinate(
            layer.strip.kernel, "tile_parameters", layer.strip.chart, p
        )
        indices = layer.logical_to_virtual(torch.arange(17 * 65, device="cuda"))
        values = _profile.evaluate_with_precision_slice(
            layer.strip.kernel.profiles[0], layer.strip.chart, center, prec, indices
        )
        assert result["nonzero_sites_per_m_group"] == int(
            ((values != 0) & (amp != 0)).sum()
        )
