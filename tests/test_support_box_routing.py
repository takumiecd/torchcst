"""Boxed support routing keeps exact I/B/idle layout after atom movement."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from experiments.cuda.linear.block_strip_linear import BlockStripLinear
from experiments.cuda.linear.support_box_routing import (
    atomic_bucket_sort,
    balanced_home_columns,
    boxed_prepare,
    decode_intrinsic_torus_compact,
    station_site_boxes,
)
from torchcst._backends.cuda.algorithms.strip_torus.fused.host import prepare
from torchcst._backends.torch.operators.strip_torus.preparation import execution_plan


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_boxed_support_matches_exact_layout_and_graph_updates(monkeypatch):
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
    x = torch.randn(7, 128, device="cuda")
    with torch.no_grad():
        p = site.atoms.p
        a = torch.arange(p.shape[0], device="cuda")
        coord = p.new_zeros((p.shape[0], 3))
        coord[:, 0] = chart.axes[0].start[0] + (a % 4) * 2.6
        coord[:, 0] += torch.where(a % 2 == 0, 0.7, 2.05)
        p[:, 2:] = chart.geometry.encode_centers(
            chart.geometry.lift_chart_coordinates(coord)
        )
        p[:, 0] = 0.8
        p[::13, 4] += 4.0

        def compare():
            ordinary = prepare(site, p, support_layout=True)
            boxed = boxed_prepare(site, p, boxes=boxes)
            witnessed = boxed_prepare(site, p, boxes=boxes, witness_cols=hints)
            fast_witnessed = boxed_prepare(
                site, p, boxes=boxes, witness_cols=hints, fast_witness=True
            )
            compact = boxed_prepare(
                site,
                p,
                boxes=boxes,
                witness_cols=hints,
                fast_witness=True,
                fast_decode=True,
            )
            with monkeypatch.context() as patch:
                patch.setattr(
                    torch,
                    "sort",
                    lambda keys, **_kwargs: atomic_bucket_sort(
                        keys, 2 * plan.routing.starts.numel() + 1
                    ),
                )
                atomic = boxed_prepare(
                    site,
                    p,
                    boxes=boxes,
                    witness_cols=hints,
                    fast_witness=True,
                    fast_decode=True,
                )
            decoded = decode_intrinsic_torus_compact(chart.geometry, p[:, 2:])
            torch.testing.assert_close(
                decoded, chart.geometry.decode_centers(p[:, 2:]), atol=3e-5, rtol=3e-5
            )
            assert torch.equal(ordinary[3], boxed[3])
            assert torch.equal(ordinary[0], boxed[0])
            assert torch.equal(ordinary[3], witnessed[3])
            assert torch.equal(ordinary[0], witnessed[0])
            assert torch.equal(ordinary[3], fast_witnessed[3])
            assert torch.equal(ordinary[0], fast_witnessed[0])
            assert torch.equal(ordinary[3], compact[3])
            assert torch.equal(ordinary[3], atomic[3])
            torch.testing.assert_close(ordinary[0], compact[0], atol=3e-5, rtol=3e-5)
            actual = layer(
                x, backend="triton_streamed", prepared=boxed, weight_chunk_rows=64
            )
            torch.testing.assert_close(
                actual, F.linear(x, layer.dense_weight()), atol=3e-5, rtol=3e-5
            )
            torch.testing.assert_close(
                layer(
                    x, backend="triton_streamed", prepared=atomic, weight_chunk_rows=64
                ),
                F.linear(x, layer.dense_weight()),
                atol=3e-5,
                rtol=3e-5,
            )
            return ordinary

        ordinary = compare()
        counts = torch.diff(ordinary[3])
        assert counts[1:-1:2].sum() > 0
        assert counts[-1] > 0
        p[:, 2] += 0.2
        compare()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                layer(
                    x,
                    backend="triton_streamed",
                    prepared=boxed_prepare(site, p, boxes=boxes),
                    weight_chunk_rows=64,
                )
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(
                x,
                backend="triton_streamed",
                prepared=boxed_prepare(
                    site, p, boxes=boxes, witness_cols=hints, fast_witness=True
                ),
                weight_chunk_rows=64,
            )
        p[:, 2] -= 0.35
        graph.replay()
        torch.testing.assert_close(
            captured, F.linear(x, layer.dense_weight()), atol=3e-5, rtol=3e-5
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_atomic_bucket_sort_is_a_complete_bucket_permutation():
    torch.manual_seed(812)
    keys = torch.randint(0, 257, (10001,), device="cuda", dtype=torch.int32)
    sorted_keys, order = atomic_bucket_sort(keys, 257)
    assert torch.equal(sorted_keys, keys[order.long()])
    assert bool((sorted_keys[1:] >= sorted_keys[:-1]).all())
    assert torch.equal(
        torch.sort(order).values,
        torch.arange(keys.numel(), device="cuda", dtype=torch.int32),
    )
