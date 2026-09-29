"""Mapped weight windows propagate input and atom gradients without full W."""

import pytest
import torch

pytest.importorskip("triton")

import experiments.cuda.linear.block_streamed_backward as streamed_backward
from experiments.cuda.linear.block_streamed_backward import (
    build_listed_forward_candidates_bounded,
    mapped_streamed_trainable,
    trainable_boxed_prepare,
)
from experiments.cuda.linear.block_strip_linear import BlockStripLinear
from experiments.cuda.linear.support_box_routing import (
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
    if atom_kernel == "staged_listed":
        for materialize_mode in ("listed_csr", "listed_bounded"):
            result = mapped_streamed_trainable(
                layer,
                x,
                boxes=boxes,
                witness_cols=hints,
                window_rows=64,
                atom_kernel="staged_listed",
                cache_windows=1,
                materialize_mode=materialize_mode,
            )
            result_dx, result_dp = torch.autograd.grad(result, (x, p), upstream)
            torch.testing.assert_close(result, expected, atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(result_dx, expected_dx, atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(result_dp, expected_dp, atol=3e-4, rtol=3e-4)


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
@pytest.mark.parametrize("listed_unroll", (1, 4))
@pytest.mark.parametrize("builder", ((8, 4), (32, 1)))
def test_bounded_lists_fall_back_when_every_tile_overflows(
    monkeypatch, listed_unroll, builder
):
    torch.manual_seed(729)
    layer = BlockStripLinear((128, 128), (64, 64), 819, device="cuda")
    site = layer.strip
    plan = execution_plan(site)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(site)
    original_builder = streamed_backward.build_listed_forward_candidates_bounded

    def one_entry_builder(*args, **kwargs):
        result = original_builder(*args, max_candidates=1, **kwargs)
        assert torch.all(result[1] == -1)
        return result

    monkeypatch.setattr(
        streamed_backward, "build_listed_forward_candidates_bounded", one_entry_builder
    )
    x = torch.randn(5, 128, device="cuda", requires_grad=True)
    upstream = torch.randn(5, 128, device="cuda")
    actual = mapped_streamed_trainable(
        layer,
        x,
        boxes=boxes,
        witness_cols=hints,
        window_rows=64,
        cache_windows=1,
        atom_kernel="staged_listed",
        materialize_mode="listed_bounded",
        listed_unroll=listed_unroll,
        listed_builder_ba=builder[0],
        listed_builder_warps=builder[1],
    )
    actual_dx, actual_dp = torch.autograd.grad(actual, (x, site.atoms.p), upstream)
    expected = layer.reference(x)
    expected_dx, expected_dp = torch.autograd.grad(
        expected, (x, site.atoms.p), upstream
    )
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(actual_dx, expected_dx, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(actual_dp, expected_dp, atol=3e-4, rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bounded_uint8_lists_guard_large_bucket_ranks():
    torch.manual_seed(815)
    layer = BlockStripLinear((128, 128), (64, 64), 1200, device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    lists, counts, capacity = build_listed_forward_candidates_bounded(
        layer, packed, circle, section, offsets
    )
    assert lists.dtype == torch.uint8
    assert capacity == 192
    groups = layer.strip.chart.tile_count
    guarded = 0
    for station in range(groups):
        previous = 2 * ((station + groups - 1) % groups) + 1
        rank_count = (
            offsets[previous + 1]
            - offsets[previous]
            + offsets[2 * station + 2]
            - offsets[2 * station]
        )
        if rank_count.item() > 256:
            guarded += 1
            assert torch.all(counts[station] == -1)
    assert guarded > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("listed_unroll", (1, 4))
@pytest.mark.parametrize("builder", ((8, 4), (32, 1)))
def test_bounded_graph_replay_after_bucket_overflow(listed_unroll, builder):
    torch.manual_seed(829)
    layer = BlockStripLinear((128, 128), (64, 64), 819, device="cuda")
    site = layer.strip
    plan = execution_plan(site)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(site)
    x = torch.randn(2, 128, device="cuda")

    def forward():
        with torch.no_grad():
            return mapped_streamed_trainable(
                layer,
                x,
                boxes=boxes,
                witness_cols=hints,
                window_rows=64,
                atom_kernel="staged_listed",
                materialize_mode="listed_bounded",
                listed_unroll=listed_unroll,
                listed_builder_ba=builder[0],
                listed_builder_warps=builder[1],
            )

    for _ in range(3):
        forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    with torch.no_grad():
        coordinate = site.atoms.p.new_zeros((site.atoms.p.shape[0], 3))
        coordinate[:, 0] = site.chart.axes[0].start[0] + 0.7
        site.atoms.p[:, 2:] = site.chart.geometry.encode_centers(
            site.chart.geometry.lift_chart_coordinates(coordinate)
        )
    prepared = trainable_boxed_prepare(
        site, site.atoms.p, boxes=boxes, witness_cols=hints
    )
    _, counts, _ = build_listed_forward_candidates_bounded(
        layer, *prepared, ba=builder[0], warps=builder[1]
    )
    assert torch.any(counts == -1)
    eager = forward()
    with torch.no_grad():
        csr = mapped_streamed_trainable(
            layer,
            x,
            boxes=boxes,
            witness_cols=hints,
            window_rows=64,
            atom_kernel="staged_listed",
            materialize_mode="listed_csr",
        )
    with torch.no_grad():
        expected = layer.reference(x)
    torch.testing.assert_close(eager, csr, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(eager, expected, atol=3e-5, rtol=3e-5)
    graph.replay()
    torch.testing.assert_close(captured, eager, atol=3e-5, rtol=3e-5)

    x_grad = x.detach().clone().requires_grad_()
    upstream = torch.randn_like(x_grad)

    def trainable(mode):
        return mapped_streamed_trainable(
            layer,
            x_grad,
            boxes=boxes,
            witness_cols=hints,
            window_rows=64,
            cache_windows=1,
            atom_kernel="staged_listed",
            materialize_mode=mode,
            listed_unroll=listed_unroll if mode == "listed_bounded" else 1,
            listed_builder_ba=builder[0],
            listed_builder_warps=builder[1],
        )

    bounded_trainable = trainable("listed_bounded")
    csr_trainable = trainable("listed_csr")
    bounded_dx, bounded_dp = torch.autograd.grad(
        bounded_trainable, (x_grad, site.atoms.p), upstream
    )
    csr_dx, csr_dp = torch.autograd.grad(
        csr_trainable, (x_grad, site.atoms.p), upstream
    )
    torch.testing.assert_close(bounded_trainable, csr_trainable, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(bounded_dx, csr_dx, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(bounded_dp, csr_dp, atol=3e-4, rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size", (128, 192))
@pytest.mark.parametrize("gemm_mode", ("tf32x3", "tf32x3_dx"))
@pytest.mark.parametrize("forward_gemm_mode", ("ieee", "tf32x3"))
@pytest.mark.parametrize(
    "materialize_mode", ("default", "listed", "listed_csr", "listed_bounded")
)
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
