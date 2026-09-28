"""Forward-only mapped windows extended with a bounded-memory backward prototype."""

import math

import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable

from prototypes.block_materialize_kernel import materialize_logical
from prototypes.block_streamed_forward import streamed_forward
from prototypes.support_box_routing import atomic_bucket_sort
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan
from torchcst.nn._backends._triton_kernels import _sites, _values
from torchcst.nn._backends._triton_preparation import (
    Pack,
    route_and_layout,
    tile_parameters,
)


def trainable_boxed_prepare(site, p, *, boxes, witness_cols):
    """Keep the amplitude and Torus decode graph while routing without gradients."""
    plan = execution_plan(site)
    station_rows = site.chart.tile_shape[0]
    center, amplitude, precision = tile_parameters(site.kernel, p)
    decoded = site.chart.geometry.decode_centers(center)
    original_sort = torch.sort

    def bucket_sort(keys, **_kwargs):
        return atomic_bucket_sort(keys, 2 * plan.routing.starts.numel() + 1)

    with torch.no_grad():
        torch.sort = bucket_sort
        try:
            _, order, offsets = route_and_layout(
                plan.routing,
                decoded,
                support=(plan.circle, plan.section, precision, station_rows),
                retain_owners=False,
                support_boxes=boxes,
                support_witness_cols=witness_cols,
                support_fast_witness=True,
            )
        finally:
            torch.sort = original_sort
    packed = Pack.apply(amplitude, precision, decoded, order)
    return packed, plan.circle, plan.section, offsets


@tr.jit
def mapped_backward_atoms(
    X,
    DY,
    P,
    Circle,
    Section,
    Offsets,
    DP,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    T: tl.constexpr,
    CG: tl.constexpr,
    G: tl.constexpr,
    PROFILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BA: tl.constexpr,
):
    tile = tl.program_id(0)
    station = tile // tr.cdiv(S, BN)
    local = (tile % tr.cdiv(S, BN)) * BN
    col_start = tl.program_id(1) * BK
    local_cols = col_start + tl.arange(0, BK)
    logical_rows = (station // CG) * S + local + tl.arange(0, BN)
    logical_cols = (station % CG) * T + local_cols
    dw = tl.full((BN, BK), 0.0, tl.float32)
    for start in range(0, M, BM):
        m = start + tl.arange(0, BM)
        dy = tl.load(
            DY + m[:, None] * N + logical_rows[None, :],
            (m[:, None] < M) & (logical_rows[None, :] < N),
            0.0,
        )
        x = tl.load(
            X + m[:, None] * K + logical_cols[None, :],
            (m[:, None] < M) & (logical_cols[None, :] < K),
            0.0,
        )
        dw = tl.dot(tl.trans(dy), x, dw, input_precision="ieee")
    sites = tl.arange(0, BN * BK)
    rows = station * S + local + sites // BK
    cols = col_start + sites % BK
    valid = (rows < G * S) & (cols < T)
    sx, sy = _sites(Circle, Section, rows, cols, valid, 4)
    gradient = tl.reshape(dw, (BN * BK,))
    for neighbor in tl.static_range(1 if G == 1 else 3):
        if G == 1:
            bucket = 0
        else:
            bucket = tl.where(
                neighbor == 0,
                2 * ((station + G - 1) % G) + 1,
                2 * station + neighbor - 1,
            )
        begin, end = tl.load(Offsets + bucket), tl.load(Offsets + bucket + 1)
        for atom_start in range(begin, end, BA):
            atoms = atom_start + tl.arange(0, BA)
            active = atoms < end
            value, slope = _values(
                P, Section, atoms, active, sx, sy, cols, valid, 4, PROFILE
            )
            amplitude = tl.load(P + atoms * 6, active, 0.0)
            da = tl.sum(gradient[:, None] * value, axis=0)
            tl.atomic_add(DP + atoms * 6, da, active, sem="relaxed")
            scale = -2.0 * gradient[:, None] * slope * amplitude[None, :]
            for dim in tl.static_range(4):
                if dim == 0:
                    site = sx
                elif dim == 1:
                    site = sy
                else:
                    site = tl.load(Section + cols * 3 + dim - 1, valid, 0.0)
                center = tl.load(P + atoms * 6 + dim + 2, active, 0.0)
                dc = tl.sum(scale * (site[:, None] - center[None, :]), axis=0)
                tl.atomic_add(DP + atoms * 6 + dim + 2, dc, active, sem="relaxed")


class _MappedStreamed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, circle, section, offsets, layer, window_rows):
        ctx.save_for_backward(x, packed, circle, section, offsets)
        ctx.layer = layer
        ctx.window_rows = window_rows
        return streamed_forward(
            layer,
            x,
            prepared=(packed, circle, section, offsets),
            weight_chunk_rows=window_rows,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, packed, circle, section, offsets = ctx.saved_tensors
        layer = ctx.layer
        n, k = layer.shape
        m = x.shape[0]
        dy = dy.contiguous()
        dx = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        dp = torch.zeros_like(packed) if ctx.needs_input_grad[1] else None
        if dx is not None and m:
            chunk = min(ctx.window_rows, n)
            w = x.new_empty((chunk, k))
            for start in range(0, n, chunk):
                rows = min(chunk, n - start)
                materialize_logical[(math.ceil(rows / 64), layer.column_groups, 2)](
                    packed,
                    circle,
                    section,
                    offsets,
                    w,
                    N=n,
                    K=k,
                    S=64,
                    T=64,
                    CG=layer.column_groups,
                    G=layer.strip.chart.tile_count,
                    D=4,
                    PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                    BN=64,
                    BK=32,
                    BA=1,
                    FACTORED=True,
                    ROW_GROUP_START=start // 64,
                    LOCAL_W=True,
                    num_warps=4,
                    enable_fp_fusion=True,
                )
                dx.addmm_(dy[:, start : start + rows], w[:rows])
        if dp is not None and m:
            mapped_backward_atoms[(layer.strip.chart.tile_count * 4, 4)](
                x,
                dy,
                packed,
                circle,
                section,
                offsets,
                dp,
                M=m,
                N=n,
                K=k,
                S=64,
                T=64,
                CG=layer.column_groups,
                G=layer.strip.chart.tile_count,
                PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                BM=16,
                BN=16,
                BK=16,
                BA=8,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return dx, dp, None, None, None, None, None


def mapped_streamed_trainable(layer, x, *, boxes, witness_cols, window_rows=1024):
    if layer.tile_shape != (64, 64) or layer.shape[0] % 64 or layer.shape[1] % 64:
        raise ValueError("trainable mapped prototype requires full 64x64 tiles")
    prepared = trainable_boxed_prepare(
        layer.strip, layer.strip.atoms.p, boxes=boxes, witness_cols=witness_cols
    )
    return _MappedStreamed.apply(x, *prepared, layer, window_rows)
