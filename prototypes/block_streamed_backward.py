"""Forward-only mapped windows extended with a bounded-memory backward prototype."""

import math

import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable

from prototypes.block_atom_major_backward import mapped_backward_atoms_station
from prototypes.block_materialize_kernel import materialize_logical
from prototypes.block_streamed_forward import streamed_forward
from prototypes.support_box_routing import atomic_bucket_sort
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan
from torchcst.nn._backends._triton_kernels import _profile, _sites, _values
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


@tr.jit
def mapped_backward_atoms_factored(
    DW,
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
    STAGED: tl.constexpr = False,
    WRITE_WEIGHT: tl.constexpr = False,
    STATION_START=0,
    ROW_START=0,
):
    """Hoist site coordinates and reuse atom/site differences across derivatives."""
    tl.static_assert(not WRITE_WEIGHT or STAGED)
    tile = tl.program_id(0)
    station = tile // tr.cdiv(S, BN) + STATION_START
    local = (tile % tr.cdiv(S, BN)) * BN
    col_start = tl.program_id(1) * BK
    local_cols = col_start + tl.arange(0, BK)
    logical_rows = (station // CG) * S + local + tl.arange(0, BN)
    logical_cols = (station % CG) * T + local_cols
    if STAGED:
        dw = tl.load(
            DW + (logical_rows[:, None] - ROW_START) * K + logical_cols[None, :],
            (logical_rows[:, None] < N) & (logical_cols[None, :] < K),
            0.0,
        )
    else:
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
    sz = tl.load(Section + cols * 3 + 1, valid, 0.0)
    sw = tl.load(Section + cols * 3 + 2, valid, 0.0)
    gradient = tl.reshape(dw, (BN * BK,))
    if WRITE_WEIGHT:
        weight = tl.full((BN * BK,), 0.0, tl.float32)
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
            cx = tl.load(P + atoms * 6 + 2, active, 0.0)
            cy = tl.load(P + atoms * 6 + 3, active, 0.0)
            cz = tl.load(P + atoms * 6 + 4, active, 0.0)
            cw = tl.load(P + atoms * 6 + 5, active, 0.0)
            dx = sx[:, None] - cx[None, :]
            dy = sy[:, None] - cy[None, :]
            dz = sz[:, None] - cz[None, :]
            dw_site = sw[:, None] - cw[None, :]
            squared = (dx * dx + dy * dy) + (dz * dz + dw_site * dw_site)
            precision = tl.load(P + atoms * 6 + 1, active, 0.0)
            value, slope = _profile(squared, precision[None, :], PROFILE)
            value = tl.where(valid[:, None] & active[None, :], value, 0.0)
            slope = tl.where(valid[:, None] & active[None, :], slope, 0.0)
            amplitude = tl.load(P + atoms * 6, active, 0.0)
            if WRITE_WEIGHT:
                weight += tl.sum(value * amplitude[None, :], axis=1)
            da = tl.sum(gradient[:, None] * value, axis=0)
            tl.atomic_add(DP + atoms * 6, da, active, sem="relaxed")
            scale = -2.0 * gradient[:, None] * slope * amplitude[None, :]
            tl.atomic_add(
                DP + atoms * 6 + 2, tl.sum(scale * dx, axis=0), active, sem="relaxed"
            )
            tl.atomic_add(
                DP + atoms * 6 + 3, tl.sum(scale * dy, axis=0), active, sem="relaxed"
            )
            tl.atomic_add(
                DP + atoms * 6 + 4, tl.sum(scale * dz, axis=0), active, sem="relaxed"
            )
            tl.atomic_add(
                DP + atoms * 6 + 5,
                tl.sum(scale * dw_site, axis=0),
                active,
                sem="relaxed",
            )
    if WRITE_WEIGHT:
        tl.store(
            DW + (logical_rows[:, None] - ROW_START) * K + logical_cols[None, :],
            tl.reshape(weight, (BN, BK)),
        )


class _MappedStreamed(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, packed, circle, section, offsets, layer, window_rows, atom_kernel
    ):
        ctx.save_for_backward(x, packed, circle, section, offsets)
        ctx.layer = layer
        ctx.window_rows = window_rows
        ctx.atom_kernel = atom_kernel
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
        if dp is not None and torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("mapped atom backward uses atomic accumulation")
        if dx is not None and m and (ctx.atom_kernel != "fused" or dp is None):
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
            kernel = (
                mapped_backward_atoms_factored
                if ctx.atom_kernel in ("factored", "staged", "atom_major", "fused")
                else mapped_backward_atoms
            )
            atom_bk = 32 if ctx.atom_kernel == "factored" else 16
            if ctx.atom_kernel in ("staged", "atom_major", "fused"):
                chunk = min(ctx.window_rows, n)
                if dx is None or ctx.atom_kernel == "fused":
                    w = x.new_empty((chunk, k))
                for start in range(0, n, chunk):
                    rows = min(chunk, n - start)
                    torch.mm(dy[:, start : start + rows].T, x, out=w[:rows])
                    if ctx.atom_kernel == "atom_major":
                        mapped_backward_atoms_station[
                            (rows // 64 * layer.column_groups, 16)
                        ](
                            w,
                            packed,
                            circle,
                            section,
                            offsets,
                            dp,
                            K=k,
                            S=64,
                            T=64,
                            CG=layer.column_groups,
                            G=layer.strip.chart.tile_count,
                            PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                            BN=16,
                            BK=32,
                            BA=4,
                            LANES=16,
                            STATION_START=start // 64 * layer.column_groups,
                            ROW_START=start,
                            num_warps=4,
                            enable_fp_fusion=False,
                        )
                    else:
                        kernel[(rows // 64 * layer.column_groups * 4, 2)](
                            w,
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
                            BK=32,
                            BA=8,
                            STAGED=True,
                            WRITE_WEIGHT=ctx.atom_kernel == "fused" and dx is not None,
                            STATION_START=start // 64 * layer.column_groups,
                            ROW_START=start,
                            num_warps=4,
                            enable_fp_fusion=False,
                        )
                        if ctx.atom_kernel == "fused" and dx is not None:
                            dx.addmm_(dy[:, start : start + rows], w[:rows])
            else:
                arguments = (x, dy, packed, circle, section, offsets, dp)
                if ctx.atom_kernel == "factored":
                    arguments = (packed, *arguments)
                kernel[(layer.strip.chart.tile_count * 4, 64 // atom_bk)](
                    *arguments,
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
                    BK=atom_bk,
                    BA=8,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return dx, dp, None, None, None, None, None, None


def mapped_streamed_trainable(
    layer, x, *, boxes, witness_cols, window_rows=1024, atom_kernel="baseline"
):
    if layer.tile_shape != (64, 64) or layer.shape[0] % 64 or layer.shape[1] % 64:
        raise ValueError("trainable mapped prototype requires full 64x64 tiles")
    if layer.shape[0] < 128:
        raise ValueError("trainable mapped prototype requires at least two row tiles")
    if type(window_rows) is not int or window_rows < 64 or window_rows % 64:
        raise ValueError("window_rows must be a positive multiple of 64")
    if atom_kernel not in ("baseline", "factored", "staged", "atom_major", "fused"):
        raise ValueError("unknown atom_kernel")
    # Keep every temporary W/dW window strictly smaller than the logical matrix.
    window_rows = min(window_rows, (layer.shape[0] // 128) * 64)
    prepared = trainable_boxed_prepare(
        layer.strip, layer.strip.atoms.p, boxes=boxes, witness_cols=witness_cols
    )
    return _MappedStreamed.apply(x, *prepared, layer, window_rows, atom_kernel)
