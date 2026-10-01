"""Forward-only mapped windows extended with a bounded-memory backward prototype."""

import math

import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable

from experiments.cuda.linear.block_atom_major_backward import (
    mapped_backward_atoms_station,
)
from experiments.cuda.linear.block_interval_backward import (
    mapped_backward_atoms_interval,
)
from experiments.cuda.linear.block_materialize_kernel import materialize_logical
from experiments.cuda.linear.block_materialize_listed import materialize_listed
from experiments.cuda.linear.block_materialize_parallel import (
    materialize_listed_parallel,
)
from experiments.cuda.linear.block_streamed_forward import (
    streamed_forward,
    weight_fp_fusion_enabled,
)
from experiments.cuda.linear.block_tile_atom_lists import (
    build_tile_atom_lists,
    mapped_backward_atoms_listed,
)
from experiments.cuda.linear.bounded_gemm import bounded_gemm
from experiments.cuda.linear.bounded_gemm_fp16x3 import bounded_gemm_fp16x3
from experiments.cuda.linear.support_box_routing import atomic_bucket_sort
from torchcst._backends.cuda.algorithms.strip_torus.fused.kernels import (
    _profile,
    _sites,
    _values,
)
from torchcst._backends.cuda.algorithms.strip_torus.fused.preparation import (
    Pack,
    route_and_layout,
    tile_parameters,
)
from torchcst._backends.torch.operators.strip_torus.preparation import (
    PROFILE_KINDS,
    execution_plan,
)


def trainable_boxed_prepare(site, p, *, boxes, witness_cols, decode_mode="torch"):
    """Keep the amplitude and Torus decode graph while routing without gradients."""
    plan = execution_plan(site)
    station_rows = site.chart.tile_shape[0]
    center, amplitude, precision = tile_parameters(site.kernel, p)
    if decode_mode == "torch":
        decoded = site.chart.geometry.decode_centers(center)
    elif decode_mode == "fused":
        if plan.routing.starts.numel() > 256:
            raise ValueError(
                "fused Torus decode is only validated through 256 stations; "
                "use decode_mode='torch' for larger charts"
            )
        from experiments.cuda.linear.torus_decode_trainable import (
            trainable_decode_intrinsic_torus,
        )

        decoded = trainable_decode_intrinsic_torus(site.chart.geometry, center)
    else:
        raise ValueError("decode_mode must be torch or fused")
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
    CULL_BOX: tl.constexpr = False,
    WRITE_PARTIAL: tl.constexpr = False,
    MAX_CANDIDATES: tl.constexpr = 0,
    STATION_START=0,
    ROW_START=0,
    Partial=None,
):
    """Hoist site coordinates and reuse atom/site differences across derivatives."""
    tl.static_assert(not WRITE_WEIGHT or STAGED)
    tl.static_assert(not WRITE_PARTIAL or STAGED)
    tl.static_assert(not (WRITE_PARTIAL and CULL_BOX))
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
    if CULL_BOX:
        lx, hx = tl.min(sx, 0), tl.max(sx, 0)
        ly, hy = tl.min(sy, 0), tl.max(sy, 0)
        lz, hz = tl.min(sz, 0), tl.max(sz, 0)
        lw, hw = tl.min(sw, 0), tl.max(sw, 0)
    gradient = tl.reshape(dw, (BN * BK,))
    if WRITE_WEIGHT:
        weight = tl.full((BN * BK,), 0.0, tl.float32)
    candidate_base = 0
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
            precision = tl.load(P + atoms * 6 + 1, active, 0.0)
            if CULL_BOX:
                ex = tl.maximum(tl.maximum(lx - cx, cx - hx), 0.0)
                ey = tl.maximum(tl.maximum(ly - cy, cy - hy), 0.0)
                ez = tl.maximum(tl.maximum(lz - cz, cz - hz), 0.0)
                ew = tl.maximum(tl.maximum(lw - cw, cw - hw), 0.0)
                lower_bound = ex * ex + ey * ey + ez * ez + ew * ew
                possible = active & (lower_bound * precision <= 1.0001)
                compute = tl.sum(possible.to(tl.int32), 0) > 0
            else:
                possible = active
                compute = True
            if compute:
                dx = sx[:, None] - cx[None, :]
                dy = sy[:, None] - cy[None, :]
                dz = sz[:, None] - cz[None, :]
                dw_site = sw[:, None] - cw[None, :]
                squared = (dx * dx + dy * dy) + (dz * dz + dw_site * dw_site)
                value, slope = _profile(squared, precision[None, :], PROFILE)
                value = tl.where(valid[:, None] & possible[None, :], value, 0.0)
                slope = tl.where(valid[:, None] & possible[None, :], slope, 0.0)
                amplitude = tl.load(P + atoms * 6, active, 0.0)
                if WRITE_WEIGHT:
                    weight += tl.sum(value * amplitude[None, :], axis=1)
                da = tl.sum(gradient[:, None] * value, axis=0)
                scale = -2.0 * gradient[:, None] * slope * amplitude[None, :]
                dcx = tl.sum(scale * dx, axis=0)
                dcy = tl.sum(scale * dy, axis=0)
                dcz = tl.sum(scale * dz, axis=0)
                dcw = tl.sum(scale * dw_site, axis=0)
                if WRITE_PARTIAL:
                    tile_in_station = (tile % tr.cdiv(S, BN)) * tr.cdiv(
                        T, BK
                    ) + tl.program_id(1)
                    slot = candidate_base + atoms - begin
                    base = (
                        (
                            (station - STATION_START)
                            * (tr.cdiv(S, BN) * tr.cdiv(T, BK))
                            + tile_in_station
                        )
                        * MAX_CANDIDATES
                        + slot
                    ) * 5
                    tl.store(Partial + base, da, active)
                    tl.store(Partial + base + 1, dcx, active)
                    tl.store(Partial + base + 2, dcy, active)
                    tl.store(Partial + base + 3, dcz, active)
                    tl.store(Partial + base + 4, dcw, active)
                else:
                    tl.atomic_add(DP + atoms * 6, da, possible, sem="relaxed")
                    tl.atomic_add(DP + atoms * 6 + 2, dcx, possible, sem="relaxed")
                    tl.atomic_add(DP + atoms * 6 + 3, dcy, possible, sem="relaxed")
                    tl.atomic_add(DP + atoms * 6 + 4, dcz, possible, sem="relaxed")
                    tl.atomic_add(DP + atoms * 6 + 5, dcw, possible, sem="relaxed")
        candidate_base += end - begin
    if WRITE_WEIGHT:
        tl.store(
            DW + (logical_rows[:, None] - ROW_START) * K + logical_cols[None, :],
            tl.reshape(weight, (BN, BK)),
        )


@tr.jit
def reduce_mapped_backward_partials(
    Partial,
    Offsets,
    DP,
    G: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    BA: tl.constexpr,
    STATION_START,
):
    local_station = tl.program_id(0)
    station = STATION_START + local_station
    slot = tl.program_id(1) * BA + tl.arange(0, BA)
    if G == 1:
        begin0 = tl.load(Offsets)
        end0 = tl.load(Offsets + 1)
        length0 = end0 - begin0
        atom = begin0 + slot
        total = length0
    else:
        bucket0 = 2 * ((station + G - 1) % G) + 1
        begin0 = tl.load(Offsets + bucket0)
        end0 = tl.load(Offsets + bucket0 + 1)
        begin1 = tl.load(Offsets + 2 * station)
        end1 = tl.load(Offsets + 2 * station + 1)
        begin2 = tl.load(Offsets + 2 * station + 1)
        end2 = tl.load(Offsets + 2 * station + 2)
        length0 = end0 - begin0
        length1 = end1 - begin1
        length2 = end2 - begin2
        atom = tl.where(
            slot < length0,
            begin0 + slot,
            tl.where(
                slot < length0 + length1,
                begin1 + slot - length0,
                begin2 + slot - length0 - length1,
            ),
        )
        total = length0 + length1 + length2
    active = slot < total
    tiles = tl.arange(0, 8)
    base = ((local_station * 8 + tiles[:, None]) * MAX_CANDIDATES + slot[None, :]) * 5
    for component in tl.static_range(5):
        partial = tl.load(Partial + base + component, active[None, :], 0.0)
        summed = tl.sum(partial, 0)
        target = tl.where(component == 0, 0, component + 1)
        tl.atomic_add(DP + atom * 6 + target, summed, active, sem="relaxed")


def build_listed_forward_candidates(
    layer, packed, circle, section, offsets, *, unchecked_fixed_capacity=None
):
    """Build lists; unchecked capacity is only for fixed-offset diagnostics."""
    g = layer.strip.chart.tile_count
    if unchecked_fixed_capacity is not None:
        max_candidates = unchecked_fixed_capacity
    else:
        counts = offsets[1:] - offsets[:-1]
        stations = torch.arange(g, device=offsets.device)
        if g == 1:
            max_candidates = int(counts[0].item())
        else:
            candidate_counts = (
                counts[2 * ((stations + g - 1) % g) + 1]
                + counts[2 * stations]
                + counts[2 * stations + 1]
            )
            max_candidates = int(candidate_counts.max().item())
    list_dtype = (
        torch.uint8
        if max_candidates <= 256
        else torch.uint16
        if max_candidates <= 65536
        else torch.int32
    )
    atom_lists = torch.empty(
        (g, 4, max_candidates), device=packed.device, dtype=list_dtype
    )
    list_counts = torch.empty((g, 4), device=packed.device, dtype=torch.int32)
    build_tile_atom_lists[(g, 4)](
        packed,
        circle,
        section,
        offsets,
        atom_lists,
        list_counts,
        G=g,
        MAX_CANDIDATES=max_candidates,
        BA=8,
        COMPACT=True,
        BR=16,
        BC=64,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return atom_lists, list_counts, max_candidates


def build_listed_forward_candidates_csr(layer, packed, circle, section, offsets):
    """Build device-reserved tile lists with a provable 8*A entry bound."""
    g = layer.strip.chart.tile_count
    atoms = packed.shape[0]
    if atoms > (2**31 - 1) // 8:
        raise ValueError("CSR candidate cursor exceeds int32 capacity")
    # A packed atom belongs to at most two station neighborhoods and each
    # station reserves at most four site-tile spans of that neighborhood.
    atom_lists = torch.empty((8 * atoms,), device=packed.device, dtype=torch.int32)
    list_counts = torch.empty((g, 4), device=packed.device, dtype=torch.int32)
    list_bases = torch.empty_like(list_counts)
    cursor = torch.zeros((), device=packed.device, dtype=torch.int32)
    build_tile_atom_lists[(g, 4)](
        packed,
        circle,
        section,
        offsets,
        atom_lists,
        list_counts,
        G=g,
        MAX_CANDIDATES=0,
        BA=8,
        COMPACT=True,
        BR=16,
        BC=64,
        CSR=True,
        Bases=list_bases,
        Cursor=cursor,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return atom_lists, list_counts, list_bases, cursor


def build_listed_forward_candidates_bounded(
    layer,
    packed,
    circle,
    section,
    offsets,
    *,
    max_candidates=192,
    ba=8,
    warps=4,
    br=16,
    bc=64,
):
    """Graph-safe compact lists; overflow tiles scan their complete bucket span."""
    if type(max_candidates) is not int or not 1 <= max_candidates <= 65536:
        raise ValueError("bounded candidate capacity must be in [1, 65536]")
    g = layer.strip.chart.tile_count
    tiles = 4096 // (br * bc)
    rank_limit = 256 if max_candidates <= 256 else 65536
    atom_lists = torch.empty(
        (g, tiles, max_candidates),
        device=packed.device,
        dtype=torch.uint8 if rank_limit == 256 else torch.uint16,
    )
    list_counts = torch.empty((g, tiles), device=packed.device, dtype=torch.int32)
    build_tile_atom_lists[(g, tiles)](
        packed,
        circle,
        section,
        offsets,
        atom_lists,
        list_counts,
        G=g,
        MAX_CANDIDATES=max_candidates,
        BA=ba,
        COMPACT=True,
        BR=br,
        BC=bc,
        BOUNDED=True,
        RANK_LIMIT=rank_limit,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return atom_lists, list_counts, max_candidates


class _MappedStreamed(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        packed,
        circle,
        section,
        offsets,
        layer,
        window_rows,
        atom_kernel,
        cache_rows,
        gemm_mode,
        forward_gemm_mode,
        materialize_mode,
        listed_unroll,
        listed_builder_ba,
        listed_builder_warps,
        cache_weight_dtype,
        weight_tile_rows,
    ):
        ctx.layer = layer
        ctx.window_rows = window_rows
        ctx.atom_kernel = atom_kernel
        ctx.materialize_mode = materialize_mode
        ctx.listed_unroll = listed_unroll
        ctx.weight_tile_rows = weight_tile_rows
        ctx.gemm_mode = gemm_mode
        ctx.weight_fp_fusion = weight_fp_fusion_enabled(x.device)
        if materialize_mode == "listed_csr":
            ctx.listed_data = build_listed_forward_candidates_csr(
                layer, packed, circle, section, offsets
            )
        elif materialize_mode in ("listed", "listed_parallel", "listed_bounded"):
            builder = (
                build_listed_forward_candidates_bounded
                if materialize_mode == "listed_bounded"
                else build_listed_forward_candidates
            )
            ctx.listed_data = (
                builder(
                    layer,
                    packed,
                    circle,
                    section,
                    offsets,
                    ba=listed_builder_ba,
                    warps=listed_builder_warps,
                )
                if materialize_mode == "listed_bounded"
                else builder(layer, packed, circle, section, offsets)
            )
        else:
            ctx.listed_data = None
        ctx.weight_listed_data = (
            build_listed_forward_candidates_bounded(
                layer,
                packed,
                circle,
                section,
                offsets,
                ba=listed_builder_ba,
                warps=listed_builder_warps,
                br=weight_tile_rows,
            )
            if materialize_mode == "listed_bounded" and weight_tile_rows != 16
            else ctx.listed_data
        )
        reuse_input_weight = ctx.needs_input_grad[0] and (
            atom_kernel != "fused" or not ctx.needs_input_grad[1]
        )
        output, cached = streamed_forward(
            layer,
            x,
            prepared=(packed, circle, section, offsets),
            weight_chunk_rows=window_rows,
            cache_weight_rows=cache_rows if reuse_input_weight else 0,
            cache_weight_dtype=cache_weight_dtype,
            return_cache=True,
            gemm_mode=forward_gemm_mode,
            materialize_mode=materialize_mode,
            listed_data=ctx.weight_listed_data,
            listed_tile_shape=(weight_tile_rows, 64),
            listed_unroll=listed_unroll,
        )
        ctx.save_for_backward(x, packed, circle, section, offsets)
        ctx.cached_w = cached
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, packed, circle, section, offsets = ctx.saved_tensors
        cached = ctx.cached_w
        layer = ctx.layer
        n, k = layer.shape
        m = x.shape[0]
        dy = dy.contiguous()
        dx = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        need_dp = ctx.needs_input_grad[1]
        if need_dp and torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("mapped atom backward uses atomic accumulation")
        if dx is not None and m and (ctx.atom_kernel != "fused" or not need_dp):
            chunk = min(ctx.window_rows, n)
            w = x.new_empty((chunk, k))
            for start in range(0, n, chunk):
                rows = min(chunk, n - start)
                weight = (
                    cached[start : start + rows]
                    if start < cached.shape[0] and cached.dtype == torch.float32
                    else w[:rows]
                )
                if start < cached.shape[0] and cached.dtype == torch.float16:
                    weight.copy_(cached[start : start + rows])
                if start >= cached.shape[0]:
                    if ctx.weight_listed_data is not None:
                        csr = ctx.materialize_mode == "listed_csr"
                        if csr:
                            atom_lists, list_counts, list_bases, _ = (
                                ctx.weight_listed_data
                            )
                            max_candidates = 0
                        else:
                            atom_lists, list_counts, max_candidates = (
                                ctx.weight_listed_data
                            )
                            list_bases = None
                        kernel = (
                            materialize_listed_parallel
                            if ctx.materialize_mode == "listed_parallel"
                            else materialize_listed
                        )
                        kernel[
                            (
                                rows
                                // 64
                                * layer.column_groups
                                * (64 // ctx.weight_tile_rows),
                            )
                        ](
                            packed,
                            circle,
                            section,
                            atom_lists,
                            list_counts,
                            offsets,
                            weight,
                            K=k,
                            CG=layer.column_groups,
                            G=layer.strip.chart.tile_count,
                            PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                            MAX_CANDIDATES=max_candidates,
                            CSR=csr,
                            BOUNDED=ctx.materialize_mode == "listed_bounded",
                            **(
                                {"BR": ctx.weight_tile_rows, "BC": 64}
                                if ctx.materialize_mode != "listed_parallel"
                                else {}
                            ),
                            Bases=list_bases,
                            **(
                                {"LOOP_UNROLL": ctx.listed_unroll}
                                if ctx.materialize_mode != "listed_parallel"
                                else {}
                            ),
                            STATION_START=start // 64 * layer.column_groups,
                            ROW_START=start,
                            num_warps=4
                            if ctx.materialize_mode == "listed_parallel"
                            else 1,
                            **(
                                {"BA": 2}
                                if ctx.materialize_mode == "listed_parallel"
                                else {}
                            ),
                            enable_fp_fusion=ctx.weight_fp_fusion,
                        )
                    else:
                        materialize_logical[
                            (math.ceil(rows / 64), layer.column_groups, 2)
                        ](
                            packed,
                            circle,
                            section,
                            offsets,
                            weight,
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
                            enable_fp_fusion=ctx.weight_fp_fusion,
                        )
                if ctx.gemm_mode in ("tf32x3", "tf32x3_dx"):
                    bounded_gemm(
                        dy[:, start : start + rows],
                        weight,
                        dx,
                        add=start > 0,
                    )
                elif ctx.gemm_mode in ("fp16x3", "fp16x3_dx", "fp16x4_dw"):
                    bounded_gemm_fp16x3(
                        dy[:, start : start + rows],
                        weight,
                        dx,
                        add=start > 0,
                    )
                else:
                    dx.addmm_(dy[:, start : start + rows], weight)
        ctx.cached_w = None
        del cached
        dp = torch.zeros_like(packed) if need_dp else None
        if dp is not None and m:
            kernel = (
                mapped_backward_atoms_factored
                if ctx.atom_kernel
                in (
                    "factored",
                    "staged",
                    "staged_partial",
                    "staged_listed",
                    "atom_major",
                    "interval",
                    "fused",
                )
                else mapped_backward_atoms
            )
            atom_bk = 32 if ctx.atom_kernel == "factored" else 16
            if ctx.atom_kernel in (
                "staged",
                "staged_partial",
                "staged_listed",
                "atom_major",
                "interval",
                "fused",
            ):
                chunk = min(ctx.window_rows, n)
                if dx is None or ctx.atom_kernel == "fused":
                    w = x.new_empty((chunk, k))
                if ctx.atom_kernel in ("staged_partial", "staged_listed"):
                    if ctx.atom_kernel == "staged_listed":
                        listed_data = (
                            ctx.listed_data
                            if ctx.listed_data is not None
                            else build_listed_forward_candidates(
                                layer, packed, circle, section, offsets
                            )
                        )
                        csr = ctx.materialize_mode == "listed_csr"
                        if csr:
                            atom_lists, list_counts, list_bases, _ = listed_data
                            max_candidates = 0
                        else:
                            atom_lists, list_counts, max_candidates = listed_data
                            list_bases = None
                    else:
                        g = layer.strip.chart.tile_count
                        counts = offsets[1:] - offsets[:-1]
                        stations = torch.arange(g, device=offsets.device)
                        if g == 1:
                            max_candidates = int(counts[0].item())
                        else:
                            candidate_counts = (
                                counts[2 * ((stations + g - 1) % g) + 1]
                                + counts[2 * stations]
                                + counts[2 * stations + 1]
                            )
                            max_candidates = int(candidate_counts.max().item())
                        partial = x.new_empty(
                            (chunk // 64 * layer.column_groups, 8, max_candidates, 5)
                        )
                for start in range(0, n, chunk):
                    rows = min(chunk, n - start)
                    if ctx.gemm_mode == "tf32x3":
                        bounded_gemm(dy[:, start : start + rows].T, x, w[:rows])
                    elif ctx.gemm_mode in ("fp16x3", "fp16x4_dw"):
                        bounded_gemm_fp16x3(
                            dy[:, start : start + rows].T,
                            x,
                            w[:rows],
                            residual_product=ctx.gemm_mode == "fp16x4_dw",
                        )
                    else:
                        torch.mm(dy[:, start : start + rows].T, x, out=w[:rows])
                    if ctx.atom_kernel == "staged_listed":
                        mapped_backward_atoms_listed[
                            (rows // 64 * layer.column_groups * 4, 1)
                        ](
                            w,
                            packed,
                            circle,
                            section,
                            atom_lists,
                            list_counts,
                            offsets,
                            dp,
                            K=k,
                            CG=layer.column_groups,
                            G=layer.strip.chart.tile_count,
                            PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                            MAX_CANDIDATES=max_candidates,
                            CSR=csr,
                            BOUNDED=ctx.materialize_mode == "listed_bounded",
                            Bases=list_bases,
                            BA=1,
                            COMPACT=True,
                            BR=16,
                            BC=64,
                            STATION_START=start // 64 * layer.column_groups,
                            ROW_START=start,
                            OPT_TRIWEIGHT=True,
                            num_warps=1,
                            enable_fp_fusion=True,
                        )
                    elif ctx.atom_kernel == "atom_major":
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
                    elif ctx.atom_kernel == "interval":
                        mapped_backward_atoms_interval[
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
                            BR=8,
                            BK=16,
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
                            WRITE_PARTIAL=ctx.atom_kernel == "staged_partial",
                            MAX_CANDIDATES=max_candidates
                            if ctx.atom_kernel == "staged_partial"
                            else 0,
                            STATION_START=start // 64 * layer.column_groups,
                            ROW_START=start,
                            Partial=partial
                            if ctx.atom_kernel == "staged_partial"
                            else None,
                            num_warps=4,
                            enable_fp_fusion=ctx.atom_kernel
                            in ("staged", "staged_partial"),
                        )
                        if ctx.atom_kernel == "staged_partial":
                            station_count = rows // 64 * layer.column_groups
                            reduce_mapped_backward_partials[
                                (station_count, tr.cdiv(max_candidates, 32))
                            ](
                                partial,
                                offsets,
                                dp,
                                G=layer.strip.chart.tile_count,
                                MAX_CANDIDATES=max_candidates,
                                BA=32,
                                STATION_START=start // 64 * layer.column_groups,
                                num_warps=4,
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
        ctx.listed_data = None
        ctx.weight_listed_data = None
        return (dx, dp) + (None,) * 15


def mapped_streamed_trainable(
    layer,
    x,
    *,
    boxes,
    witness_cols,
    window_rows=1024,
    atom_kernel="baseline",
    cache_windows=0,
    gemm_mode="ieee",
    forward_gemm_mode="ieee",
    materialize_mode="default",
    listed_unroll=1,
    listed_builder_ba=8,
    listed_builder_warps=4,
    cache_weight_dtype=torch.float32,
    weight_tile_rows=16,
):
    if layer.tile_shape != (64, 64) or layer.shape[0] % 64 or layer.shape[1] % 64:
        raise ValueError("trainable mapped prototype requires full 64x64 tiles")
    if layer.shape[0] < 128:
        raise ValueError("trainable mapped prototype requires at least two row tiles")
    if type(window_rows) is not int or window_rows < 64 or window_rows % 64:
        raise ValueError("window_rows must be a positive multiple of 64")
    if atom_kernel not in (
        "baseline",
        "factored",
        "staged",
        "staged_partial",
        "staged_listed",
        "atom_major",
        "interval",
        "fused",
    ):
        raise ValueError("unknown atom_kernel")
    # Keep every temporary W/dW window strictly smaller than the logical matrix.
    window_rows = min(window_rows, (layer.shape[0] // 128) * 64)
    if type(cache_windows) is not int or cache_windows < 0:
        raise ValueError("cache_windows must be a nonnegative integer")
    cache_rows = cache_windows * window_rows
    if cache_weight_dtype == torch.float32 and cache_rows > layer.shape[0] // 2:
        raise ValueError("float32 cached windows must cover no more than half of W")
    if cache_weight_dtype == torch.float16 and cache_rows > layer.shape[0]:
        raise ValueError("float16 cached windows must cover no more than full W")
    if cache_weight_dtype not in (torch.float32, torch.float16):
        raise ValueError("cache_weight_dtype must be float32 or float16")
    if gemm_mode not in (
        "ieee",
        "tf32x3",
        "tf32x3_dx",
        "fp16x3",
        "fp16x3_dx",
        "fp16x4_dw",
    ):
        raise ValueError("unknown gemm_mode")
    if gemm_mode != "ieee" and atom_kernel != "staged_listed":
        raise ValueError("bounded GEMM is implemented for staged_listed only")
    if forward_gemm_mode not in ("ieee", "tf32x3", "fp16x3"):
        raise ValueError("unknown forward_gemm_mode")
    if materialize_mode not in (
        "default",
        "listed",
        "listed_parallel",
        "listed_csr",
        "listed_bounded",
    ):
        raise ValueError("unknown materialize_mode")
    if listed_unroll not in (1, 2, 4, 8):
        raise ValueError("listed_unroll must be 1, 2, 4, or 8")
    if listed_builder_ba not in (2, 4, 8, 16, 32, 64):
        raise ValueError("listed_builder_ba must be a supported power of two")
    if listed_builder_warps not in (1, 2, 4, 8):
        raise ValueError("listed_builder_warps must be 1, 2, 4, or 8")
    if weight_tile_rows not in (8, 16):
        raise ValueError("weight_tile_rows must be 8 or 16")
    if weight_tile_rows != 16 and materialize_mode != "listed_bounded":
        raise ValueError("8-row weight tiles require listed_bounded")
    prepared = trainable_boxed_prepare(
        layer.strip, layer.strip.atoms.p, boxes=boxes, witness_cols=witness_cols
    )
    return _MappedStreamed.apply(
        x,
        *prepared,
        layer,
        window_rows,
        atom_kernel,
        cache_rows,
        gemm_mode,
        forward_gemm_mode,
        materialize_mode,
        listed_unroll,
        listed_builder_ba,
        listed_builder_warps,
        cache_weight_dtype,
        weight_tile_rows,
    )
