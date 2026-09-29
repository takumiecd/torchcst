"""Experimental trainable CST operator evaluated only at interpolation anchors."""

from dataclasses import dataclass

import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable

from prototypes.block_streamed_backward import (
    build_listed_forward_candidates_bounded,
    trainable_boxed_prepare,
)
from prototypes.diagnose_cst_sampled_blocks import (
    column_basis,
    evenly_spaced_indices,
    interpolation_matrix,
)
from torchcst.kernels.compact import Triweight


@dataclass(frozen=True)
class AnchorLayout:
    output_basis: torch.Tensor
    input_basis: torch.Tensor
    local_output_basis: torch.Tensor
    local_input_basis: torch.Tensor
    row_sites: torch.Tensor
    row_positions: torch.Tensor
    col_sites: torch.Tensor
    rows_per_block: int
    cols_per_block: int
    row_pad: int
    col_pad: int


def make_anchor_layout(layer, row_count=16, column_segments=8):
    """Build fixed local cubic bases and their sampled-site lookup tables."""
    if layer.tile_shape != (64, 64):
        raise ValueError("anchor prototype requires 64x64 CST tiles")
    n, k = layer.shape
    if n % 64 or k % 64:
        raise ValueError("anchor prototype requires full CST tiles")
    row_anchors = evenly_spaced_indices(64, row_count)
    local_output = interpolation_matrix(64, row_anchors)
    local_input, column_anchors = column_basis(column_segments)
    row_groups = [
        [(index, row) for index, row in enumerate(row_anchors) if row // 16 == tile]
        for tile in range(4)
    ]
    row_pad = tr.next_power_of_2(max(map(len, row_groups)))
    col_pad = tr.next_power_of_2(len(column_anchors))
    sites = torch.full((4, row_pad), -1, dtype=torch.int32)
    positions = torch.full_like(sites, -1)
    for tile, group in enumerate(row_groups):
        for lane, (position, site) in enumerate(group):
            sites[tile, lane] = site
            positions[tile, lane] = position
    cols = torch.full((col_pad,), -1, dtype=torch.int32)
    cols[: len(column_anchors)] = torch.tensor(column_anchors, dtype=torch.int32)
    device = layer.strip.atoms.p.device
    return AnchorLayout(
        output_basis=torch.block_diag(*([local_output] * (n // 64))).to(device),
        input_basis=torch.block_diag(*([local_input] * (k // 64))).to(device),
        local_output_basis=local_output.to(device),
        local_input_basis=local_input.to(device),
        row_sites=sites.to(device),
        row_positions=positions.to(device),
        col_sites=cols.to(device),
        rows_per_block=row_count,
        cols_per_block=len(column_anchors),
        row_pad=row_pad,
        col_pad=col_pad,
    )


@tr.jit
def _anchor_sites(
    Circle,
    Section,
    RowSites,
    RowPositions,
    ColSites,
    station,
    row_tile,
    RP: tl.constexpr,
    CP: tl.constexpr,
):
    rl = tl.arange(0, RP)
    cl = tl.arange(0, CP)
    row = tl.load(RowSites + row_tile * RP + rl)
    row_position = tl.load(RowPositions + row_tile * RP + rl)
    col = tl.load(ColSites + cl)
    valid_row = row >= 0
    valid_col = col >= 0
    safe_row = tl.maximum(row, 0)
    safe_col = tl.maximum(col, 0)
    cos = tl.load(Circle + (station * 64 + safe_row) * 2)
    sin = tl.load(Circle + (station * 64 + safe_row) * 2 + 1)
    rho = tl.load(Section + safe_col * 3)
    z = tl.load(Section + safe_col * 3 + 1)
    w = tl.load(Section + safe_col * 3 + 2)
    sx = cos[:, None] * rho[None, :]
    sy = sin[:, None] * rho[None, :]
    valid = valid_row[:, None] & valid_col[None, :]
    return sx, sy, z[None, :], w[None, :], row_position, valid


@tr.jit
def _rank_to_atom(rank, Offsets, station, G: tl.constexpr):
    if G == 1:
        begin = tl.load(Offsets)
        atom = begin + rank
        capacity = tl.load(Offsets + 1) - begin
    else:
        prev = 2 * ((station + G - 1) % G) + 1
        begin0 = tl.load(Offsets + prev)
        length0 = tl.load(Offsets + prev + 1) - begin0
        begin1 = tl.load(Offsets + 2 * station)
        length1 = tl.load(Offsets + 2 * station + 1) - begin1
        begin2 = tl.load(Offsets + 2 * station + 1)
        length2 = tl.load(Offsets + 2 * station + 2) - begin2
        atom = tl.where(
            rank < length0,
            begin0 + rank,
            tl.where(
                rank < length0 + length1,
                begin1 + rank - length0,
                begin2 + rank - length0 - length1,
            ),
        )
        capacity = length0 + length1 + length2
    return atom, capacity


@tr.jit
def build_anchor_candidate_lists(
    P, Circle, Section, Offsets, RowSites, RowPositions, ColSites,
    Lists, Counts, G: tl.constexpr, RP: tl.constexpr, CP: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr, BA: tl.constexpr,
):
    station = tl.program_id(0)
    row_tile = tl.program_id(1)
    sx, sy, sz, sw, _, valid = _anchor_sites(
        Circle, Section, RowSites, RowPositions, ColSites, station, row_tile, RP, CP
    )
    inf = float("inf")
    lx = tl.min(tl.where(valid, sx, inf))
    hx = tl.max(tl.where(valid, sx, -inf))
    ly = tl.min(tl.where(valid, sy, inf))
    hy = tl.max(tl.where(valid, sy, -inf))
    lz = tl.min(tl.where(valid, tl.broadcast_to(sz, (RP, CP)), inf))
    hz = tl.max(tl.where(valid, tl.broadcast_to(sz, (RP, CP)), -inf))
    lw = tl.min(tl.where(valid, tl.broadcast_to(sw, (RP, CP)), inf))
    hw = tl.max(tl.where(valid, tl.broadcast_to(sw, (RP, CP)), -inf))
    _, capacity = _rank_to_atom(0, Offsets, station, G)
    base = (station * 4 + row_tile) * MAX_CANDIDATES
    count = 0
    bucket_rank = 0
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
            atom = atom_start + tl.arange(0, BA)
            active = atom < end
            cx = tl.load(P + atom * 6 + 2, active, 0.0)
            cy = tl.load(P + atom * 6 + 3, active, 0.0)
            cz = tl.load(P + atom * 6 + 4, active, 0.0)
            cw = tl.load(P + atom * 6 + 5, active, 0.0)
            precision = tl.load(P + atom * 6 + 1, active, 0.0)
            ex = tl.maximum(tl.maximum(lx - cx, cx - hx), 0.0)
            ey = tl.maximum(tl.maximum(ly - cy, cy - hy), 0.0)
            ez = tl.maximum(tl.maximum(lz - cz, cz - hz), 0.0)
            ew = tl.maximum(tl.maximum(lw - cw, cw - hw), 0.0)
            lower_bound = ex * ex + ey * ey + ez * ez + ew * ew
            possible = active & (lower_bound * precision <= 1.0001)
            position = count + tl.cumsum(possible.to(tl.int32), 0) - 1
            tl.store(
                Lists + base + position,
                bucket_rank + atom - begin,
                possible & (position < MAX_CANDIDATES) & (capacity <= 256),
            )
            count += tl.sum(possible.to(tl.int32), 0)
        bucket_rank += end - begin
    tl.store(Counts + station * 4 + row_tile,
             tl.where((count <= MAX_CANDIDATES) & (capacity <= 256), count, -1))


@tr.jit
def materialize_anchor_listed(
    P,
    Circle,
    Section,
    Lists,
    Counts,
    Offsets,
    RowSites,
    RowPositions,
    ColSites,
    Samples,
    CG: tl.constexpr,
    G: tl.constexpr,
    K_ANCHOR: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    RP: tl.constexpr,
    CP: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    BA: tl.constexpr,
):
    program = tl.program_id(0)
    station = program // 4
    row_tile = program % 4
    sx, sy, sz, sw, row_position, valid = _anchor_sites(
        Circle, Section, RowSites, RowPositions, ColSites, station, row_tile, RP, CP
    )
    weight = tl.full((RP, CP), 0.0, tl.float32)
    count = tl.load(Counts + program)
    _, capacity = _rank_to_atom(0, Offsets, station, G)
    limit = tl.where(count < 0, capacity, count)
    for start in tl.range(0, limit, BA, loop_unroll_factor=4 if BA == 1 else 1):
        lanes = start + tl.arange(0, BA)
        active = lanes < limit
        rank = tl.load(
            Lists + program * MAX_CANDIDATES + lanes,
            active & (count >= 0),
            0,
        ).to(tl.int32)
        rank = tl.where(count < 0, lanes, rank)
        atom, _ = _rank_to_atom(rank, Offsets, station, G)
        cx = tl.load(P + atom * 6 + 2, active, 0.0)
        cy = tl.load(P + atom * 6 + 3, active, 0.0)
        cz = tl.load(P + atom * 6 + 4, active, 0.0)
        cw = tl.load(P + atom * 6 + 5, active, 0.0)
        precision = tl.load(P + atom * 6 + 1, active, 0.0)
        amplitude = tl.load(P + atom * 6, active, 0.0)
        dx = sx[:, :, None] - cx[None, None, :]
        dy = sy[:, :, None] - cy[None, None, :]
        dz = sz[:, :, None] - cz[None, None, :]
        dw = sw[:, :, None] - cw[None, None, :]
        squared = (dx * dx + dy * dy) + (dz * dz + dw * dw)
        gap = tl.maximum(1.0 - squared * precision[None, None, :], 0.0)
        contribution = amplitude[None, None, :] * gap * gap * gap
        weight += tl.sum(contribution, 2)
    row_index = (station // CG) * R + row_position
    col_index = (station % CG) * C + tl.arange(0, CP)
    tl.store(
        Samples + row_index[:, None] * K_ANCHOR + col_index[None, :], weight, valid
    )


@tr.jit
def backward_anchor_listed(
    DS,
    P,
    Circle,
    Section,
    Lists,
    Counts,
    Offsets,
    RowSites,
    RowPositions,
    ColSites,
    DP,
    CG: tl.constexpr,
    G: tl.constexpr,
    K_ANCHOR: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    RP: tl.constexpr,
    CP: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    BA: tl.constexpr,
):
    program = tl.program_id(0)
    station = program // 4
    row_tile = program % 4
    sx, sy, sz, sw, row_position, valid = _anchor_sites(
        Circle, Section, RowSites, RowPositions, ColSites, station, row_tile, RP, CP
    )
    row_index = (station // CG) * R + row_position
    col_index = (station % CG) * C + tl.arange(0, CP)
    gradient = tl.load(
        DS + row_index[:, None] * K_ANCHOR + col_index[None, :], valid, 0.0
    )
    count = tl.load(Counts + program)
    _, capacity = _rank_to_atom(0, Offsets, station, G)
    limit = tl.where(count < 0, capacity, count)
    for start in range(0, limit, BA):
        lanes = start + tl.arange(0, BA)
        active = lanes < limit
        rank = tl.load(
            Lists + program * MAX_CANDIDATES + lanes,
            active & (count >= 0),
            0,
        ).to(tl.int32)
        rank = tl.where(count < 0, lanes, rank)
        atom, _ = _rank_to_atom(rank, Offsets, station, G)
        cx = tl.load(P + atom * 6 + 2, active, 0.0)
        cy = tl.load(P + atom * 6 + 3, active, 0.0)
        cz = tl.load(P + atom * 6 + 4, active, 0.0)
        cw = tl.load(P + atom * 6 + 5, active, 0.0)
        precision = tl.load(P + atom * 6 + 1, active, 0.0)
        amplitude = tl.load(P + atom * 6, active, 0.0)
        dx = sx[:, :, None] - cx[None, None, :]
        dy = sy[:, :, None] - cy[None, None, :]
        dz = sz[:, :, None] - cz[None, None, :]
        dw = sw[:, :, None] - cw[None, None, :]
        squared = (dx * dx + dy * dy) + (dz * dz + dw * dw)
        gap = tl.maximum(1.0 - squared * precision[None, None, :], 0.0)
        weighted = tl.where(
            active[None, None, :], gradient[:, :, None] * gap * gap, 0.0
        )
        da = tl.sum(tl.sum(weighted * gap, 0), 0)
        multiplier = 6.0 * precision * amplitude
        dcx = multiplier * tl.sum(tl.sum(weighted * dx, 0), 0)
        dcy = multiplier * tl.sum(tl.sum(weighted * dy, 0), 0)
        dcz = multiplier * tl.sum(tl.sum(weighted * dz, 0), 0)
        dcw = multiplier * tl.sum(tl.sum(weighted * dw, 0), 0)
        tl.atomic_add(DP + atom * 6, da, active, sem="relaxed")
        tl.atomic_add(DP + atom * 6 + 2, dcx, active, sem="relaxed")
        tl.atomic_add(DP + atom * 6 + 3, dcy, active, sem="relaxed")
        tl.atomic_add(DP + atom * 6 + 4, dcz, active, sem="relaxed")
        tl.atomic_add(DP + atom * 6 + 5, dcw, active, sem="relaxed")


class _AnchorSamples(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        packed,
        circle,
        section,
        offsets,
        layer,
        layout,
        forward_lanes,
        backward_lanes,
        list_mode,
    ):
        if type(layer.strip.kernel.profile) is not Triweight:
            raise ValueError("anchor prototype requires Triweight")
        if list_mode == "full_tile":
            lists, counts, capacity = build_listed_forward_candidates_bounded(
                layer, packed, circle, section, offsets, ba=32, warps=1
            )
        elif list_mode == "anchors":
            capacity = 192
            lists = torch.empty(
                (layer.strip.chart.tile_count, 4, capacity),
                device=packed.device, dtype=torch.uint8,
            )
            counts = torch.empty(
                (layer.strip.chart.tile_count, 4),
                device=packed.device, dtype=torch.int32,
            )
            build_anchor_candidate_lists[(layer.strip.chart.tile_count, 4)](
                packed, circle, section, offsets, layout.row_sites,
                layout.row_positions, layout.col_sites, lists, counts,
                G=layer.strip.chart.tile_count, RP=layout.row_pad,
                CP=layout.col_pad, MAX_CANDIDATES=capacity, BA=32,
                num_warps=1, enable_fp_fusion=False,
            )
        else:
            raise ValueError("list_mode must be full_tile or anchors")
        n, k = layer.shape
        samples = packed.new_empty(
            (n // 64 * layout.rows_per_block, k // 64 * layout.cols_per_block)
        )
        args = (
            packed,
            circle,
            section,
            lists,
            counts,
            offsets,
            layout.row_sites,
            layout.row_positions,
            layout.col_sites,
        )
        options = {
            "CG": layer.column_groups,
            "G": layer.strip.chart.tile_count,
            "K_ANCHOR": samples.shape[1],
            "R": layout.rows_per_block,
            "C": layout.cols_per_block,
            "RP": layout.row_pad,
            "CP": layout.col_pad,
            "MAX_CANDIDATES": capacity,
            "BA": forward_lanes,
            "num_warps": 1,
            "enable_fp_fusion": True,
        }
        materialize_anchor_listed[(layer.strip.chart.tile_count * 4,)](
            *args, samples, **options
        )
        ctx.save_for_backward(packed, circle, section, offsets, lists, counts)
        ctx.layer = layer
        ctx.layout = layout
        ctx.capacity = capacity
        ctx.backward_lanes = backward_lanes
        return samples

    @staticmethod
    @once_differentiable
    def backward(ctx, ds):
        packed, circle, section, offsets, lists, counts = ctx.saved_tensors
        layer, layout = ctx.layer, ctx.layout
        if torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("anchor atom backward uses atomic accumulation")
        dp = torch.zeros_like(packed)
        backward_anchor_listed[(layer.strip.chart.tile_count * 4,)](
            ds.contiguous(),
            packed,
            circle,
            section,
            lists,
            counts,
            offsets,
            layout.row_sites,
            layout.row_positions,
            layout.col_sites,
            dp,
            CG=layer.column_groups,
            G=layer.strip.chart.tile_count,
            K_ANCHOR=ds.shape[1],
            R=layout.rows_per_block,
            C=layout.cols_per_block,
            RP=layout.row_pad,
            CP=layout.col_pad,
            MAX_CANDIDATES=ctx.capacity,
            BA=ctx.backward_lanes,
            num_warps=1,
            enable_fp_fusion=True,
        )
        return dp, None, None, None, None, None, None, None, None


def anchor_samples(
    layer, layout, *, boxes, witness_cols, forward_lanes=1, backward_lanes=1,
    decode_mode="torch", list_mode="full_tile",
):
    if forward_lanes not in (1, 2, 4, 8):
        raise ValueError("forward_lanes must be 1, 2, 4, or 8")
    if backward_lanes not in (1, 2, 4, 8):
        raise ValueError("backward_lanes must be 1, 2, 4, or 8")
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip, layer.strip.atoms.p, boxes=boxes, witness_cols=witness_cols,
        decode_mode=decode_mode,
    )
    return _AnchorSamples.apply(
        packed, circle, section, offsets, layer, layout, forward_lanes,
        backward_lanes, list_mode,
    )


def anchor_trainable(
    layer,
    x,
    layout,
    *,
    boxes,
    witness_cols,
    basis_mode="dense",
    forward_lanes=1,
    backward_lanes=1,
    decode_mode="torch",
    list_mode="full_tile",
):
    samples = anchor_samples(
        layer,
        layout,
        boxes=boxes,
        witness_cols=witness_cols,
        forward_lanes=forward_lanes,
        backward_lanes=backward_lanes,
        decode_mode=decode_mode,
        list_mode=list_mode,
    )
    if basis_mode == "dense":
        return ((x @ layout.input_basis) @ samples.T) @ layout.output_basis.T
    if basis_mode == "block":
        m, k = x.shape
        n = layer.shape[0]
        projected = (x.reshape(m, k // 64, 64) @ layout.local_input_basis).reshape(
            m, -1
        )
        hidden = projected @ samples.T
        return (
            hidden.reshape(m, n // 64, layout.rows_per_block)
            @ layout.local_output_basis.T
        ).reshape(m, n)
    raise ValueError("basis_mode must be dense or block")
