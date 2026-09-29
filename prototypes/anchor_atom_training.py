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
    for i in tl.range(0, limit, loop_unroll_factor=4):
        rank = tl.load(Lists + program * MAX_CANDIDATES + i, count >= 0, 0).to(tl.int32)
        rank = tl.where(count < 0, i, rank)
        atom, _ = _rank_to_atom(rank, Offsets, station, G)
        cx = tl.load(P + atom * 6 + 2)
        cy = tl.load(P + atom * 6 + 3)
        cz = tl.load(P + atom * 6 + 4)
        cw = tl.load(P + atom * 6 + 5)
        precision = tl.load(P + atom * 6 + 1)
        amplitude = tl.load(P + atom * 6)
        dx = sx - cx
        dy = sy - cy
        dz = sz - cz
        dw = sw - cw
        squared = (dx * dx + dy * dy) + (dz * dz + dw * dw)
        gap = tl.maximum(1.0 - squared * precision, 0.0)
        weight += amplitude * gap * gap * gap
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
    for i in range(limit):
        rank = tl.load(Lists + program * MAX_CANDIDATES + i, count >= 0, 0).to(tl.int32)
        rank = tl.where(count < 0, i, rank)
        atom, _ = _rank_to_atom(rank, Offsets, station, G)
        cx = tl.load(P + atom * 6 + 2)
        cy = tl.load(P + atom * 6 + 3)
        cz = tl.load(P + atom * 6 + 4)
        cw = tl.load(P + atom * 6 + 5)
        precision = tl.load(P + atom * 6 + 1)
        amplitude = tl.load(P + atom * 6)
        dx = sx - cx
        dy = sy - cy
        dz = sz - cz
        dw = sw - cw
        squared = (dx * dx + dy * dy) + (dz * dz + dw * dw)
        gap = tl.maximum(1.0 - squared * precision, 0.0)
        weighted = gradient * gap * gap
        da = tl.sum(weighted * gap)
        multiplier = 6.0 * precision * amplitude
        dcx = multiplier * tl.sum(weighted * dx)
        dcy = multiplier * tl.sum(weighted * dy)
        dcz = multiplier * tl.sum(weighted * dz)
        dcw = multiplier * tl.sum(weighted * dw)
        tl.atomic_add(DP + atom * 6, da, sem="relaxed")
        tl.atomic_add(DP + atom * 6 + 2, dcx, sem="relaxed")
        tl.atomic_add(DP + atom * 6 + 3, dcy, sem="relaxed")
        tl.atomic_add(DP + atom * 6 + 4, dcz, sem="relaxed")
        tl.atomic_add(DP + atom * 6 + 5, dcw, sem="relaxed")


class _AnchorSamples(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, circle, section, offsets, layer, layout):
        if type(layer.strip.kernel.profile) is not Triweight:
            raise ValueError("anchor prototype requires Triweight")
        lists, counts, capacity = build_listed_forward_candidates_bounded(
            layer, packed, circle, section, offsets, ba=32, warps=1
        )
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
            num_warps=1,
            enable_fp_fusion=True,
        )
        return dp, None, None, None, None, None


def anchor_samples(layer, layout, *, boxes, witness_cols):
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip, layer.strip.atoms.p, boxes=boxes, witness_cols=witness_cols
    )
    return _AnchorSamples.apply(packed, circle, section, offsets, layer, layout)


def anchor_trainable(layer, x, layout, *, boxes, witness_cols, basis_mode="dense"):
    samples = anchor_samples(layer, layout, boxes=boxes, witness_cols=witness_cols)
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
