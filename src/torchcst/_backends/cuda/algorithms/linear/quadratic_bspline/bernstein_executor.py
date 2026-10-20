"""Immutable positive node banks and bounded H/G, with complete Polar VJP."""

import torch
import triton as tr
from torch.autograd.function import once_differentiable

from . import bernstein_node_kernels as nk
from . import bernstein_owner_kernels as ok
from .executor import _prepare, _report, _stamp


def _bank(x, n, recipe, label):
    bank = x.new_empty((2 * n - 1, 3, len(x)))
    _stamp(label, 0)
    for level in range(n.bit_length()):
        width = 1 << level
        compiled = nk.bank[(n // width, tr.cdiv(len(x), recipe.bank_batch))](
            x,
            bank,
            n,
            len(x),
            width,
            recipe.bank_batch,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _report(label + f"_level{level}", compiled)
    _stamp(label, 1)
    return bank


def _owner(ids, atoms, n, dim, recipe, label, histogram=None):
    _stamp(label, 0)
    count = (
        histogram
        if histogram is not None
        else ids.new_zeros((2 * n - 1,), dtype=torch.int32)
    )
    if histogram is None:
        compiled = ok.histogram[(tr.cdiv(atoms * recipe.node_capacity, 256),)](
            ids,
            count,
            atoms,
            n,
            dim,
            recipe.node_capacity,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _report(label + "_histogram", compiled)
    offsets = ids.new_empty((2 * n,), dtype=torch.int32)
    cursor = ids.new_empty((2 * n - 1,), dtype=torch.int32)
    owners = ids.new_empty((atoms * recipe.node_capacity,), dtype=torch.int32)
    compiled = ok.offsets[(1,)](
        count, offsets, cursor, 2 * n - 1, 2 * n, num_warps=4, enable_fp_fusion=False
    )
    _report(label + "_scan", compiled)
    compiled = ok.fill[(tr.cdiv(atoms * recipe.node_capacity, 256),)](
        ids,
        cursor,
        owners,
        atoms,
        n,
        dim,
        recipe.node_capacity,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _report(label + "_fill", compiled)
    _stamp(label, 1)
    return offsets, owners


def _project(
    h,
    meta,
    ids,
    offsets,
    owners,
    origin,
    spacing,
    delta,
    y,
    n,
    dim,
    first,
    actual,
    recipe,
    index,
):
    label = ("output" if dim == 0 else "input") + f"_chunk{index}"
    _stamp("node_reduce_" + label, 0)
    compiled = ok.reduce_nodes[(2 * n - 1, tr.cdiv(actual, recipe.owner_batch))](
        h,
        meta,
        ids,
        offsets,
        owners,
        origin,
        spacing,
        delta,
        len(meta),
        n,
        actual,
        h.shape[1],
        dim,
        recipe.node_capacity,
        recipe.owner_rows,
        recipe.owner_batch,
        recipe.control_float32,
        recipe.owner_float32,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _report("node_reduce_" + label, compiled)
    _stamp("node_reduce_" + label, 1)
    _stamp("node_restore_" + label, 0)
    compiled = nk.restore[(tr.cdiv(n, 16), tr.cdiv(actual, 8))](
        delta,
        y,
        n,
        actual,
        16,
        8,
        n.bit_length() - 1,
        PAD=h.shape[1] - actual,
        FIRST=first,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _report("node_restore_" + label, compiled)
    _stamp("node_restore_" + label, 1)


class _Nodes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, chart, recipe, moments):
        no, ni = chart.grid_shape[0][0], chart.grid_shape[1][0]
        atoms, batch = len(p), len(x)
        ctx.recipe, ctx.sizes, ctx.empty = recipe, (no, ni), not atoms
        if not atoms:
            ctx.save_for_backward(x)
            return x.new_zeros((batch, no))
        meta, origin, spacing, source = _prepare(p, value, chart, recipe, moments)
        del source
        ids = x.new_empty((2, atoms, recipe.node_capacity), dtype=torch.int16)
        counts = x.new_empty((2, 3, atoms), dtype=torch.uint8)
        _stamp("node_prepare", 0)
        histogram = x.new_zeros((2 * no - 1,), dtype=torch.int32)
        compiled = ok.prepare_ids[(tr.cdiv(atoms, 128),)](
            meta,
            ids,
            counts,
            histogram,
            origin,
            spacing,
            atoms,
            no,
            ni,
            recipe.node_capacity,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _report("node_prepare", compiled)
        _stamp("node_prepare", 1)
        bank = _bank(x, ni, recipe, "bank_input")
        offsets, owners = _owner(
            ids, atoms, no, 0, recipe, "owner_prepare_output", histogram
        )
        del histogram
        pb = min(batch, recipe.batch_chunk)
        h = x.new_empty((atoms, pb), dtype=torch.float64)
        delta = x.new_empty((2 * no - 1, 3, pb), dtype=torch.float64)
        y = x.new_empty((batch, no))
        for index, first in enumerate(range(0, batch, recipe.batch_chunk)):
            actual = min(recipe.batch_chunk, batch - first)
            label = f"h_generate_chunk{index}"
            _stamp(label, 0)
            compiled = ok.forward[
                (
                    tr.cdiv(atoms, recipe.query_atoms),
                    tr.cdiv(actual, recipe.query_batch),
                )
            ](
                bank,
                meta,
                ids,
                counts,
                origin,
                spacing,
                h,
                atoms,
                ni,
                batch,
                first,
                actual,
                pb,
                recipe.query_atoms,
                recipe.query_batch,
                recipe.node_capacity,
                recipe.control_float32,
                recipe.query_float32,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _report(label, compiled)
            _stamp(label, 1)
            _project(
                h,
                meta,
                ids,
                offsets,
                owners,
                origin,
                spacing,
                delta,
                y,
                no,
                0,
                first,
                actual,
                recipe,
                index,
            )
        # Frame snapshots are immutable. Owner tables/H/delta are not retained;
        # every backward allocates its own scratch, including on other streams.
        ctx.save_for_backward(
            x,
            meta,
            origin,
            spacing,
            ids,
            counts,
            bank if ctx.needs_input_grad[1] else x.new_empty(0),
        )
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        if ctx.empty:
            (x,) = ctx.saved_tensors
            return (
                torch.zeros_like(x) if ctx.needs_input_grad[0] else None,
                x.new_empty((0, 4)) if ctx.needs_input_grad[1] else None,
                None,
                None,
                None,
                None,
            )
        x, meta, origin, spacing, ids, counts, input_bank = ctx.saved_tensors
        dy = dy.contiguous()
        recipe = ctx.recipe
        no, ni = ctx.sizes
        atoms, batch = len(meta), len(dy)
        need_dx, need_dp = ctx.needs_input_grad[:2]
        adjoint_bank = _bank(dy, no, recipe, "bank_adjoint")
        if need_dx:
            offsets, owners = _owner(ids, atoms, ni, 1, recipe, "owner_prepare_input")
        else:
            _stamp("owner_prepare_input", 0)
            _stamp("owner_prepare_input", 1)
        pb = min(batch, recipe.batch_chunk)
        g = dy.new_empty((atoms, pb), dtype=torch.float64) if need_dx else None
        delta = (
            dy.new_empty((2 * ni - 1, 3, pb), dtype=torch.float64) if need_dx else None
        )
        dx = x.new_empty(x.shape) if need_dx else None
        partial = dy.new_empty((atoms, 4), dtype=torch.float64) if need_dp else None
        for index, first in enumerate(range(0, batch, recipe.batch_chunk)):
            actual = min(recipe.batch_chunk, batch - first)
            label = f"g_vjp_chunk{index}"
            _stamp(label, 0)
            compiled = ok.backward[(tr.cdiv(atoms, recipe.query_atoms),)](
                input_bank if need_dp else None,
                adjoint_bank,
                meta,
                ids,
                counts,
                origin,
                spacing,
                g,
                partial,
                atoms,
                ni,
                no,
                batch,
                first,
                actual,
                pb,
                recipe.query_atoms,
                tr.next_power_of_2(actual),
                recipe.node_capacity,
                recipe.control_float32,
                recipe.query_float32,
                need_dx,
                need_dp,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _report(label, compiled)
            _stamp(label, 1)
            if need_dx:
                _project(
                    g,
                    meta,
                    ids,
                    offsets,
                    owners,
                    origin,
                    spacing,
                    delta,
                    dx,
                    ni,
                    1,
                    first,
                    actual,
                    recipe,
                    index,
                )
            else:
                for name in (
                    f"node_reduce_input_chunk{index}",
                    f"node_restore_input_chunk{index}",
                ):
                    _stamp(name, 0)
                    _stamp(name, 1)
        _stamp("atom_finalize", 0)
        dp = None
        if need_dp:
            dp = dy.new_empty((atoms, 4))
            compiled = ok.finalize[(tr.cdiv(4 * atoms, 256),)](
                partial, dp, atoms, num_warps=4, enable_fp_fusion=False
            )
            _report("atom_finalize", compiled)
        _stamp("atom_finalize", 1)
        return dx, dp, None, None, None, None


def apply(x, p, value, chart, recipe, moments):
    return _Nodes.apply(x.contiguous(), p, value, chart, recipe, moments)
