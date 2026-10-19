"""Per-forward endpoint order and bounded FP64 scatter workspaces."""

import torch
import triton as tr

from . import adjoint_kernels as ak
from .executor import _report, _stamp


def order(cuts, count, sizes, recipe):
    _stamp("adjoint_order", 0)
    if recipe.adjoint_order != "none":
        a = cuts.shape[-1]
        dim = 1 if recipe.adjoint_order.startswith("input") else 0
        if recipe.adjoint_order.endswith("_bucket"):
            levels = (sizes[dim] + 1).bit_length()
            hist = torch.zeros(
                (sizes[dim] + 1) * levels, dtype=torch.int32, device=cuts.device
            )
            cursor = torch.zeros_like(hist)
            rank = torch.empty(a, dtype=torch.int64, device=cuts.device)
            ak.bucket_hist[(tr.cdiv(a, 128),)](cuts, count, hist, a, dim, levels, 128)
            ends = hist.cumsum(0, dtype=torch.int32)
            ak.bucket_place[(tr.cdiv(a, 128),)](
                cuts, count, ends, cursor, rank, a, dim, levels, 128
            )
        else:
            keys = torch.empty(a, dtype=torch.int64, device=cuts.device)
            ak.keys[(tr.cdiv(a, 128),)](cuts, count, keys, a, sizes[dim], dim, 128)
            rank = keys.argsort(stable=True)
    else:
        rank = torch.empty(0, dtype=torch.int64, device=cuts.device)
    _stamp("adjoint_order", 1)
    return rank


def backward(
    rank,
    input_bank,
    saved_h,
    saved_hp,
    adjoint_bank,
    meta,
    cuts,
    co,
    delta,
    dp,
    count,
    origin,
    spacing,
    sizes,
    b,
    recipe,
    mode,
    need_dx,
    vjp,
):
    a, shards = len(meta), recipe.scatter_shards
    zero = recipe.scatter_scope == "zero" and shards > 1
    if zero:
        edges = torch.zeros((shards, 3, b), dtype=torch.float64, device=meta.device)
        target = delta
    elif shards > 1:
        target = torch.zeros(
            (shards, 3, sizes[1] + 1, b), dtype=torch.float64, device=meta.device
        )
        edges = target
    else:
        target = edges = delta
    compiled = ak.backward[(tr.cdiv(a, recipe.adjoint_atoms),)](
        rank,
        edges,
        input_bank,
        saved_h,
        saved_hp,
        adjoint_bank,
        meta,
        cuts,
        co,
        meta.new_empty(0),
        target,
        dp,
        count,
        origin,
        spacing,
        a,
        b,
        *sizes,
        recipe.adjoint_atoms,
        tr.next_power_of_2(b),
        recipe.reciprocal,
        recipe.coefficient_cache,
        recipe.coefficient_derivatives,
        False,
        recipe.edge_reduce,
        True,
        recipe.adjoint_share,
        recipe.adjoint_reduce,
        need_dx,
        vjp,
        mode,
        recipe.adjoint_order != "none",
        shards,
        zero,
        recipe.fold_zero,
        num_warps=recipe.adjoint_warps,
        enable_fp_fusion=False,
    )
    _report("stream_backward", compiled)
    _stamp("dx_reduce", 0)
    if need_dx and shards > 1:
        if zero:
            delta[:, 0, :].copy_(edges.sum(0))
        else:
            delta = target.sum(0)
    _stamp("dx_reduce", 1)
    return delta
