"""A disjoint, live per-atom sum of grouped narrow W and broad moment paths."""

import triton as tr

from . import kernels as k
from . import parallel_kernels as pk
from .executor import _query, _report, _scatter, _stamp
from .grouping import partition_atoms


def forward(ctx, x, meta, origin, spacing, recipe, bases, sizes):
    a = len(meta)
    _stamp("group", 0)
    order, count = partition_atoms(meta, origin, spacing, sizes, recipe)
    _stamp("group", 1)
    w = x.new_zeros(sizes)
    _stamp("w_build", 0)
    compiled = k.matrix_build[(tr.cdiv(a, recipe.matrix_atoms),)](
        meta,
        w,
        origin,
        spacing,
        a,
        *sizes,
        recipe.matrix_sites,
        recipe.matrix_atoms,
        recipe.reciprocal,
        Order=order,
        GROUPED=True,
        Count=count,
        PARTITIONED=True,
        num_warps=recipe.matrix_warps,
        enable_fp_fusion=False,
    )
    _stamp("w_build", 1)
    _report("w_build", compiled)
    _stamp("y_gemm", 0)
    y = x @ w.T
    _stamp("y_gemm", 1)
    h, hp = _query(x, meta, origin, spacing, 1, recipe, bases[1], order, count)
    y = _scatter(h, meta, origin, spacing, 0, sizes[0], recipe, order, count, y)
    ctx.sizes, ctx.recipe, ctx.bases = sizes, recipe, bases
    ctx.save_for_backward(meta, x, w, h, hp, origin, spacing, order, count)
    return y


def backward(ctx, dy):
    meta, x, w, h, hp, origin, spacing, order, count = ctx.saved_tensors
    recipe = ctx.recipe
    a = len(meta)
    dy = dy.contiguous()
    dx = dp = None
    _stamp("dx_gemm", 0)
    if ctx.needs_input_grad[0]:
        dx = dy @ w
    _stamp("dx_gemm", 1)
    if ctx.needs_input_grad[1]:
        _stamp("dw_gemm", 0)
        dw = dy.T @ x
        _stamp("dw_gemm", 1)
        dp = dy.new_empty((a, 4))
        _stamp("matrix_atom_vjp", 0)
        compiled = k.matrix_vjp[(tr.cdiv(a, recipe.matrix_atoms),)](
            meta,
            dw,
            dp,
            origin,
            spacing,
            a,
            *ctx.sizes,
            recipe.matrix_sites,
            recipe.matrix_atoms,
            recipe.reciprocal,
            True,
            Order=order,
            GROUPED=True,
            Count=count,
            PARTITIONED=True,
            num_warps=recipe.matrix_warps,
            enable_fp_fusion=False,
        )
        _stamp("matrix_atom_vjp", 1)
        _report("matrix_atom_vjp", compiled)
        del dw
    g, gp = _query(dy, meta, origin, spacing, 0, recipe, ctx.bases[0], order, count)
    if ctx.needs_input_grad[0]:
        dx = _scatter(
            g, meta, origin, spacing, 1, ctx.sizes[1], recipe, order, count, dx
        )
    if ctx.needs_input_grad[1]:
        _stamp("atom_vjp", 0)
        compiled = pk.vjp_grouped[(tr.cdiv(a, recipe.axis_atoms),)](
            h,
            hp,
            g,
            gp,
            meta,
            dp,
            a,
            len(dy),
            recipe.axis_atoms,
            tr.next_power_of_2(len(dy)),
            Order=order,
            Count=count,
            ROUTED=True,
            num_warps=recipe.axis_warps,
            enable_fp_fusion=False,
        )
        _stamp("atom_vjp", 1)
        _report("atom_vjp", compiled)
    return dx, dp, None, None, None, None, None
