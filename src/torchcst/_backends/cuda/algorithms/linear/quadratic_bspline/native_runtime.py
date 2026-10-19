"""Native small-support launch wrappers; no dense matrix temporaries."""

import triton as tr

from . import native_kernels as nk
from .executor import _report, _stamp


def forward(x, y, meta, count, origin, spacing, sizes, recipe):
    _stamp("native_forward", 0)
    mixed = getattr(recipe, "native_float32", False)
    if mixed:
        from . import native_mixed_kernels as mk

        kernel = mk.forward
    else:
        kernel = (
            nk.forward_loop if getattr(recipe, "native_loop", False) else nk.forward
        )
    batch = (
        recipe.native_batch
        if getattr(recipe, "native_loop", False)
        else tr.next_power_of_2(len(x))
    )
    compiled = kernel[(tr.cdiv(len(meta), recipe.native_atoms),)](
        x,
        y,
        meta,
        count,
        origin,
        spacing,
        len(meta),
        len(x),
        *sizes,
        recipe.native_atoms,
        batch,
        recipe.native_sites,
        recipe.reciprocal,
        *((True,) if mixed else ()),
        num_warps=recipe.native_warps,
        enable_fp_fusion=False,
    )
    _report("native_forward", compiled)
    if mixed:
        cold = kernel[(tr.cdiv(len(meta), recipe.native_atoms),)](
            x,
            y,
            meta,
            count,
            origin,
            spacing,
            len(meta),
            len(x),
            *sizes,
            recipe.native_atoms,
            batch,
            recipe.native_sites,
            recipe.reciprocal,
            False,
            num_warps=recipe.native_warps,
            enable_fp_fusion=False,
        )
        _report("native_forward_cold", cold)
    _stamp("native_forward", 1)


def backward(
    x, dy, meta, count, origin, spacing, sizes, recipe, need_dx, need_dp, zero_dp=False
):
    dx = x.new_zeros(x.shape) if need_dx else None
    dp = (
        (x.new_zeros((len(meta), 4)) if zero_dp else x.new_empty((len(meta), 4)))
        if need_dp
        else x.new_empty(0)
    )
    _stamp("native_backward", 0)
    mixed = getattr(recipe, "native_float32", False)
    if mixed:
        from . import native_mixed_kernels as mk

        kernel = mk.backward
    else:
        kernel = (
            nk.backward_loop if getattr(recipe, "native_loop", False) else nk.backward
        )
    batch = (
        recipe.native_batch
        if getattr(recipe, "native_loop", False)
        else tr.next_power_of_2(len(x))
    )
    compiled = kernel[(tr.cdiv(len(meta), recipe.native_atoms),)](
        x,
        dy,
        dx,
        dp,
        meta,
        count,
        origin,
        spacing,
        len(meta),
        len(x),
        *sizes,
        recipe.native_atoms,
        batch,
        recipe.native_sites,
        recipe.reciprocal,
        need_dx,
        need_dp,
        *((True, recipe.contract_float32) if mixed else ()),
        num_warps=recipe.native_warps,
        enable_fp_fusion=False,
    )
    _report("native_backward", compiled)
    if mixed:
        cold = kernel[(tr.cdiv(len(meta), recipe.native_atoms),)](
            x,
            dy,
            dx,
            dp,
            meta,
            count,
            origin,
            spacing,
            len(meta),
            len(x),
            *sizes,
            recipe.native_atoms,
            batch,
            recipe.native_sites,
            recipe.reciprocal,
            need_dx,
            need_dp,
            False,
            recipe.contract_float32,
            num_warps=recipe.native_warps,
            enable_fp_fusion=False,
        )
        _report("native_backward_cold", cold)
    _stamp("native_backward", 1)
    return dx, dp
