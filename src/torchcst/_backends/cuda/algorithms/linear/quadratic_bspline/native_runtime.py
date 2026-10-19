"""Native small-support launch wrappers; no dense matrix temporaries."""

import triton as tr

from . import native_kernels as nk
from .executor import _report, _stamp


def forward(x, y, meta, count, origin, spacing, sizes, recipe):
    _stamp("native_forward", 0)
    compiled = nk.forward[(tr.cdiv(len(meta), recipe.native_atoms),)](
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
        tr.next_power_of_2(len(x)),
        recipe.native_sites,
        recipe.reciprocal,
        num_warps=recipe.native_warps,
        enable_fp_fusion=False,
    )
    _report("native_forward", compiled)
    _stamp("native_forward", 1)


def backward(x, dy, meta, count, origin, spacing, sizes, recipe, need_dx, need_dp):
    dx = x.new_zeros(x.shape) if need_dx else None
    dp = x.new_empty((len(meta), 4)) if need_dp else x.new_empty(0)
    _stamp("native_backward", 0)
    compiled = nk.backward[(tr.cdiv(len(meta), recipe.native_atoms),)](
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
        tr.next_power_of_2(len(x)),
        recipe.native_sites,
        recipe.reciprocal,
        need_dx,
        need_dp,
        num_warps=recipe.native_warps,
        enable_fp_fusion=False,
    )
    _report("native_backward", compiled)
    _stamp("native_backward", 1)
    return dx, dp
