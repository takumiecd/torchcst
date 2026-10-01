"""Adapters to existing autograd implementations, without changing GPU kernels."""

from functools import cache
from types import SimpleNamespace


@cache
def _geometry(operator, device):
    # Only structural offsets are cached here. Norms, supports, row buckets and
    # gradients remain owned by each forward/backward invocation.
    return SimpleNamespace(
        sizes=operator.sizes, origin=operator.origin, spacing=operator.spacing
    )


def execute_full(*, x, parameters, operator, recipe):
    from ....normalized_strip.full import _AtlasApply

    return _AtlasApply.apply(
        x,
        parameters,
        _geometry(operator, x.device),
        recipe.atom_num_warps,
        recipe.sorted_forward,
        recipe.sorted_backward,
        recipe.support,
        recipe.enable_fp_fusion,
        recipe.saved_support_flags,
        recipe.tuple_grads,
    )


def execute_window(*, x, parameters, operator, recipe):
    from ....normalized_strip.window import PackedWindowProvider
    from ....normalized_strip.window_gemm import window_linear

    rows = min(recipe.window_rows, operator.n)
    return window_linear(
        x,
        parameters,
        PackedWindowProvider(
            _geometry(operator, x.device), rows, recipe.enable_fp_fusion
        ),
        rows=operator.n,
        window=rows,
    )
