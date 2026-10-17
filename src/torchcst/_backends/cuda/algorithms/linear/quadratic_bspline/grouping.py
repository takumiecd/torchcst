"""Live, graph-safe atom ordering by both clipped axis-support patch counts."""

import torch
import triton as tr
import triton.language as tl

from .kernels import _interval


@tr.jit
def support_keys(
    Meta,
    Origin,
    Spacing,
    Keys,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    BS: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    a = tl.program_id(0) * 128 + tl.arange(0, 128)
    radius = tl.load(Meta + a * 12 + 1, a < A, 1)
    co = tl.load(Meta + a * 12 + 2, a < A, 0)
    ci = tl.load(Meta + a * 12 + 3, a < A, 0)
    iho = tl.load(Meta + a * 12 + 10, a < A, 1)
    ihi = tl.load(Meta + a * 12 + 11, a < A, 1)
    lo, _ = _interval(co, radius, Origin, Spacing, NO, 0, 0, RECIPROCAL, iho)
    _, hi = _interval(co, radius, Origin, Spacing, NO, 0, 2, RECIPROCAL, iho)
    li, _ = _interval(ci, radius, Origin, Spacing, NI, 1, 0, RECIPROCAL, ihi)
    _, ri = _interval(ci, radius, Origin, Spacing, NI, 1, 2, RECIPROCAL, ihi)
    po = tl.cdiv(hi - lo, BS)
    pi = tl.cdiv(ri - li, BS)
    # The radix exceeds every possible input patch count, including full support.
    tl.store(Keys + a, po * (tr.cdiv(NI, BS) + 1) + pi, a < A)


def order_atoms(meta, origin, spacing, sizes, recipe):
    """Rebuild every forward; only IDs, never Parameters or optimizer states, move."""
    keys = torch.empty(len(meta), dtype=torch.int32, device=meta.device)
    support_keys[(tr.cdiv(len(meta), 128),)](
        meta,
        origin,
        spacing,
        keys,
        len(meta),
        *sizes,
        recipe.matrix_sites,
        recipe.reciprocal,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return torch.argsort(keys)
