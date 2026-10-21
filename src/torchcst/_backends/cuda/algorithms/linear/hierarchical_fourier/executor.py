"""Bounded per-chunk coefficients/H, shared Z and output-owner C reductions.

No X-dependent moment or live geometry survives into another invocation.
Backward recomputes chunk-local H/G from saved X/P, with Tensor version checks.
"""

import math
from functools import cache

import torch
from torch.autograd.function import once_differentiable

from .contract import FourierLinearSpec, FourierRecipe


def bank_families(spec, recipe):
    """Conservative finite bank set over the declared bounded continuous I."""
    p = spec.period
    a0 = (p.minimum_tile_width / 2 + 1.25 * p.minimum_half_period) / (
        1 / p.relative_width_floor - 1
    )
    pairs = []
    if recipe.strategy == "shared":
        for j in range(12):
            tile = p.minimum_tile_width * 2**j
            ns = round(tile / spec.spacing)
            lo = max(0.1, a0 * 2**j, recipe.width_cutoff)
            hi = min(10.0, 1.15 * a0 * 2 ** (j + 1))
            if lo > hi or ns < recipe.minimum_sites or ns > spec.size:
                continue
            step = p.minimum_half_period * 2**j
            for n in range(
                math.ceil((lo + tile / 2) / step), math.ceil((hi + tile / 2) / step) + 2
            ):
                pairs.append((j, n * 2**j, ns))
    return pairs


@cache
def tables(spec, recipe, device):
    from torchcst._backends.torch.profiles.fourier_peak import _projection

    matrix, offset = _projection(spec.modes, torch.device("cpu"))
    pairs = bank_families(spec, recipe)
    kmax = max((k for _, k, _ in pairs), default=1)
    lookup = torch.full((12, kmax + 1), -1, dtype=torch.int32)
    banks, nodes = [], []
    basis = []
    n = torch.arange(spec.modes + 1, dtype=torch.float64)
    x = torch.arange(spec.size, dtype=torch.float64) * spec.spacing
    for bank, (j, k, ns) in enumerate(pairs):
        lookup[j, k] = bank
        banks.append((ns, len(nodes)))
        nodes.extend((bank, t * ns, ns) for t in range(spec.size // ns))
        angle = x[:, None] * n * math.pi / (k * spec.period.minimum_half_period)
        basis.append(torch.cat((angle.cos(), angle[:, 1:].sin()), -1).float())
    r = 2 * spec.modes + 1
    phi = torch.stack(basis) if basis else torch.empty(0, spec.size, r)
    return {
        "matrix": matrix.to(device),
        "offset": offset.to(device),
        "lookup": lookup.to(device),
        "banks": torch.tensor(banks, dtype=torch.int32, device=device).reshape(-1, 2),
        "nodes": torch.tensor(nodes, dtype=torch.int32, device=device).reshape(-1, 3),
        "phi": phi.to(device),
        "kmax": kmax,
        "static_bytes": sum(
            t.numel() * t.element_size() for t in (matrix, offset, lookup, phi)
        )
        + 8 * len(banks)
        + 12 * len(nodes),
    }


def _prepare(p, spec, recipe, tab):
    import triton as tr

    from . import kernels as k

    a, m = len(p), spec.modes
    meta = torch.empty((a, 4, 4), device=p.device, dtype=torch.float64)
    co = torch.empty((a, 4, m + 1), device=p.device, dtype=torch.float64)
    ci = torch.empty_like(co)
    norm = torch.empty((a, 2, 3), device=p.device, dtype=torch.float64)
    beta = torch.empty((2, a, 4, 2, 2 * m + 1), device=p.device, dtype=torch.float32)
    period = spec.period
    a0 = (period.minimum_tile_width / 2 + 1.25 * period.minimum_half_period) / (
        1 / period.relative_width_floor - 1
    )
    k.prepare[(a,)](
        p,
        tab["matrix"],
        tab["offset"],
        tab["lookup"],
        meta,
        co,
        ci,
        a,
        m,
        tr.next_power_of_2(m + 1),
        period.minimum_tile_width,
        period.minimum_half_period,
        a0,
        tab["kmax"],
        recipe.width_cutoff,
        recipe.strategy == "shared",
        num_warps=4,
        enable_fp_fusion=False,
    )
    k.norms[(a, 2)](
        p, meta, co, ci, norm, a, spec.size, m, 128, num_warps=4, enable_fp_fusion=False
    )
    k.pack[(a, 2)](
        p,
        meta,
        co,
        ci,
        beta,
        a,
        m,
        2 * m + 1,
        tr.next_power_of_2(2 * m + 1),
        num_warps=4,
        enable_fp_fusion=False,
    )
    order = begin = None
    if len(tab["banks"]):
        bank = meta[:, :, 2].reshape(-1).int()
        key = torch.where(bank >= 0, bank, len(tab["banks"]))
        key, order = key.sort(stable=True)
        begin = torch.searchsorted(
            key, torch.arange(len(tab["banks"]) + 1, device=p.device, dtype=key.dtype)
        )
    return meta, co, ci, norm, beta, order, begin


def _moments(x, spec, tab):
    import triton as tr

    from . import kernels as k

    b = len(x)
    j = len(tab["nodes"])
    r = 2 * spec.modes + 1
    z = x.new_empty((b, j, r))
    if j:
        k.moments[(j, tr.cdiv(b, 16))](
            x,
            tab["phi"],
            tab["nodes"],
            z,
            b,
            spec.size,
            r,
            j,
            16,
            64,
            tr.next_power_of_2(r),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return z


def _contract(x, p, prepared, spec, tab, z, axis, grad):
    import triton as tr

    from . import kernels as k

    meta, co, ci, _norm, beta, _, _ = prepared
    b, a = len(x), len(p)
    r = 2 * spec.modes + 1
    h = x.new_empty((b, a, 3))
    k.contract[(a, b)](
        x,
        p,
        meta,
        co,
        ci,
        beta,
        tab["banks"],
        z,
        h,
        b,
        spec.size,
        a,
        spec.modes,
        r,
        len(tab["nodes"]),
        axis,
        grad,
        4,
        128,
        tr.next_power_of_2(r),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return h


def _deposit(h, p, prepared, spec, tab, y, c, axis):
    import triton as tr

    from . import kernels as k

    meta, co, ci, norm, beta, order, begin = prepared
    b, a = len(h), len(p)
    r = 2 * spec.modes + 1
    j = len(tab["nodes"])
    if j:
        k.aggregate[(j, tr.cdiv(b, 16))](
            h,
            p,
            meta,
            norm,
            beta,
            order,
            begin,
            tab["nodes"],
            c,
            b,
            spec.size,
            a,
            r,
            j,
            axis,
            16,
            32,
            tr.next_power_of_2(r),
            num_warps=4,
            enable_fp_fusion=False,
        )
    k.deposit[(a * 4, tr.cdiv(b, 4))](
        h,
        p,
        meta,
        co,
        ci,
        norm,
        tab["banks"],
        y,
        b,
        spec.size,
        a,
        spec.modes,
        axis,
        4,
        128,
        num_warps=4,
        enable_fp_fusion=False,
    )


def _reconstruct(c, spec, tab, y):
    import triton as tr

    from . import kernels as k

    b = len(y)
    r = 2 * spec.modes + 1
    j = len(tab["nodes"])
    if j:
        k.reconstruct[(tr.cdiv(b, 4), tr.cdiv(spec.size, 32))](
            c,
            tab["phi"],
            tab["banks"],
            y,
            b,
            spec.size,
            r,
            j,
            len(tab["banks"]),
            4,
            32,
            tr.next_power_of_2(r),
            num_warps=4,
            enable_fp_fusion=False,
        )


class _Apply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, spec, recipe, tab):
        z = _moments(x, spec, tab)
        y = x.new_zeros((len(x), spec.size))
        c = x.new_zeros(z.shape)
        for start in range(0, len(p), recipe.atom_chunk):
            pc = p[start : start + recipe.atom_chunk]
            prep = _prepare(pc, spec, recipe, tab)
            h = _contract(x, pc, prep, spec, tab, z, 1, False)
            _deposit(h, pc, prep, spec, tab, y, c, 0)
        _reconstruct(c, spec, tab, y)
        ctx.save_for_backward(x, p)
        ctx.spec, ctx.recipe, ctx.tab = spec, recipe, tab
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import kernels as k

        x, p = ctx.saved_tensors
        spec, recipe, tab = ctx.spec, ctx.recipe, ctx.tab
        dy = dy.contiguous()
        zx, zg = _moments(x, spec, tab), _moments(dy, spec, tab)
        dx = torch.zeros_like(x)
        dp = torch.empty_like(p)
        c = x.new_zeros(zx.shape)
        for start in range(0, len(p), recipe.atom_chunk):
            pc = p[start : start + recipe.atom_chunk]
            prep = _prepare(pc, spec, recipe, tab)
            h = _contract(x, pc, prep, spec, tab, zx, 1, True)
            g = _contract(dy, pc, prep, spec, tab, zg, 0, True)
            _deposit(g, pc, prep, spec, tab, dx, c, 1)
            k.vjp[(len(pc),)](
                h,
                g,
                pc,
                prep[3],
                dp[start : start + len(pc)],
                len(x),
                len(pc),
                tr.next_power_of_2(len(x)),
                num_warps=4,
                enable_fp_fusion=False,
            )
        _reconstruct(c, spec, tab, dx)
        return dx, dp, None, None, None


def apply(x, p, spec, recipe):
    if type(spec) is not FourierLinearSpec or type(recipe) is not FourierRecipe:
        raise TypeError("requires FourierLinearSpec and FourierRecipe")
    if (
        x.device.type != "cuda"
        or x.dtype != torch.float32
        or p.device != x.device
        or p.dtype != x.dtype
        or x.ndim != 2
        or p.ndim != 2
        or p.shape[1] != 4
        or x.shape[1] != spec.size
        or not x.is_contiguous()
        or not p.is_contiguous()
        or not 1 <= len(x) <= 64
        or len(p) < 1
    ):
        raise ValueError("requires contiguous CUDA FP32 X[B,N], P[A,4], B1..64")
    if torch.is_autocast_enabled("cuda") or torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("requires IEEE FP32, no autocast or TF32")
    tab = tables(spec, recipe, x.device)
    return _Apply.apply(x, p, spec, recipe, tab)
