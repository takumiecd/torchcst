"""Fixed-budget, device-resident PCG for the separated damped frame Gram.

This eager reference path bounds tile workspace; it deliberately does not
capture the full unrolled solve into a CUDA graph. It is an opt-in experiment,
not a claim of faster training than the direct solve.
"""

import math
from dataclasses import dataclass

import torch

from . import factored_taylor as ft


@dataclass(frozen=True)
class PCGOptions:
    iterations: int = 64
    rtol: float = 1e-5
    block_size: int = 32

    def __post_init__(self):
        for name in ("iterations", "block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"PCG {name} must be a positive integer")
        if not math.isfinite(self.rtol) or not 0 < self.rtol < 1:
            raise ValueError("PCG rtol must be finite and between zero and one")


def frame_diagonal(f, d, block_size):
    rows = []
    for start in range(0, d.numel(), block_size):
        stop = min(start + block_size, d.numel())
        us, vs = ft._frame_column_block(f, d, start, stop)
        result = d.new_zeros(stop - start)
        for i in range(4):
            for j in range(4):
                result = result + (us[i] * us[j]).sum(-1) * (vs[i] * vs[j]).sum(-1)
        rows.append(result)
    return torch.cat(rows).reshape_as(d)


@torch.no_grad()
def solve(factors, displacement, rhs, *, damping, options=None):
    """Return solution, validity, actual relative residual, active iterations.

    Factors are promoted before contraction, unlike the existing float32
    assembled Gram path. The same real-valued damped system is targeted, but
    rounding and approximate solve errors must be assessed separately.
    """
    if options is None:
        options = PCGOptions()
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("PCG requires finite positive damping")
    if rhs.shape != displacement.shape:
        raise ValueError("PCG rhs must match displacement shape")
    f = tuple(value.double() for value in factors)
    d, b = displacement.double(), rhs.double()

    def action(value):
        return ft.frame_gram_matvec(
            f, d, value, block_size=options.block_size, damping=damping
        )

    diagonal = frame_diagonal(f, d, options.block_size) + damping
    valid = torch.isfinite(diagonal).all() & (diagonal > 0).all()
    tiny = torch.finfo(b.dtype).tiny
    inverse = diagonal.clamp_min(tiny).reciprocal()
    x, r = torch.zeros_like(b), b.clone()
    z = r * inverse
    p = z.clone()
    rz = (r * z).sum()
    threshold = options.rtol * b.norm()
    count = torch.zeros((), device=b.device, dtype=torch.int64)
    for _ in range(options.iterations):
        active = r.norm() > threshold
        ap = action(p)
        pap = (p * ap).sum()
        healthy = (pap > 0) & torch.isfinite(pap) & torch.isfinite(rz)
        valid = valid & (~active | healthy)
        advance = active & healthy
        alpha = torch.where(advance, rz / pap.clamp_min(tiny), 0.0)
        x = x + alpha * p
        r = r - alpha * ap
        z = inverse * r
        next_rz = (r * z).sum()
        beta = torch.where(advance, next_rz / rz.clamp_min(tiny), 0.0)
        p = torch.where(advance, z + beta * p, torch.zeros_like(p))
        rz = next_rz
        count = count + advance.to(count.dtype)
    result = x.to(rhs.dtype)
    # Check the returned, rounded solution against the actual operator, not
    # only the recursively updated residual. No silent host fallback.
    residual = (action(result.double()) - b).norm()
    relative = residual / b.norm().clamp_min(tiny)
    valid = valid & torch.isfinite(result).all() & (residual <= threshold)
    return result, valid, relative, count
