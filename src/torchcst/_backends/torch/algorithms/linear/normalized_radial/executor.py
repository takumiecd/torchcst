"""Support-local Torch reference for the fixed normalized radial Triweight kernel."""

import math

import torch

_SIGMA_LOW = 0.03
_SIGMA_HIGH = 3.25
_FLOOR = 1e-6


def apply(x, p, plan):
    """Bounded temporary site chunks; never construct an operator-sized W."""
    y = x.new_zeros((len(x), plan.sizes[0])) + p.reshape(-1)[:1].mul(0).sum()
    eps = torch.finfo(p.dtype).eps
    for atom in p:
        center = atom[2:]
        axes = []
        for c, o, s, n in zip(
            center.detach().tolist(), plan.origin, plan.spacing, plan.sizes
        ):
            margin = 32 * eps * (abs(c) + abs(o) + n * s + _SIGMA_HIGH + 1)
            lo = max(0, min(n, math.ceil((c - _SIGMA_HIGH - margin - o) / s)))
            hi = max(lo, min(n, math.floor((c + _SIGMA_HIGH + margin - o) / s) + 1))
            axes.append((lo, hi))
        lengths = tuple(hi - lo for lo, hi in axes)
        total = math.prod(lengths)
        if not total:
            continue
        # The CUDA contract retains the interior derivative at both endpoints.
        # Explicit choices avoid version-dependent scalar/tensor clamp tie VJPs.
        lower, upper = atom.new_tensor([math.log(_SIGMA_LOW), math.log(_SIGMA_HIGH)])
        log_width = torch.where(
            atom[1] < lower, lower, torch.where(atom[1] > upper, upper, atom[1])
        )
        precision = torch.exp(-2 * log_width)

        def chunk(
            start,
            total=total,
            lengths=lengths,
            axes=axes,
            center=center,
            precision=precision,
        ):
            index = torch.arange(start, min(start + 4096, total), device=p.device)
            r2 = index % lengths[2] + axes[2][0]
            r1 = index // lengths[2] % lengths[1] + axes[1][0]
            r0 = index // (lengths[1] * lengths[2]) + axes[0][0]
            d = [
                o + r.to(p.dtype) * s - c
                for o, r, s, c in zip(plan.origin, (r0, r1, r2), plan.spacing, center)
            ]
            q = ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]) * precision
            t = (1 - q).clamp_min(0)
            return r0, r1 * plan.sizes[2] + r2, t * t * t

        norm2 = atom.new_zeros(())
        for start in range(0, total, 4096):
            k = chunk(start)[2]
            norm2 = norm2 + (k * k).sum()
        # Safe zero-support derivative, retaining the inclusive floor tie.
        norm = norm2.clamp_min(1e-30).sqrt().clamp_min(_FLOOR)
        for start in range(0, total, 4096):
            rows, cols, k = chunk(start)
            values = x[:, cols] * (atom[0] * k / norm)
            y = y.index_add(1, rows, values)
    return y + x.reshape(-1)[:1].mul(0).sum()
