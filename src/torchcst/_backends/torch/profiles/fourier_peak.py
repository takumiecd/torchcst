"""FP64 shape-only reference: coefficients depend solely on q=I/L and M."""

import math
from functools import cache

import torch

from torchcst.kernels.profiles.fourier_peak import FourierPeakSpec


@cache
@torch.inference_mode(False)
def _projection(modes, device):
    k = torch.arange(modes + 1, dtype=torch.float64, device=device)
    zeros = 2 if modes == 4 else 3
    a = torch.stack(
        [(-1.0) ** k * (k / modes) ** (2 * j) for j in range(zeros)]
        + [torch.ones_like(k)]
    )
    inv_metric = torch.diag(torch.where(k == 0, 0.5, 1.0).double())
    right = torch.linalg.solve(a @ inv_metric @ a.T, a @ inv_metric).T
    return torch.eye(modes + 1, dtype=k.dtype, device=device) - right @ a, right[:, -1]


def coefficients(spec, q):
    """Return b(q) and db/dq, with final axis k=0..M (cosine convention).

    Analytic transform of (1-u²)^4, with a stable small-argument series,
    followed by the fixed continuous-L2 endpoint/peak projection.
    """
    if type(spec) is not FourierPeakSpec or spec.revision != 1:
        raise ValueError("unsupported Fourier peak profile or revision")
    if not q.is_floating_point() or not bool(
        torch.isfinite(q).all() & (q > 0).all() & (q < 1).all()
    ):
        raise ValueError("q=I/L must be finite and in (0, 1)")
    q = q.double()
    k = torch.arange(spec.modes + 1, dtype=q.dtype, device=q.device)
    z = math.pi * q[..., None] * k
    small = z.abs() < 4
    t = torch.where(small, z, 0.0)
    term = torch.full_like(t, 256 / 315)
    integral, zprime = term, torch.zeros_like(term)
    for n in range(20):
        term = term * (-t.square() / ((2 * n + 2) * (2 * n + 11)))
        integral = integral + term
        zprime = zprime + (2 * n + 2) * term
    v = torch.where(small, 4.0, z)
    v2 = v.square()
    large = (
        768
        * ((105 - 45 * v2 + v2.square()) * v.sin() - (105 * v - 10 * v * v2) * v.cos())
        / v**9
    )
    large_prime = (
        -768
        * (
            (945 - 420 * v2 + 15 * v2.square()) * v.sin()
            - (945 * v - 105 * v * v2 + v * v2.square()) * v.cos()
        )
        / v**9
    )
    integral = torch.where(small, integral, large)
    zprime = torch.where(small, zprime, large_prime)
    factor = torch.where(k == 0, 1.0, 2.0) / 2
    c = factor * q[..., None] * integral
    cq = factor * (integral + zprime)
    matrix, offset = _projection(spec.modes, q.device)
    return c @ matrix.T + offset, cq @ matrix.T


def values(spec, phase, q):
    """Unmasked periodic profile F(phase,q); phase=pi*(x-mu)/L."""
    b, _ = coefficients(spec, q)
    k = torch.arange(spec.modes + 1, device=phase.device, dtype=torch.float64)
    return (torch.cos(phase.double()[..., None] * k) * b).sum(-1)
