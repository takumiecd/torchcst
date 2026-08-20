"""How much moving an atom moves the represented map.

Tensors in, tensors out: this module knows of no store, no module and no
engine, the contract ``compute/backends`` already keeps.  What lives here is
the Gauss-Newton metric a coordinate step is measured in,

    J^2_s = w^2 . ||k_out||^2 . sum_i ||d k_in / d s_i||^2

with each norm optionally taken under the traffic the site actually carries
instead of under the identity.  Frobenius is the metric of a customer base
arriving equally from every direction, which belongs to no layer that exists;
E-func measured what the assumption costs by rescoring a census under the real
traffic and watching conv1's error fall from 0.36 to 0.16.
"""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def radial_moment(k: Tensor, mu: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    """``(sum_n k[n,i]^2 ||mu_n - x_i||^2, sum_n k[n,i]^2)``, no ``[N, K, d]`` cube."""
    k_sq = k.square()
    mass = k_sq.sum(0)
    mu_sq = mu.square().sum(1)
    cross = k_sq.transpose(0, 1) @ mu  # [K, d]
    moment = (
        (k_sq * mu_sq[:, None]).sum(0)
        - 2.0 * (cross * x).sum(1)
        + mass * x.square().sum(1)
    )
    return moment, mass


def columns(kernel, mu: Tensor, centers: Tensor) -> tuple[Tensor, Tensor]:
    """``(kappa, d kappa / d c per unit displacement)`` for one side.

    The second return is ``2 * profile_grad``, the factor satisfying
    ``d kappa / d c = factor * (c - x)``; squaring it against the radial moment
    is what makes the metric family-generic.
    """
    sigma = kernel.sigma.detach().to(centers)
    squared_distance = torch.cdist(mu, centers).square()
    return (
        kernel.profile(squared_distance, sigma),
        2.0 * kernel.profile_grad(squared_distance, sigma),
    )


def weighted(matrix: Tensor, traffic: Tensor | None) -> Tensor:
    """``diag(k^T Sigma k)`` per atom -- the column norm the traffic sees.

    ``None`` is Sigma = identity, which is the Frobenius norm and the metric
    this arc used before anyone asked what the identity was claiming.
    """
    if traffic is None:
        return matrix.square().sum(0)
    rows = traffic.to(device=matrix.device, dtype=matrix.dtype)
    return (rows @ matrix).square().sum(0) / rows.shape[0]


def directional(factor: Tensor, mu: Tensor, centers: Tensor,
                traffic: Tensor | None) -> Tensor:
    """``sum_i || Sigma^(1/2) d kappa / d c_i ||^2`` per atom.

    One chart axis at a time, so the ``[N, K, d]`` displacement cube is never
    built; each axis costs one ``[N, K]`` temporary, the shape the kernel
    matrix already occupies.
    """
    total = torch.zeros(centers.shape[0], device=centers.device,
                        dtype=centers.dtype)
    for axis in range(centers.shape[1]):
        derivative = factor * (centers[:, axis][None, :] - mu[:, axis][:, None])
        total = total + weighted(derivative, traffic)
    return total


def jacobian_sq(*, k_in: Tensor, g_in: Tensor, k_out: Tensor, g_out: Tensor,
                mu_in: Tensor, mu_out: Tensor, source: Tensor, target: Tensor,
                w_sq: Tensor, traffic_in: Tensor | None = None,
                traffic_out: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """``(J^2_s, J^2_t)`` for one block of atoms.

    With no traffic this is the rule lm1d froze.  With it, both sides are
    weighted -- the Gauss-Newton metric factorises the way K-FAC's does, the
    input side by the activations arriving and the output side by the gradients
    leaving, and weighting one alone would be half a correction.
    """
    if traffic_in is None and traffic_out is None:
        r_in, _ = radial_moment(g_in, mu_in, source)
        r_out, _ = radial_moment(g_out, mu_out, target)
        return (w_sq * k_out.square().sum(0) * r_in,
                w_sq * k_in.square().sum(0) * r_out)
    return (
        w_sq * weighted(k_out, traffic_out)
        * directional(g_in, mu_in, source, traffic_in),
        w_sq * weighted(k_in, traffic_in)
        * directional(g_out, mu_out, target, traffic_out),
    )
