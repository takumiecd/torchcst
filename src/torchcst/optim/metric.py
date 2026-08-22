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


def normalized_columns(kernel, mu: Tensor, centers: Tensor) -> tuple[Tensor, Tensor]:
    """Unit columns and pre-projection coordinate-derivative factors.

    The kernel supplies a profile and ``d profile / d squared_distance`` under
    one arbitrary positive scale per column.  Dividing both by the profile
    norm removes that scale.  :func:`projected_directional` then removes the
    radial derivative of whichever scale the family chose, leaving the exact
    derivative of the delivered L2-normalised column.

    An honestly zero compact-support column returns a zero column and factor;
    it therefore contributes zero metric rather than ``NaN``.
    """
    sigma = kernel.sigma.detach().to(centers)
    squared_distance = torch.cdist(mu, centers).square()
    value, profile_grad = kernel.scaled_profile_pair(squared_distance, sigma)
    norm = torch.linalg.vector_norm(value, dim=0, keepdim=True)
    divisor = norm.clamp_min(torch.finfo(value.dtype).tiny)
    return value / divisor, 2.0 * profile_grad / divisor


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


def projected_directional(
    unit: Tensor,
    factor: Tensor,
    mu: Tensor,
    centers: Tensor,
    traffic: Tensor | None,
    *,
    per_axis: bool = False,
) -> Tensor:
    """Squared derivatives of an L2-normalised column after tangent projection.

    ``factor * (center - mu)`` is the scaled raw derivative divided by the
    scaled column norm.  Removing its component parallel to ``unit`` applies
    ``I - unit unit^T`` and makes the result independent of the arbitrary
    positive scale permitted by the kernel contract.
    """
    axes: list[Tensor] = []
    for axis in range(centers.shape[1]):
        derivative = factor * (
            centers[:, axis][None, :] - mu[:, axis][:, None]
        )
        radial = (unit * derivative).sum(0, keepdim=True)
        tangent = derivative - unit * radial
        axes.append(weighted(tangent, traffic).clamp_min(0))
    stacked = torch.stack(axes, dim=1)
    return stacked if per_axis else stacked.sum(1)


def projected_directional_gram(
    unit: Tensor,
    factor: Tensor,
    mu: Tensor,
    centers: Tensor,
    traffic: Tensor | None,
) -> Tensor:
    """Per-atom Gram of the projected axis derivatives, ``[K, d, d]``.

    The diagonal reproduces :func:`projected_directional` with
    ``per_axis=True``; the off-diagonal entries are the within-atom axis
    couplings the diagonal metric ignores.  On an irregular neuron cloud the
    parity argument that kills them in the continuum does not apply, and they
    carry most of the off-diagonal energy of ``J.T @ J``.
    """
    tangents: list[Tensor] = []
    for axis in range(centers.shape[1]):
        derivative = factor * (
            centers[:, axis][None, :] - mu[:, axis][:, None]
        )
        radial = (unit * derivative).sum(0, keepdim=True)
        tangent = derivative - unit * radial
        if traffic is not None:
            rows = traffic.to(device=tangent.device, dtype=tangent.dtype)
            tangent = (rows @ tangent) / float(rows.shape[0]) ** 0.5
        tangents.append(tangent)
    stacked = torch.stack(tangents, dim=-1)
    gram = torch.einsum("nka,nkb->kab", stacked, stacked)
    gram.diagonal(dim1=-2, dim2=-1).clamp_min_(0)
    return gram


def gauged_jacobian_gram(
    *,
    unit_in: Tensor,
    factor_in: Tensor,
    unit_out: Tensor,
    factor_out: Tensor,
    mu_in: Tensor,
    mu_out: Tensor,
    source: Tensor,
    target: Tensor,
    mass_sq: Tensor,
    traffic_in: Tensor | None = None,
    traffic_out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Within-atom source/target metric blocks after L2 column normalisation.

    Same gauge as :func:`gauged_jacobian_sq`, but each side keeps the full
    ``d x d`` axis Gram instead of only its diagonal.  Normalisation makes the
    amplitude row and the source-target cross block exactly zero, so these two
    small blocks are the complete within-atom metric; cross-atom coupling is
    still neglected.
    """
    source_gram = projected_directional_gram(
        unit_in, factor_in, mu_in, source, traffic_in
    )
    target_gram = projected_directional_gram(
        unit_out, factor_out, mu_out, target, traffic_out
    )
    out_mass = weighted(unit_out, traffic_out)
    in_mass = weighted(unit_in, traffic_in)
    return (
        (mass_sq * out_mass)[:, None, None] * source_gram,
        (mass_sq * in_mass)[:, None, None] * target_gram,
    )


def gauged_jacobian_sq(
    *,
    unit_in: Tensor,
    factor_in: Tensor,
    unit_out: Tensor,
    factor_out: Tensor,
    mu_in: Tensor,
    mu_out: Tensor,
    source: Tensor,
    target: Tensor,
    mass_sq: Tensor,
    traffic_in: Tensor | None = None,
    traffic_out: Tensor | None = None,
    per_axis: bool = False,
) -> tuple[Tensor, Tensor]:
    """Diagonal Frobenius pullback metric after L2 column normalisation.

    Normalisation exactly decouples an atom's amplitude, source, and target
    blocks.  This function keeps the diagonal within each source/target block
    and ignores cross-atom coupling: a block-Jacobi diagonal approximation,
    not a claim that the full ``J.T @ J`` is diagonal.
    """
    source_diag = projected_directional(
        unit_in, factor_in, mu_in, source, traffic_in, per_axis=per_axis
    )
    target_diag = projected_directional(
        unit_out, factor_out, mu_out, target, traffic_out, per_axis=per_axis
    )
    out_mass = weighted(unit_out, traffic_out)
    in_mass = weighted(unit_in, traffic_in)
    if per_axis:
        out_mass = out_mass[:, None]
        in_mass = in_mass[:, None]
        mass_sq = mass_sq[:, None]
    return mass_sq * out_mass * source_diag, mass_sq * in_mass * target_diag


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
