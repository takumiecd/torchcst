"""Amplitude gauges: what the number stored beside an atom's coordinates means.

``W = K_out diag(w) K_in^T`` fixes the *product* of an atom's amplitude and the
two columns it casts, not the split between them.  A gauge is that split, and
picking one is picking which quantity the optimizer is allowed to move.

Under :class:`Amplitude` -- the classical split, and the default -- the stored
number is the amplitude and the columns arrive as the kernel casts them.  An
atom's actual effect on ``W`` is then ``w * ||k_in|| * ||k_out||``, which the
atom can shrink two ways: honestly, by shrinking ``w`` where rent and prune
can see it, or quietly, by walking somewhere the columns are small.  The
second way is a back door, and it is not a small one.  Splitting the position
gradient of a one-atom fit by parity in ``w`` separates the two terms exactly,

    dL/ds = -w c'(s)          value alignment, odd in w
            + w^2 n(s) n'(s)  norm escape, even in w

and the escape half runs to three quarters of the alignment half, pointing
outward regardless of whether the atom is any use where it stands.

Under :class:`UnitFootprint` the stored number is the atom's Frobenius mass in
``W`` -- the same quantity rent already prices -- because each chart's column
is delivered at unit norm.  The escape term is then not suppressed but
algebraically absent: it equals ``w^2 . d||u||^2/ds`` and ``||u||`` is one by
construction.  Measured over four chart spacings and three offsets, the even
half falls from order one to machine epsilon.

Normalising by the column's *sum* rather than its norm does not do this.  The
loss is a squared error, so the quantity whose drift creates the escape term
is the L2 norm specifically; a column normalised to sum to one still has a
moving L2 norm, and the surviving escape term is in places larger than the
one it replaced.  What to hold fixed is not a matter of taste -- the loss
picks it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .kernels import ContinuousKernel


@dataclass(frozen=True)
class Amplitude:
    """Store the amplitude; deliver the kernel's columns untouched."""

    def columns(
        self,
        kernel: ContinuousKernel,
        query: Tensor,
        centers: Tensor,
        extras: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        return kernel(query, centers, extras)


@dataclass(frozen=True)
class UnitFootprint:
    """Store the atom's mass in ``W``; deliver each chart's column at unit norm.

    The amplitude a backend multiplies is then the Frobenius norm of the
    atom's rank-one contribution, so the optimizer moves the quantity the
    economy charges for.  Today those are different currencies: policy prices
    ``mass``, the optimizer steps ``w``, and the exchange rate between them is
    a function of where the atom is standing.

    Columns come from :meth:`ContinuousKernel.scaled_columns` rather than from
    the kernel directly, because normalising cannot rescue a column that has
    already underflowed -- which, in the reduced-precision paths, happens at
    ordinary distances.  A family with compact support may still hand back an
    honestly zero column for an atom outside every neuron's support; that one
    is clamped, leaving the atom contributing nothing, exactly as it does
    under :class:`Amplitude`.
    """

    def columns(
        self,
        kernel: ContinuousKernel,
        query: Tensor,
        centers: Tensor,
        extras: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        scaled = kernel.scaled_columns(query, centers, extras)
        norm = torch.linalg.vector_norm(scaled, dim=0, keepdim=True)
        return scaled / norm.clamp_min(torch.finfo(scaled.dtype).tiny)


#: What a compute module may be given as its amplitude gauge.
AmplitudeGauge = Amplitude | UnitFootprint


def require_gauge(value: object, name: str) -> AmplitudeGauge:
    if not isinstance(value, (Amplitude, UnitFootprint)):
        raise TypeError(f"{name} must be an Amplitude or UnitFootprint gauge")
    return value
