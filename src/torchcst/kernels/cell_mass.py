"""Explicit cell-mass mathematics for the first one-dimensional prototype.

These declarations do not select a backend or reinterpret a sampled-L2 plan.
The first supported density family is the existing nonnegative Chebyshev square.
The support period is a mathematical parameter, not a CUDA tile dimension.
"""

import math
from dataclasses import dataclass, field

from .normalization import NormalizationSpec
from .profiles.fourier import ChebyshevSquareSpec


@dataclass(frozen=True, kw_only=True)
class CellMassAxisSpec:
    """A circular grid of cells and a compact, periodized density.

    Site i owns [origin + (i-.5)*spacing, origin + (i+.5)*spacing].
    One native period is retained on [-support_half_width, +support_half_width]
    around the atom center and then truly periodized onto the grid circle.
    Sigma controls the existing native shape within that fixed support.
    """

    profile: ChebyshevSquareSpec
    site_count: int
    spacing: float
    support_half_width: float
    origin: float = 0.0

    def __post_init__(self):
        if type(self.profile) is not ChebyshevSquareSpec:
            raise TypeError("cell-mass prototype requires ChebyshevSquareSpec")
        if type(self.site_count) is not int or self.site_count < 1:
            raise ValueError("site_count must be a positive integer")
        for name in ("spacing", "support_half_width", "origin"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.spacing <= 0 or self.support_half_width <= 0:
            raise ValueError("spacing and support_half_width must be positive")
        if not math.isfinite(self.period) or not math.isfinite(self.profile_period):
            raise ValueError("grid and profile periods must be finite")

    @property
    def period(self):
        return self.site_count * self.spacing

    @property
    def profile_period(self):
        return 2 * self.support_half_width


@dataclass(frozen=True, kw_only=True)
class CellMassLinearSpec:
    """Y[b,o] = sum_a amp[a] * u[a,o] * sum_i X[b,i] * v[a,i].

    The research numeric entry points accept physical atom rows ordered
    (amplitude, sigma, output_center, input_center). They are not Polar rows.
    Sigma is shared by the two factors; its mathematical derivative includes
    both factors. No optimizer/update policy is selected by this declaration.
    """

    input: CellMassAxisSpec
    output: CellMassAxisSpec
    normalization: NormalizationSpec = field(
        default_factory=lambda: NormalizationSpec(
            kind="continuous_mass", domain="profile_support"
        )
    )

    def __post_init__(self):
        if not isinstance(self.input, CellMassAxisSpec) or not isinstance(
            self.output, CellMassAxisSpec
        ):
            raise TypeError("cell-mass linear requires two CellMassAxisSpec values")
        if self.normalization != NormalizationSpec(
            kind="continuous_mass", domain="profile_support"
        ):
            raise ValueError("cell-mass linear requires continuous profile mass")
