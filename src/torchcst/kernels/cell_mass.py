"""Cell masses of a periodic density on a tile-aligned, one-period window.

This research contract approximates the small Fourier tails as zero outside the
selected tiles. It does not clip individual cells at an atom-centred support.
"""

import math
from dataclasses import dataclass, field

from .normalization import NormalizationSpec
from .profiles.fourier import ChebyshevSquareSpec


@dataclass(frozen=True, kw_only=True)
class CellMassAxisSpec:
    """A cell grid with an explicit tile-window approximation contract.

    Each tile contains tile_sites whole cells. For odd K=period_tiles, keep
    owner-K//2 .. owner+K//2, unwrapped around the atom centre. Their total width
    P=K*T is exactly the local Fourier period. All other tiles have zero mass.
    I=(K//2)*T is the bank's intended support half width; the extra tile supplies
    zero-approximated padding. The coefficient density's small tails inside the
    selected window are retained. Routing is piecewise constant in backward.

    The window/period belong to the approximation, not a launch tuning option.
    K=1 is allowed only when the sole tile covers the entire chart circle.
    """

    profile: ChebyshevSquareSpec
    site_count: int
    spacing: float
    tile_sites: int
    period_tiles: int = 3
    origin: float = 0.0

    def __post_init__(self):
        if type(self.profile) is not ChebyshevSquareSpec:
            raise TypeError("cell-mass prototype requires ChebyshevSquareSpec")
        for name in ("site_count", "tile_sites", "period_tiles"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.site_count % self.tile_sites:
            raise ValueError("tile_sites must divide site_count")
        if self.period_tiles % 2 != 1 or self.period_tiles > self.tile_count:
            raise ValueError("period_tiles must be odd and fit inside the chart")
        if self.period_tiles == 1 and self.tile_count != 1:
            raise ValueError("one-period tile requires a single chart tile")
        for name in ("spacing", "origin"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.spacing <= 0:
            raise ValueError("spacing must be positive")
        if not math.isfinite(self.period) or not math.isfinite(self.profile_period):
            raise ValueError("grid and profile periods must be finite")

    @property
    def tile_count(self):
        return self.site_count // self.tile_sites

    @property
    def tile_width(self):
        return self.tile_sites * self.spacing

    @property
    def support_half_width(self):
        """Intended bank support I; distinct from the Fourier half-period L."""
        return (self.period_tiles // 2) * self.tile_width

    @property
    def period(self):
        return self.site_count * self.spacing

    @property
    def profile_period(self):
        return self.period_tiles * self.tile_width


@dataclass(frozen=True, kw_only=True)
class CellMassLinearSpec:
    """Y[b,o] = sum_a amp[a] * u[a,o] * sum_i X[b,i] * v[a,i].

    Physical rows are (amplitude, sigma, output_center, input_center). Sigma is
    shared by both factors and its derivative includes both paths. The retained
    one-period window is the approximate profile's support for normalization.
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
