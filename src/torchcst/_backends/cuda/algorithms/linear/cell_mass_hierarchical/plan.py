"""Pure metadata for hierarchical cell-mass preparation.

The profile is the existing ChebyshevSquare density, with its original sigma
and effective-radius mapping. This is not a public Linear KernelSpec.
"""

import math
from dataclasses import dataclass

from torchcst.kernels.profiles.fourier import ChebyshevSquareSpec


@dataclass(frozen=True)
class BankPlan:
    """One local three-tile bank, or the special full-chart bank."""

    level: int
    tile_sites: int
    tile_count: int
    period: float
    node_offset: int
    full_chart: bool = False


@dataclass(frozen=True, kw_only=True)
class AxisPlan:
    """One periodic site axis using the existing profile width parameter.

    min_tile_sites selects mathematical bank boundaries. It is not a CUDA
    block size. All bank moments use the same coordinate origin so matching
    parent/child frequencies can be added without phase conversion.
    """

    site_count: int
    min_tile_sites: int
    spacing: float = 1.0
    origin: float = 0.0
    modes: int = 8
    tail_bound: float = 1e-3
    # Optional existing Polar width law: W,w_c,kappa,lower_kappa,power,
    # sigma_min,sigma_birth,sigma_max,upper_floor. No per-atom storage.
    polar: tuple[float, ...] | None = None

    def __post_init__(self):
        for name in ("site_count", "min_tile_sites"):
            value = getattr(self, name)
            if type(value) is not int or value < 1 or value & (value - 1):
                raise ValueError(f"{name} must be a positive power of two")
        if self.site_count > 1024 or self.min_tile_sites > self.site_count:
            raise ValueError(
                "preparation supports min_tile_sites <= site_count <= 1024"
            )
        if type(self.modes) is not int or self.modes not in range(2, 17, 2):
            raise ValueError("modes must be even and lie in [2,16]")
        ChebyshevSquareSpec(modes=self.modes, tail_bound=self.tail_bound)
        for name in ("spacing", "origin"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.spacing <= 0:
            raise ValueError("spacing must be positive")
        if self.polar is not None:
            if (
                type(self.polar) is not tuple
                or len(self.polar) != 9
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                    for v in self.polar
                )
            ):
                raise ValueError("invalid Polar axis metadata")
            _w, _wc, kappa, _lower, power, minimum, birth, maximum, floor = self.polar
            if (
                kappa <= 1
                or power > 1
                or not minimum <= birth <= maximum <= self.period / 3
                or not minimum <= floor <= maximum
            ):
                raise ValueError("Polar axis width law does not fit this grid")
        if not math.isfinite(2 * self.period) or self.period <= 0:
            raise ValueError("chart period must be positive and representable")
        for bank in self.banks:
            if not math.isfinite(2 * math.pi * self.modes / bank.period):
                raise ValueError("bank frequencies must be representable")

    @property
    def period(self):
        return self.site_count * self.spacing

    @property
    def min_tile_width(self):
        return self.min_tile_sites * self.spacing

    @property
    def full_level(self):
        return (self.site_count // self.min_tile_sites).bit_length() - 1

    @property
    def basis_count(self):
        return 1 + 2 * self.modes

    @property
    def profile(self):
        return ChebyshevSquareSpec(modes=self.modes, tail_bound=self.tail_bound)

    @property
    def banks(self):
        banks = []
        tile_sites = self.min_tile_sites
        offset = 0
        level = 0
        while 3 * tile_sites <= self.site_count:
            count = self.site_count // tile_sites
            banks.append(
                BankPlan(
                    level, tile_sites, count, 3 * tile_sites * self.spacing, offset
                )
            )
            offset += count
            tile_sites *= 2
            level += 1
        banks.append(
            BankPlan(self.full_level, self.site_count, 1, self.period, offset, True)
        )
        return tuple(banks)

    @property
    def node_count(self):
        last = self.banks[-1]
        return last.node_offset + last.tile_count

    def native_config(self):
        return [
            self.site_count,
            self.spacing,
            self.min_tile_sites,
            self.origin,
            self.modes,
            self.tail_bound,
        ] + ([] if self.polar is None else list(self.polar))


@dataclass(frozen=True, kw_only=True)
class HierarchyPlan:
    """Input/output preparation geometry; contains no atom values or buffers."""

    input: AxisPlan
    output: AxisPlan

    def __post_init__(self):
        if type(self.input) is not AxisPlan or type(self.output) is not AxisPlan:
            raise TypeError("input and output must be AxisPlan")
        if (self.input.polar is None) != (self.output.polar is None) or (
            self.input.polar is not None
            and self.input.polar[:5] != self.output.polar[:5]
        ):
            raise ValueError(
                "input/output must share their Polar amplitude and activity law"
            )


def classify_axis(sigma, center, axis):
    """Readable scalar definition returning (bank level, canonical owner).

    The GPU computes this independently for each atom. The scalar function is
    useful for inspecting a plan without allocating tensors or a CUDA runtime.
    """
    if not math.isfinite(sigma) or not 0 < sigma <= axis.period / 3:
        raise ValueError("sigma must lie in (0, chart period/3]")
    if not math.isfinite(center):
        raise ValueError("center must be finite")
    tile_sites = axis.min_tile_sites
    level = 0
    while sigma > tile_sites * axis.spacing:
        tile_sites *= 2
        level += 1
    if 3 * tile_sites > axis.site_count:
        return axis.full_level, 0
    # Separate remainders avoid overflowing center-origin for finite inputs.
    relative = (
        math.fmod(center, axis.period) - math.fmod(axis.origin, axis.period)
    ) % axis.period
    owner = math.floor((relative + axis.spacing / 2) / (tile_sites * axis.spacing))
    return level, owner % (axis.site_count // tile_sites)
