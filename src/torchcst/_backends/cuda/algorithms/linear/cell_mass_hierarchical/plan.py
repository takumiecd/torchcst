"""Pure metadata for hierarchical cell-mass preparation.

This is an execution plan for classification and Fourier moments, not a new
profile or a public Linear KernelSpec. The density used by H/C is still a
separate mathematical contract. Importing this file never imports CUDA/Torch.
"""

import math
from dataclasses import dataclass


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
    """One periodic site axis; sigma means full concentration width.

    min_tile_sites selects mathematical bank boundaries. It is not a CUDA
    block size. All bank moments use the same coordinate origin so matching
    parent/child frequencies can be added without phase conversion.
    """

    site_count: int
    min_tile_sites: int
    spacing: float = 1.0
    origin: float = 0.0
    modes: int = 8

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
        for name in ("spacing", "origin"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.spacing <= 0:
            raise ValueError("spacing must be positive")
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
        ]


@dataclass(frozen=True, kw_only=True)
class HierarchyPlan:
    """Input/output preparation geometry; contains no atom values or buffers."""

    input: AxisPlan
    output: AxisPlan

    def __post_init__(self):
        if type(self.input) is not AxisPlan or type(self.output) is not AxisPlan:
            raise TypeError("input and output must be AxisPlan")


def classify_axis(sigma, center, axis):
    """Readable scalar definition returning (bank level, canonical owner).

    The GPU computes this independently for each atom. The scalar function is
    useful for inspecting a plan without allocating tensors or a CUDA runtime.
    """
    if not math.isfinite(sigma) or not 0 < sigma < axis.period:
        raise ValueError("sigma must lie in (0, chart period)")
    if not math.isfinite(center):
        raise ValueError("center must be finite")
    half_width = sigma / 2
    tile_sites = axis.min_tile_sites
    level = 0
    while half_width > tile_sites * axis.spacing:
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
