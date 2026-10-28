"""Immutable radius hierarchy metadata, with uncapped local periods."""

import math
from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class BankPlan:
    level: int
    tile_sites: int
    tile_count: int
    period: float
    node_offset: int
    window_tiles: int

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.tile_sites, self.tile_count)):
            raise ValueError("bank tile sizes and counts must be positive integers")
        if (
            type(self.level) is not int
            or self.level < 0
            or type(self.node_offset) is not int
            or self.node_offset < 0
        ):
            raise ValueError("bank level and offset must be nonnegative integers")
        if type(self.window_tiles) is not int or self.window_tiles not in (2, 3):
            raise ValueError("radius banks require K=2 or K=3")
        if (
            type(self.period) not in (int, float)
            or not math.isfinite(self.period)
            or self.period <= 0
        ):
            raise ValueError("bank period must be finite and positive")

    @property
    def k(self):
        return self.window_tiles


@dataclass(frozen=True, kw_only=True)
class AxisPlan:
    site_count: int
    base_tile_sites: int
    spacing: float = 1.0
    origin: float = 0.0
    modes: int = 8
    merge_factor: int = 2
    polar: tuple[float, ...] | None = None

    def __post_init__(self):
        if any(
            type(v) is not int or v < 1 for v in (self.site_count, self.base_tile_sites)
        ):
            raise ValueError("site_count and base_tile_sites must be positive integers")
        if self.site_count > 128 or self.site_count % self.base_tile_sites:
            raise ValueError("base_tile_sites must divide site_count <= 128")
        if type(self.merge_factor) is not int or self.merge_factor < 2:
            raise ValueError("merge_factor must be an integer at least two")
        ratio = self.site_count // self.base_tile_sites
        while ratio > 1 and ratio % self.merge_factor == 0:
            ratio //= self.merge_factor
        if ratio != 1:
            raise ValueError("chart/base ratio must be a power of merge_factor")
        if type(self.modes) is not int or not 1 <= self.modes <= 16:
            raise ValueError("modes must lie in [1,16], including odd degrees")
        if (
            any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (self.spacing, self.origin)
            )
            or self.spacing <= 0
        ):
            raise ValueError("spacing must be positive and origin finite")
        if not math.isfinite(3 * self.period) or self.period <= 0:
            raise ValueError("chart and padded local periods must be representable")
        if self.polar is not None:
            if (
                type(self.polar) is not tuple
                or len(self.polar) != 9
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                    for v in self.polar
                )
            ):
                raise ValueError("invalid Polar radius metadata")
            _, _, kappa, _, power, minimum, birth, maximum, floor = self.polar
            if (
                kappa <= 1
                or power > 1
                or not minimum <= birth <= maximum <= self.period / 2
                or not minimum <= floor <= maximum
            ):
                raise ValueError("Polar radius bounds must fit the chart half-period")
        if any(
            not math.isfinite(2 * math.pi * self.modes / bank.period)
            for bank in self.banks
        ):
            raise ValueError("bank frequencies must be representable")

    @property
    def period(self):
        return self.site_count * self.spacing

    @property
    def basis_count(self):
        return 1 + 2 * self.modes

    @cached_property
    def banks(self):
        # AxisPlan is frozen: these banks depend only on its immutable fields.
        # Validation warms this once; normal execution must not rebuild every
        # BankPlan when asking for node counts or compact-buffer boundaries.
        banks = []
        tile_sites, level, offset = self.base_tile_sites, 0, 0
        while tile_sites <= self.site_count:
            count = self.site_count // tile_sites
            for k in (2, 3):
                banks.append(
                    BankPlan(
                        level,
                        tile_sites,
                        count,
                        k * tile_sites * self.spacing,
                        offset,
                        k,
                    )
                )
                offset += count
            tile_sites *= self.merge_factor
            level += 1
        return tuple(banks)

    @property
    def node_count(self):
        last = self.banks[-1]
        return last.node_offset + last.tile_count

    def prefix_nodes(self, direct_levels):
        """Number of low-level nodes omitted from a compact moment/C buffer."""
        if (
            type(direct_levels) is not int
            or not 0 <= direct_levels <= len(self.banks) // 2
        ):
            raise ValueError("direct levels must lie between zero and level count")
        return sum(b.tile_count for b in self.banks if b.level < direct_levels)

    def active_nodes(self, direct_levels):
        return self.node_count - self.prefix_nodes(direct_levels)

    def native_config(self):
        return [
            self.site_count,
            self.spacing,
            self.base_tile_sites,
            self.origin,
            self.modes,
            self.merge_factor,
        ] + ([] if self.polar is None else list(self.polar))


@dataclass(frozen=True, kw_only=True)
class HierarchyPlan:
    input: AxisPlan
    output: AxisPlan

    def __post_init__(self):
        if type(self.input) is not AxisPlan or type(self.output) is not AxisPlan:
            raise TypeError("input/output must be radius AxisPlan instances")
        if (self.input.polar is None) != (self.output.polar is None) or (
            self.input.polar is not None
            and self.input.polar[:5] != self.output.polar[:5]
        ):
            raise ValueError(
                "input/output must share the Polar amplitude and activity law"
            )
