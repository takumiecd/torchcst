"""Support that changes mathematical weights, separate from launch settings."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class FourierTileSupportSpec:
    """Owner tile and its two neighbours; local profile period is 3*width.

    Tile origins coincide with chart origins. A centre exactly on a boundary
    belongs to the tile on its right. Each axis uses its own positive width.
    """

    tile_widths: tuple[float, ...]

    def __post_init__(self):
        if (
            not isinstance(self.tile_widths, tuple)
            or not self.tile_widths
            or any(
                type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                for x in self.tile_widths
            )
        ):
            raise ValueError("tile_widths must be a nonempty tuple of positive widths")
