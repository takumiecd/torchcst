"""Declarations of atom parameter meanings."""

from .activity_width import DirectAmpWidthSpec, PolarAmpWidthSpec
from .amp_width import AmpWidthSpec
from .base import BandwidthBounds, FixedWidthSpec, ParameterizationSpec

__all__ = [
    "AmpWidthSpec",
    "BandwidthBounds",
    "DirectAmpWidthSpec",
    "FixedWidthSpec",
    "ParameterizationSpec",
    "PolarAmpWidthSpec",
]
