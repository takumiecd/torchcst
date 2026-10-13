"""Revision-4 Fourier product execution, with lazy GPU kernel imports."""

from .algorithm import FourierTilesAlgorithm
from .recipe import FourierTilesRecipe

__all__ = ["FourierTilesAlgorithm", "FourierTilesRecipe"]
