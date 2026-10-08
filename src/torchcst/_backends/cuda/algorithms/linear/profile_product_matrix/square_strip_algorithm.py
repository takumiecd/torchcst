"""Explicit large-output Strip capabilities; previous plans retain their limits."""

from dataclasses import dataclass
from typing import ClassVar

from .algorithm_v4 import LargeSplitMatrixStripAlgorithm
from .large_algorithm import LargeMatrixStripAlgorithm, LargePreparationStripAlgorithm


@dataclass(frozen=True)
class SquareStripPreparationAlgorithm(LargePreparationStripAlgorithm):
    id: str = "research_square_strip_profile_product_prepared"
    max_output: ClassVar[int] = 8192


@dataclass(frozen=True)
class SquareStripMatrixAlgorithm(LargeMatrixStripAlgorithm):
    id: str = "research_square_strip_profile_product_matrix"
    max_output: ClassVar[int] = 8192


@dataclass(frozen=True)
class SquareStripSplitAlgorithm(LargeSplitMatrixStripAlgorithm):
    id: str = "research_square_strip_profile_product_matrix"
    max_output: ClassVar[int] = 8192
