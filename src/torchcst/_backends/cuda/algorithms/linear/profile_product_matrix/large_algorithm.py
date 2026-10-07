"""Explicit larger-chart research plans using unchanged exact executors."""

from dataclasses import dataclass
from typing import ClassVar

from torchcst._backends.schema import SupportResult
from torchcst.operators.context import LinearContext

from ..profile_product_global.algorithm_v3 import (
    PreparationProductAlgorithm,
    PreparationStripAlgorithm,
)
from .algorithm_v3 import GroupedMatrixProductAlgorithm, GroupedMatrixStripAlgorithm


@dataclass(frozen=True)
class LargeMatrixProductAlgorithm(GroupedMatrixProductAlgorithm):
    id: str = "research_profile_product_large_matrix"
    revision: str = "v1"
    max_sites: ClassVar[int] = 8192
    max_atoms: ClassVar[int] = 4194304


@dataclass(frozen=True)
class LargePreparationProductAlgorithm(PreparationProductAlgorithm):
    id: str = "research_profile_product_large_prepared"
    revision: str = "v1"
    max_sites: ClassVar[int] = 8192
    max_atoms: ClassVar[int] = 4194304


class _LargeStripSupport:
    max_input: ClassVar[int] = 8192

    def supports(self, context, recipe):
        result = super().supports(context, recipe)
        if isinstance(context, LinearContext) and context.atom_count > 4194304:
            return SupportResult(result.reasons + ("requires at most4194304 atoms",))
        return result


@dataclass(frozen=True)
class LargeMatrixStripAlgorithm(_LargeStripSupport, GroupedMatrixStripAlgorithm):
    id: str = "research_strip_profile_product_large_matrix"
    revision: str = "v1"


@dataclass(frozen=True)
class LargePreparationStripAlgorithm(_LargeStripSupport, PreparationStripAlgorithm):
    id: str = "research_strip_profile_product_large_prepared"
    revision: str = "v1"
