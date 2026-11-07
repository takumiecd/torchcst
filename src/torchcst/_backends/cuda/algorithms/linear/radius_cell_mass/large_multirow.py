"""Explicit square, linear-Atom-count research cohort; no public selection."""

from dataclasses import dataclass

import torch

from torchcst._backends.schema import ExecutionPlan, SupportResult
from torchcst._builtin_callbacks import seal_functions as _seal_functions

from .algorithm import RadiusCellMassAlgorithm, hierarchy_from_operator
from .packed import (
    PACKED_RECOMPUTE,
    RadiusPackedParallelAlgorithm,
    RadiusPackedRecomputeAlgorithm,
)
from .plans import GLOBAL_ACTIVE_GRAD4_NATIVE
from .row8 import RadiusRow8Algorithm


class _LargeSquareScope:
    def _allows_large_cohort(self, context, recipe):
        return True  # supports() below checks this research cohort explicitly.

    def supports(self, context, recipe):
        result = RadiusCellMassAlgorithm.supports(self, context, recipe)
        if result.reasons:
            return result
        plan = hierarchy_from_operator(context.operator)
        n = plan.input.site_count
        reasons = []
        if (
            n not in (2048, 8192, 16384)
            or plan.output.site_count != n
            or not 4 <= context.atom_count <= 8 * n
            or not 1 <= context.m <= 256
            or context.dtype != torch.float32
        ):
            reasons.append(
                "large multirow requires square N2048/8192/16384, A4..8N, R1..256, FP32"
            )
        if any(
            a.polar is None
            or a.modes != 8
            or a.base_tile_sites != 1
            or a.merge_factor != 2
            or a.spacing != 1
            or a.origin != 0
            or a.polar[0] > 1
            or a.polar[5] < 0.5
            or a.polar[7] > 50
            for a in (plan.input, plan.output)
        ):
            reasons.append(
                "large multirow requires origin0/spacing1, Polar M8/base1/merge2/radius[.5,50]"
            )
        return SupportResult(tuple(reasons))


@dataclass(frozen=True)
class RadiusLargeRow8Algorithm(_LargeSquareScope, RadiusRow8Algorithm):
    id: str = "cuda_radius_large_multirow_row8"


@dataclass(frozen=True)
class RadiusLargePackedAlgorithm(_LargeSquareScope, RadiusPackedParallelAlgorithm):
    id: str = "cuda_radius_large_multirow_fp32"


@dataclass(frozen=True)
class RadiusLargeRecomputeAlgorithm(_LargeSquareScope, RadiusPackedRecomputeAlgorithm):
    id: str = "cuda_radius_large_multirow_recompute"


@dataclass(frozen=True)
class RadiusLargeGeometryAlgorithm(RadiusLargePackedAlgorithm):
    id: str = "cuda_radius_large_multirow_geometry"

    @property
    def cache_geometry(self):
        return True


LARGE_GEOMETRY = ExecutionPlan(
    RadiusLargeGeometryAlgorithm().id, "v1", GLOBAL_ACTIVE_GRAD4_NATIVE.recipe
)


LARGE_RECOMPUTE = ExecutionPlan(
    RadiusLargeRecomputeAlgorithm().id, "v1", PACKED_RECOMPUTE.recipe
)

LARGE_ROW8 = ExecutionPlan(
    RadiusLargeRow8Algorithm().id, "v1", GLOBAL_ACTIVE_GRAD4_NATIVE.recipe
)
LARGE_PACKED = ExecutionPlan(
    RadiusLargePackedAlgorithm().id, "v1", GLOBAL_ACTIVE_GRAD4_NATIVE.recipe
)


_seal_functions(globals())
