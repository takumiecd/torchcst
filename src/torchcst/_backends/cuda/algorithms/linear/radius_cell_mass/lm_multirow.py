"""Explicit FP32 LM projection cohorts; no public/default registration."""

from dataclasses import dataclass

import torch

from torchcst._backends.schema import ExecutionPlan, SupportResult
from torchcst._builtin_callbacks import seal_functions as _seal_functions

from .algorithm import RadiusCellMassAlgorithm, hierarchy_from_operator
from .packed import PACKED_RECOMPUTE, RadiusPackedParallelAlgorithm
from .plans import GLOBAL_ACTIVE_GRAD4_NATIVE


@dataclass(frozen=True)
class RadiusLMLevelsAlgorithm(RadiusPackedParallelAlgorithm):
    id: str = "cuda_radius_lm_multirow_levels"

    @property
    def cache_geometry(self):
        return True

    @property
    def level_moments(self):
        return True

    def _allows_large_cohort(self, context, recipe):
        return True  # The explicit shape/Atom cohort below remains mandatory.

    def supports(self, context, recipe):
        result = RadiusCellMassAlgorithm.supports(self, context, recipe)
        if result.reasons:
            return result
        plan = hierarchy_from_operator(context.operator)
        counts = {
            (256, 256): 2048,
            (256, 1024): 4096,
            (1024, 256): 4096,
            (512, 512): 4096,
            (512, 2048): 8192,
            (2048, 512): 8192,
        }
        reasons = []
        if (
            context.atom_count
            != counts.get((plan.input.site_count, plan.output.site_count))
            or not 1 <= context.m <= 256
            or context.dtype != torch.float32
        ):
            reasons.append(
                "LM multirow requires qualified N256/512 square or expansion4, c8, R1..256, FP32"
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
                "LM multirow requires origin0/spacing1, Polar M8/base1/merge2/radius[.5,50]"
            )
        return SupportResult(tuple(reasons))


@dataclass(frozen=True)
class RadiusLMLevelsStream64Algorithm(RadiusLMLevelsAlgorithm):
    id: str = "cuda_radius_lm_multirow_levels_stream64"

    @property
    def recompute(self):
        return True

    @property
    def stream_rows(self):
        return 64


@dataclass(frozen=True)
class RadiusLMPhysicalLevelsAlgorithm(RadiusLMLevelsAlgorithm):
    """Explicit FFN physical-radius cohort, with the full radius VJP."""

    id: str = "cuda_radius_lm_physical_levels"

    def supports(self, context, recipe):
        result = RadiusCellMassAlgorithm.supports(self, context, recipe)
        if result.reasons:
            return result
        plan = hierarchy_from_operator(context.operator)
        reasons = []
        if (
            (plan.input.site_count, plan.output.site_count)
            not in ((512, 2048), (2048, 512))
            or context.atom_count != 8192
            or not 1 <= context.m <= 256
            or context.dtype != torch.float32
        ):
            reasons.append(
                "physical LM requires width512/expansion4, c8, R1..256, FP32"
            )
        if any(
            a.polar is not None
            or a.modes != 8
            or a.base_tile_sites != 1
            or a.merge_factor != 2
            or a.spacing != 1
            or a.origin != 0
            for a in (plan.input, plan.output)
        ):
            reasons.append(
                "physical LM requires physical M8/base1/merge2/origin0/spacing1"
            )
        return SupportResult(tuple(reasons))


LM_LEVELS = ExecutionPlan(
    RadiusLMLevelsAlgorithm().id, "v1", GLOBAL_ACTIVE_GRAD4_NATIVE.recipe
)
LM_LEVELS_STREAM64 = ExecutionPlan(
    RadiusLMLevelsStream64Algorithm().id, "v1", PACKED_RECOMPUTE.recipe
)
LM_PHYSICAL_LEVELS = ExecutionPlan(
    RadiusLMPhysicalLevelsAlgorithm().id, "v1", GLOBAL_ACTIVE_GRAD4_NATIVE.recipe
)

_seal_functions(globals())
