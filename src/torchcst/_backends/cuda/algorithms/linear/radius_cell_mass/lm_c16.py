"""Explicit c16 M8 FFN cohort; reuse the unchanged c8 arithmetic/executor."""

from dataclasses import dataclass, replace
from functools import cache

import torch

from torchcst._backends.schema import DefaultRecipe, ExecutionPlan, SupportResult
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import LinearBinding, linear_execution

from .algorithm import RadiusCellMassAlgorithm, hierarchy_from_operator
from .lm_multirow import RadiusLMPhysicalLevelsAlgorithm
from .plans import GLOBAL_ACTIVE_GRAD4_NATIVE
from .radius_envelope import (
    RadiusEnvelopeWideAmplitudeLMAlgorithm,
    physical_operator,
)


@dataclass(frozen=True)
class RadiusLMPhysicalC16LevelsAlgorithm(RadiusLMPhysicalLevelsAlgorithm):
    id: str = "cuda_radius_lm_physical_c16_levels"

    def supports(self, context, recipe):
        result = RadiusCellMassAlgorithm.supports(self, context, recipe)
        if result.reasons:
            return result
        plan = hierarchy_from_operator(context.operator)
        if (
            (plan.input.site_count, plan.output.site_count)
            not in ((512, 2048), (2048, 512))
            or context.atom_count != 16384
            or not 1 <= context.m <= 256
            or context.dtype != torch.float32
            or any(
                a.polar is not None
                or a.modes != 8
                or a.base_tile_sites != 1
                or a.merge_factor != 2
                or a.spacing != 1
                or a.origin != 0
                for a in (plan.input, plan.output)
            )
        ):
            reason = (
                "c16 physical LM requires width512/expansion4, A16384, R1..256, "
                "FP32, physical M8/base1/merge2/origin0/spacing1"
            )
            return SupportResult((reason,))
        return SupportResult(())


LM_PHYSICAL_C16_LEVELS = ExecutionPlan(
    RadiusLMPhysicalC16LevelsAlgorithm().id,
    "v1",
    GLOBAL_ACTIVE_GRAD4_NATIVE.recipe,
)


@cache
def _physical_dispatcher():
    from torchcst._backends.dispatch import Dispatcher
    from torchcst._backends.registry import Registry

    registry = Registry()
    registry.register(RadiusLMPhysicalC16LevelsAlgorithm())
    return Dispatcher(registry=registry)


@dataclass(frozen=True)
class RadiusEnvelopeWideAmplitudeC16LMAlgorithm(RadiusEnvelopeWideAmplitudeLMAlgorithm):
    id: str = "cuda_radius_envelope_wide_amplitude_c16_lm"

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires a LinearContext",))
        try:
            self.validate_recipe(recipe)
            operator = physical_operator(context.operator)
            spec = context.operator.kernel.parameterization
            if spec.radius_min < 0.5 or spec.radius_max > 50 or spec.amplitude_max > 32:
                raise ValueError("requires radius[.5,50], amplitude_max<=32")
        except (ValueError, TypeError) as error:
            return SupportResult((str(error),))
        return RadiusLMPhysicalC16LevelsAlgorithm().supports(
            replace(context, operator=operator), LM_PHYSICAL_C16_LEVELS.recipe
        )

    def execute(self, state, inputs):
        from torchcst._backends.torch.parameterizations.radius_envelope import (
            prepared_physical_parameters,
        )

        _, parameters, _, _ = linear_execution(state, inputs)
        physical = prepared_physical_parameters(parameters, state.buffers["constants"])
        binding = LinearBinding(state.layout, physical)
        return _physical_dispatcher().run(binding, inputs, plan=LM_PHYSICAL_C16_LEVELS)


RADIUS_ENVELOPE_WIDE_AMPLITUDE_C16_LM = ExecutionPlan(
    RadiusEnvelopeWideAmplitudeC16LMAlgorithm().id, "v1", DefaultRecipe()
)
