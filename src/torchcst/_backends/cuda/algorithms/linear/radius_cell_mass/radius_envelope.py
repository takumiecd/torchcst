"""Opt-in envelope map composed with the qualified physical FFN Algorithm.

No CUDA arithmetic or old Plan semantics change. The map has its own contract;
stored q rows are never passed to a native physical-radius entrypoint.
"""

from dataclasses import dataclass, replace
from functools import cache

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, ExecutionPlan, SupportResult
from torchcst.kernels.parameterizations import RadiusEnvelopeSpec
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import LinearInputs, linear_execution


def physical_operator(operator):
    from torchcst import kernels

    spec = operator.kernel.parameterization
    if operator.kernel.revision != 8 or type(spec) is not RadiusEnvelopeSpec:
        raise ValueError("requires radius_envelope coordinates")
    kernel = kernels.product(
        *operator.kernel.profiles,
        coordinates=kernels.physical(
            radius_init=(spec.radius_min, spec.radius_max), radius_floor=spec.radius_min
        ),
    )
    return replace(operator, kernel=kernel)


@cache
def _physical_dispatcher():
    from torchcst._backends.dispatch import Dispatcher
    from torchcst._backends.registry import Registry

    from .lm_multirow import RadiusLMPhysicalLevelsAlgorithm

    registry = Registry()
    registry.register(RadiusLMPhysicalLevelsAlgorithm())
    return Dispatcher(registry=registry)


@dataclass(frozen=True)
class RadiusEnvelopeLMAlgorithm(Algorithm[DefaultRecipe]):
    id: str = "cuda_radius_envelope_lm"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = DefaultRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not DefaultRecipe:
            raise ValueError("radius envelope adapter has no recipe settings")

    def supports(self, context, recipe):
        from .lm_multirow import LM_PHYSICAL_LEVELS, RadiusLMPhysicalLevelsAlgorithm

        if not isinstance(context, LinearContext):
            return SupportResult(("requires a LinearContext",))
        try:
            self.validate_recipe(recipe)
            operator = physical_operator(context.operator)
            spec = context.operator.kernel.parameterization
            if spec.radius_min < 0.5 or spec.radius_max > 50 or spec.amplitude_max > 1:
                return SupportResult(
                    ("envelope LM requires radius[.5,50], amplitude_max<=1",)
                )
        except (ValueError, TypeError) as error:
            return SupportResult((str(error),))
        return RadiusLMPhysicalLevelsAlgorithm().supports(
            replace(context, operator=operator), LM_PHYSICAL_LEVELS.recipe
        )

    def workspace_bound(self, context, recipe):
        return None

    def execute(self, state, inputs):
        from torchcst._backends.torch.parameterizations.radius_envelope import (
            physical_parameters,
        )
        from torchcst.operators.execution import LinearBinding

        from .lm_multirow import LM_PHYSICAL_LEVELS

        _, parameters, operator, _ = linear_execution(state, inputs)
        spec = operator.kernel.parameterization
        physical = physical_parameters(
            parameters,
            **{
                name: getattr(spec, name)
                for name in ("radius_min", "radius_max", "tau", "amplitude_max")
            },
        )
        binding = LinearBinding(physical_operator(operator), physical)
        return _physical_dispatcher().run(binding, inputs, plan=LM_PHYSICAL_LEVELS)


RADIUS_ENVELOPE_LM = ExecutionPlan(
    RadiusEnvelopeLMAlgorithm().id, "v1", DefaultRecipe()
)
