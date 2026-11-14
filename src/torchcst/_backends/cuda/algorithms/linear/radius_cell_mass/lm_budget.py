"""Explicit M8 Atom budgets; unchanged numerical kernels, separate admission."""

import hashlib
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

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
class RadiusLMPhysicalBudgetLevelsAlgorithm(RadiusLMPhysicalLevelsAlgorithm):
    id: str = "cuda_radius_lm_physical_budget_levels"

    def _packed_extension(self):
        return budget_extension()

    def supports(self, context, recipe):
        result = RadiusCellMassAlgorithm.supports(self, context, recipe)
        if result.reasons:
            return result
        plan = hierarchy_from_operator(context.operator)
        if (
            (plan.input.site_count, plan.output.site_count)
            not in ((512, 2048), (2048, 512))
            or context.atom_count
            not in (16384, 24576, 32768, 49152, 65536, 98304, 131072)
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
                "budget physical LM requires width512/expansion4, A16384/24576/32768/49152/65536/98304/131072, R1..256, "
                "FP32, physical M8/base1/merge2/origin0/spacing1"
            )
            return SupportResult((reason,))
        return SupportResult(())


LM_PHYSICAL_BUDGET_LEVELS = ExecutionPlan(
    RadiusLMPhysicalBudgetLevelsAlgorithm().id,
    "v1",
    GLOBAL_ACTIVE_GRAD4_NATIVE.recipe,
)


@cache
def budget_extension():
    """Dedicated host admission; all mathematical kernels remain shared."""
    from torch.utils.cpp_extension import load

    directory = Path(__file__).parent
    source, host = directory / "pipeline.cu", directory / "host_autograd.cpp"
    inputs = [source, host, *sorted(directory.glob("*.cuh"))]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in inputs)).hexdigest()[:12]
    return load(
        name=f"torchcst_radius_lm_budget_{digest}",
        sources=[str(source), str(host)],
        extra_cflags=["-O2"],
        extra_cuda_cflags=[
            "-O2",
            "--ftz=false",
            "-DRADIUS_ROW_BATCH=8",
            "-DRADIUS_LM_BUDGET=1",
        ],
        verbose=False,
    )


@cache
def _physical_dispatcher():
    from torchcst._backends.dispatch import Dispatcher
    from torchcst._backends.registry import Registry

    registry = Registry()
    registry.register(RadiusLMPhysicalBudgetLevelsAlgorithm())
    return Dispatcher(registry=registry)


@dataclass(frozen=True)
class RadiusEnvelopeWideAmplitudeBudgetLMAlgorithm(
    RadiusEnvelopeWideAmplitudeLMAlgorithm
):
    id: str = "cuda_radius_envelope_wide_amplitude_budget_lm"

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
        return RadiusLMPhysicalBudgetLevelsAlgorithm().supports(
            replace(context, operator=operator), LM_PHYSICAL_BUDGET_LEVELS.recipe
        )

    def execute(self, state, inputs):
        from torchcst._backends.torch.parameterizations.radius_envelope import (
            prepared_physical_parameters,
        )

        _, parameters, _, _ = linear_execution(state, inputs)
        physical = prepared_physical_parameters(parameters, state.buffers["constants"])
        binding = LinearBinding(state.layout, physical)
        return _physical_dispatcher().run(
            binding, inputs, plan=LM_PHYSICAL_BUDGET_LEVELS
        )


RADIUS_ENVELOPE_WIDE_AMPLITUDE_BUDGET_LM = ExecutionPlan(
    RadiusEnvelopeWideAmplitudeBudgetLMAlgorithm().id, "v1", DefaultRecipe()
)
