"""Fixed-c64 M8 width study; separate shape admission, unchanged arithmetic."""

import hashlib
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import ClassVar

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

WIDTH_COHORTS = tuple(
    (ni, no, 128 * width)
    for width in (512, 1024, 2048)
    for ni, no in ((width, 4 * width), (4 * width, width))
)
QKV_2048_COHORTS = tuple(
    (2048, 2048, 2048 * coefficient) for coefficient in (48, 56, 64, 72, 80, 128)
)
SHARED_WIDE_DOWN_COHORT = (4096, 1024, 196608)
DENSE_FOURIER_COHORTS = (SHARED_WIDE_DOWN_COHORT, *QKV_2048_COHORTS)
PROJECTION_COHORTS = (
    (1024, 1024, 65536),  # Historical QKV
    (1024, 1024, 49152),  # O only
    *DENSE_FOURIER_COHORTS,
)


@dataclass(frozen=True)
class RadiusLMPhysicalWidthLevelsAlgorithm(RadiusLMPhysicalLevelsAlgorithm):
    id: str = "cuda_radius_lm_physical_width_levels"

    def _packed_extension(self):
        return width_extension()

    def supports(self, context, recipe):
        result = RadiusCellMassAlgorithm.supports(self, context, recipe)
        if result.reasons:
            return result
        plan = hierarchy_from_operator(context.operator)
        if (
            (plan.input.site_count, plan.output.site_count, context.atom_count)
            not in (*WIDTH_COHORTS, *PROJECTION_COHORTS)
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
                "width physical LM requires an admitted FFN, QKV or O shape, R1..256, "
                "FP32, physical M8/base1/merge2/origin0/spacing1"
            )
            return SupportResult((reason,))
        return SupportResult(())


LM_PHYSICAL_WIDTH_LEVELS = ExecutionPlan(
    RadiusLMPhysicalWidthLevelsAlgorithm().id,
    "v1",
    GLOBAL_ACTIVE_GRAD4_NATIVE.recipe,
)


@dataclass(frozen=True)
class RadiusLMPhysicalSharedWideAlgorithm(RadiusLMPhysicalWidthLevelsAlgorithm):
    """Aggregate wide hierarchy tiles once across the expansion blocks."""

    id: str = "cuda_radius_lm_physical_shared_wide"

    @property
    def shared_wide(self):
        return True

    def supports(self, context, recipe):
        result = super().supports(context, recipe)
        if result.reasons:
            return result
        plan = hierarchy_from_operator(context.operator)
        if (
            plan.input.site_count,
            plan.output.site_count,
            context.atom_count,
        ) != SHARED_WIDE_DOWN_COHORT:
            return SupportResult(
                ("shared wide tiles require the width-1024 down cohort",)
            )
        return result


LM_PHYSICAL_SHARED_WIDE = ExecutionPlan(
    RadiusLMPhysicalSharedWideAlgorithm().id, "v1", GLOBAL_ACTIVE_GRAD4_NATIVE.recipe
)


@dataclass(frozen=True)
class RadiusLMPhysicalDenseFourierWeightAlgorithm(RadiusLMPhysicalWidthLevelsAlgorithm):
    id: str = "cuda_radius_lm_physical_dense_fourier_weight"

    def supports(self, context, recipe):
        result = super().supports(context, recipe)
        if not result.supported:
            return result
        plan = hierarchy_from_operator(context.operator)
        if (
            plan.input.site_count,
            plan.output.site_count,
            context.atom_count,
        ) not in DENSE_FOURIER_COHORTS:
            return SupportResult(
                ("dense Fourier weights require an admitted FFN down or width-2048 QKV cohort",)
            )
        if context.precision.allow_tf32:
            return SupportResult(("dense Fourier weights require IEEE FP32 matmul",))
        return result

    @property
    def recompute(self):
        # The recipe disables retained per-Atom response tensors.
        return True

    @property
    def packed_levels_entry(self):
        return "packed_dense_fourier_weight_linear"


LM_PHYSICAL_DENSE_FOURIER_WEIGHT = ExecutionPlan(
    RadiusLMPhysicalDenseFourierWeightAlgorithm().id,
    "v1",
    replace(GLOBAL_ACTIVE_GRAD4_NATIVE.recipe, save_response=False),
)


@cache
def width_extension():
    """Dedicated host admission; all mathematical kernels remain shared."""
    from torch.utils.cpp_extension import load

    directory = Path(__file__).parent
    source, host = directory / "pipeline.cu", directory / "host_autograd.cpp"
    inputs = [source, host, *sorted(directory.glob("*.cuh"))]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in inputs)).hexdigest()[:12]
    return load(
        name=f"torchcst_radius_lm_width_{digest}",
        sources=[str(source), str(host)],
        extra_cflags=["-O2", "-DRADIUS_LM_WIDTH=1"],
        extra_cuda_cflags=[
            "-O2",
            "--ftz=false",
            "-DRADIUS_ROW_BATCH=8",
            "-DRADIUS_LM_WIDTH=1",
        ],
        verbose=False,
    )


@cache
def _physical_dispatcher():
    from torchcst._backends.dispatch import Dispatcher
    from torchcst._backends.registry import Registry

    registry = Registry()
    registry.register(RadiusLMPhysicalWidthLevelsAlgorithm())
    registry.register(RadiusLMPhysicalSharedWideAlgorithm())
    registry.register(RadiusLMPhysicalDenseFourierWeightAlgorithm())
    return Dispatcher(registry=registry)


@dataclass(frozen=True)
class RadiusEnvelopeWideAmplitudeWidthLMAlgorithm(
    RadiusEnvelopeWideAmplitudeLMAlgorithm
):
    id: str = "cuda_radius_envelope_wide_amplitude_width_lm"
    physical_plan: ClassVar[ExecutionPlan] = LM_PHYSICAL_WIDTH_LEVELS
    physical_algorithm: ClassVar[type[RadiusLMPhysicalWidthLevelsAlgorithm]] = (
        RadiusLMPhysicalWidthLevelsAlgorithm
    )

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires a LinearContext",))
        try:
            self.validate_recipe(recipe)
            operator = physical_operator(context.operator)
            spec = context.operator.kernel.parameterization
            if (
                spec.radius_min < 0.5
                or spec.radius_max > 512
                or spec.amplitude_max > 32
            ):
                raise ValueError("requires radius[.5,512], amplitude_max<=32")
        except (ValueError, TypeError) as error:
            return SupportResult((str(error),))
        return self.physical_algorithm().supports(
            replace(context, operator=operator), self.physical_plan.recipe
        )

    def execute(self, state, inputs):
        from torchcst._backends.torch.parameterizations.radius_envelope import (
            prepared_physical_parameters,
        )

        _, parameters, _, _ = linear_execution(state, inputs)
        physical = prepared_physical_parameters(parameters, state.buffers["constants"])
        binding = LinearBinding(state.layout, physical)
        return _physical_dispatcher().run(binding, inputs, plan=self.physical_plan)


RADIUS_ENVELOPE_WIDE_AMPLITUDE_WIDTH_LM = ExecutionPlan(
    RadiusEnvelopeWideAmplitudeWidthLMAlgorithm().id, "v1", DefaultRecipe()
)


@dataclass(frozen=True)
class RadiusEnvelopeSharedWideLMAlgorithm(RadiusEnvelopeWideAmplitudeWidthLMAlgorithm):
    id: str = "cuda_radius_envelope_shared_wide_lm"
    physical_plan: ClassVar[ExecutionPlan] = LM_PHYSICAL_SHARED_WIDE
    physical_algorithm: ClassVar[type[RadiusLMPhysicalWidthLevelsAlgorithm]] = (
        RadiusLMPhysicalSharedWideAlgorithm
    )


RADIUS_ENVELOPE_SHARED_WIDE_LM = ExecutionPlan(
    RadiusEnvelopeSharedWideLMAlgorithm().id, "v1", DefaultRecipe()
)


@dataclass(frozen=True)
class RadiusEnvelopeDenseFourierWeightLMAlgorithm(RadiusEnvelopeWideAmplitudeWidthLMAlgorithm):
    id: str = "cuda_radius_envelope_dense_fourier_weight_lm"
    physical_plan: ClassVar[ExecutionPlan] = LM_PHYSICAL_DENSE_FOURIER_WEIGHT
    physical_algorithm: ClassVar[type[RadiusLMPhysicalWidthLevelsAlgorithm]] = (
        RadiusLMPhysicalDenseFourierWeightAlgorithm
    )


RADIUS_ENVELOPE_DENSE_FOURIER_WEIGHT_LM = ExecutionPlan(
    RadiusEnvelopeDenseFourierWeightLMAlgorithm().id, "v1", DefaultRecipe()
)
