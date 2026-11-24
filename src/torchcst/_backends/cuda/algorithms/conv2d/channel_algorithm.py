"""Explicit CUDA Fourier plans for rational-radius channel taps."""

from dataclasses import dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import ExecutionPlan, SupportResult
from torchcst.kernels.parameterizations import RationalRadiusSpec
from torchcst.kernels.radius_cell_mass import radius_cell_mass_axes
from torchcst.operators.conv2d import Conv2dContext, Conv2dInputs


@dataclass(frozen=True)
class ChannelFourierRecipe:
    active_banks_only: bool = False
    sort_owners: bool = False


@dataclass(frozen=True)
class ChannelFourierAlgorithm(Algorithm):
    id: str = "cuda_channel_fourier_conv2d"
    revision: str = "v1"
    operation_id: str = "conv2d"
    semantics_id: str = "channel_taps_conv2d_v1"
    recipe_type: type = ChannelFourierRecipe
    input_type: type = Conv2dInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not ChannelFourierRecipe or any(
            type(flag) is not bool
            for flag in (recipe.active_banks_only, recipe.sort_owners)
        ):
            raise ValueError("expected a ChannelFourierRecipe with boolean flags")

    def supports(self, context, recipe):
        if not isinstance(context, Conv2dContext) or not context.has_channel_taps:
            return SupportResult(
                ("requires atom_layout='channel_taps' or a 1×1 layer",)
            )
        if context.kernel_size not in ((1, 1), (3, 3)) or context.groups != 1:
            return SupportResult(("requires 1×1 or 3×3 with groups=1",))
        if (
            context.atom_count % context.taps
            or not 1 <= context.atom_count // context.taps <= 4096
            or context.parameter_dim != 4
        ):
            return SupportResult(("requires 1..4096 four-field Atoms per tap",))
        kernel = context.operator.kernel
        spec = kernel.parameterization
        if type(spec) is not RationalRadiusSpec:
            return SupportResult(("requires kernels.rational_radius coordinates",))
        try:
            axes = radius_cell_mass_axes(context.operator.charts, kernel)
        except ValueError as error:
            return SupportResult((str(error),))
        if any(
            axis.site_count not in (16, 32, 64, 128)
            or axis.spacing != 1
            or axis.origin != 0
            or axis.binding.profile.modes != 8
            or axis.binding.base_tile_sites != 1
            or axis.binding.merge_factor != 2
            for axis in axes
        ):
            return SupportResult(
                ("requires 16/32/64/128-site origin-0 unit tori with M8/base1/merge2",)
            )
        if (
            not 0.5 <= spec.radius_min
            or spec.radius_max not in (1, 2, 4, 8, 16, 32, 64)
            or spec.amplitude_max > 32
        ):
            return SupportResult(
                (
                    "requires radius_min >= 0.5, a dyadic radius_max, and amplitude_max <= 32",
                )
            )
        if (
            context.device.type != "cuda"
            or context.device.compute_capability is None
            or context.device.compute_capability < (8, 0)
            or context.dtype != torch.float32
            or context.parameter_dtype != torch.float32
            or context.precision.autocast
            or context.precision.allow_tf32
            or context.deterministic
        ):
            return SupportResult(
                (
                    "requires CUDA FP32 on compute capability 8.0+, with autocast, TF32 and deterministic algorithms off",
                )
            )
        if recipe.active_banks_only and context.execution_mode == "cuda_graph":
            return SupportResult(("occupied-bank Fourier requires eager execution",))
        return SupportResult()

    def workspace_bound(self, context, recipe):
        return None

    def execute(self, state, inputs):
        from torchcst._backends.torch.algorithms.conv2d import apply_tap_weights

        from .radius_channel import channel_fourier_weights, channel_rational_parameters

        context = state.context
        spec = context.operator.kernel.parameterization
        rows = state.binding.channel_tap_rows()
        physical = channel_rational_parameters(
            rows,
            radius_min=spec.radius_min,
            radius_birth=spec.radius_birth,
            radius_cap=int(spec.radius_max),
            amplitude_max=spec.amplitude_max,
        )
        weight = channel_fourier_weights(
            physical,
            context.operator.in_features,
            output_sites=context.operator.out_features,
            radius_cap=int(spec.radius_max),
            sort_owners=state.recipe.sort_owners,
            active_banks_only=state.recipe.active_banks_only,
        )
        return apply_tap_weights(inputs.x, weight, context)


CHANNEL_FOURIER = ExecutionPlan(
    "cuda_channel_fourier_conv2d", "v1", ChannelFourierRecipe()
)
CHANNEL_FOURIER_ACTIVE = ExecutionPlan(
    CHANNEL_FOURIER.algorithm_id,
    CHANNEL_FOURIER.algorithm_revision,
    ChannelFourierRecipe(active_banks_only=True),
)
