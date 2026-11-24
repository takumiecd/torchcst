"""Torch reference algorithms for CSTConv2d."""

from dataclasses import dataclass
from typing import Literal

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, ExecutionPlan, SupportResult
from torchcst.operators.conv2d import Conv2dContext, Conv2dInputs


def apply_conv2d(x, weight, context):
    import torch.nn.functional as F

    return F.conv2d(
        x,
        weight,
        stride=context.stride,
        padding=context.padding,
        dilation=context.dilation,
        groups=context.groups,
    )


def apply_tap_weights(x, weight, context):
    import torch

    _, out_sites, in_sites = weight.shape
    filters = weight.permute(1, 2, 0).reshape(out_sites, in_sites, *context.kernel_size)
    layout = (
        torch.channels_last
        if x.is_contiguous(memory_format=torch.channels_last)
        else torch.contiguous_format
    )
    return apply_conv2d(x, filters.contiguous(memory_format=layout), context)


def _kernel_spec(context):
    kernel = context.operator.kernel
    while kernel.composition == "amplitude":
        kernel = kernel.inner
    return kernel


@dataclass(frozen=True)
class TorchConv2dRecipe:
    method: Literal["factored", "materialized", "composed"]


@dataclass(frozen=True)
class TorchConv2dAlgorithm(Algorithm):
    id: str = "torch_conv2d"
    revision: str = "v1"
    operation_id: str = "conv2d"
    semantics_id: str = "conv2d_kernel_atom_sum_v1"
    recipe_type: type = TorchConv2dRecipe
    input_type: type = Conv2dInputs

    def validate_recipe(self, recipe):
        if (
            type(recipe) is not TorchConv2dRecipe
            or type(recipe.method) is not str
            or recipe.method not in ("factored", "materialized", "composed")
        ):
            raise ValueError("expected a TorchConv2dRecipe with a supported method")

    def supports(self, context, recipe):
        if not isinstance(context, Conv2dContext):
            return SupportResult(("requires a Conv2dContext",))
        kernel = _kernel_spec(context)
        if recipe.method != "materialized" and (
            context.atom_layout != "patch" or kernel.composition != "separable"
        ):
            return SupportResult(("requires a separable kernel on the patch layout",))
        if recipe.method == "factored" and context.groups != 1:
            return SupportResult(("factored Conv2d requires groups=1",))
        if context.execution_mode == "cuda_graph" and kernel.revision in (7, 8):
            return SupportResult(
                ("Torch radius cell integration requires eager execution",)
            )
        return SupportResult()

    def matches(self, context, recipe):
        if not self.supports(context, recipe).supported:
            return False
        if recipe.method != "factored":
            return True
        ni, no = context.operator.in_features, context.operator.out_features
        return context.atom_count * (ni + no) <= ni * no

    def workspace_bound(self, context, recipe):
        return None

    def execute(self, state, inputs):
        import torch.nn.functional as F

        from torchcst._backends.torch.kernels import execution as kernel

        binding, context = state.binding, state.context
        p = binding.atom_state.parameters_for_execution()
        if state.recipe.method == "materialized":
            return apply_conv2d(
                inputs.x, binding._materialize_atoms(p).sum(dim=0), context
            )
        in_features = context.operator.in_features // context.taps
        if state.recipe.method == "composed":
            weight = kernel.compose_weight(binding.kernel, *binding.cst_charts(), p)
            filters = weight.reshape(
                context.operator.out_features, in_features, *context.kernel_size
            )
            return apply_conv2d(inputs.x, filters, context)
        fi, fo = kernel.factors(binding.kernel, *binding.cst_charts(), p)
        filters = fi.T.reshape(p.size(0), in_features, *context.kernel_size)
        responses = apply_conv2d(inputs.x, filters, context)
        mixing = fo.reshape(context.operator.out_features, p.size(0), 1, 1)
        return F.conv2d(responses, mixing)


@dataclass(frozen=True)
class TorchChannelConv2dAlgorithm(Algorithm):
    id: str = "torch_channel_conv2d"
    revision: str = "v1"
    operation_id: str = "conv2d"
    semantics_id: str = "channel_taps_conv2d_v1"
    recipe_type: type = DefaultRecipe
    input_type: type = Conv2dInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not DefaultRecipe:
            raise TypeError("expected DefaultRecipe")

    def supports(self, context, recipe):
        if not isinstance(context, Conv2dContext) or not context.has_channel_taps:
            return SupportResult(
                ("requires atom_layout='channel_taps' or a 1×1 layer",)
            )
        kernel = _kernel_spec(context)
        if kernel.composition != "separable":
            return SupportResult(("requires a separable kernel",))
        if context.atom_count % context.taps:
            return SupportResult(("Atom count must be divisible by the tap count",))
        if context.execution_mode == "cuda_graph" and kernel.revision in (7, 8):
            return SupportResult(
                ("Torch radius cell integration requires eager execution",)
            )
        return SupportResult()

    def matches(self, context, recipe):
        return (
            isinstance(context, Conv2dContext)
            and context.atom_layout == "channel_taps"
            and self.supports(context, recipe).supported
        )

    def workspace_bound(self, context, recipe):
        return None

    def execute(self, state, inputs):
        import torch

        from torchcst._backends.torch.kernels.execution import factors

        binding, context = state.binding, state.context
        rows = binding.channel_tap_rows()
        with torch.autocast(device_type=rows.device.type, enabled=False):
            fi, fo = factors(binding.kernel, *binding.cst_charts(), rows.flatten(0, 1))
            i = fi.T.reshape(*rows.shape[:2], fi.size(0))
            o = fo.T.reshape(*rows.shape[:2], fo.size(0))
            weight = o.transpose(1, 2) @ i
        return apply_tap_weights(inputs.x, weight, context)


TORCH_CHANNEL_CONV2D = ExecutionPlan("torch_channel_conv2d", "v1", DefaultRecipe())
TORCH_FACTORED_CONV2D = ExecutionPlan(
    "torch_conv2d", "v1", TorchConv2dRecipe("factored")
)
TORCH_MATERIALIZED_CONV2D = ExecutionPlan(
    "torch_conv2d", "v1", TorchConv2dRecipe("materialized")
)
TORCH_COMPOSED_CONV2D = ExecutionPlan(
    "torch_conv2d", "v1", TorchConv2dRecipe("composed")
)
