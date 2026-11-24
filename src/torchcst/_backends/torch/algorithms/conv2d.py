"""Torch reference for CSTConv2d with independent channel taps."""

from dataclasses import dataclass

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, ExecutionPlan, SupportResult
from torchcst.operators.conv2d import Conv2dContext, Conv2dInputs


def apply_tap_weights(x, weight, context):
    import torch
    import torch.nn.functional as F

    _, out_sites, in_sites = weight.shape
    filters = weight.permute(1, 2, 0).reshape(out_sites, in_sites, *context.kernel_size)
    layout = (
        torch.channels_last
        if x.is_contiguous(memory_format=torch.channels_last)
        else torch.contiguous_format
    )
    return F.conv2d(
        x,
        filters.contiguous(memory_format=layout),
        stride=context.stride,
        padding=context.padding,
        dilation=context.dilation,
        groups=context.groups,
    )


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
        kernel = context.operator.kernel
        while kernel.composition == "amplitude":
            kernel = kernel.inner
        if kernel.composition != "separable":
            return SupportResult(("requires a separable kernel",))
        if context.atom_count % context.taps:
            return SupportResult(("Atom count must be divisible by the tap count",))
        if context.execution_mode == "cuda_graph" and kernel.revision in (7, 8):
            return SupportResult(
                ("Torch radius cell integration requires eager execution",)
            )
        return SupportResult()

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
