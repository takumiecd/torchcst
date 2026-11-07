"""Narrow FP32-intermediate/recompute candidates; no production selection."""

from dataclasses import dataclass, replace

import torch

from torchcst._backends.schema import ExecutionPlan
from torchcst._builtin_callbacks import seal_functions as _seal_functions
from torchcst.operators.execution import linear_execution

from .algorithm import RadiusCellMassAlgorithm
from .plans import GLOBAL_ACTIVE_GRAD4_NATIVE
from .row8 import RadiusRow8Algorithm


@dataclass(frozen=True)
class RadiusPackedAlgorithm(RadiusRow8Algorithm):
    id: str = "cuda_radius_cell_mass_packed"

    @property
    def recompute(self):
        return False

    @property
    def parallel(self):
        return False

    def validate_recipe(self, recipe):
        RadiusCellMassAlgorithm.validate_recipe(
            self, replace(recipe, save_response=True)
        )
        expected = replace(
            GLOBAL_ACTIVE_GRAD4_NATIVE.recipe, save_response=not self.recompute
        )
        if recipe != expected:
            raise ValueError("packed recipe must declare its exact retention policy")

    def execute(self, state, inputs):
        x, parameters, _, _ = linear_execution(state, inputs)
        from .executor import _extension, checked_radius_linear

        if (
            type(x) not in (torch.Tensor, torch.nn.Parameter)
            or type(parameters) not in (torch.Tensor, torch.nn.Parameter)
            or torch.overrides.has_torch_function((x, parameters))
            or torch._C._are_functorch_transforms_active()
            or torch._C._get_tracing_state() is not None
            or torch.autograd.forward_ad._current_level != -1
            or torch._C._is_torch_function_mode_enabled()
            or torch._C._len_torch_dispatch_stack()
        ):
            return checked_radius_linear(
                state.layout, x, parameters, recipe=GLOBAL_ACTIVE_GRAD4_NATIVE.recipe
            )
        return _extension(8).packed_rows_linear(
            x.contiguous(),
            parameters.contiguous(),
            state.layout.input.native_config(),
            state.layout.output.native_config(),
            self.recompute,
            self.parallel,
        )


@dataclass(frozen=True)
class RadiusPackedRecomputeAlgorithm(RadiusPackedAlgorithm):
    id: str = "cuda_radius_cell_mass_packed_recompute"

    @property
    def recompute(self):
        return True


@dataclass(frozen=True)
class RadiusPackedParallelAlgorithm(RadiusPackedRecomputeAlgorithm):
    id: str = "cuda_radius_cell_mass_packed_parallel"

    @property
    def parallel(self):
        return True


PACKED_NATIVE = ExecutionPlan(
    "cuda_radius_cell_mass_packed", "v1", GLOBAL_ACTIVE_GRAD4_NATIVE.recipe
)
PACKED_RECOMPUTE = ExecutionPlan(
    "cuda_radius_cell_mass_packed_recompute",
    "v1",
    replace(GLOBAL_ACTIVE_GRAD4_NATIVE.recipe, save_response=False),
)
PACKED_PARALLEL = ExecutionPlan(
    "cuda_radius_cell_mass_packed_parallel", "v1", PACKED_RECOMPUTE.recipe
)

_seal_functions(globals())
