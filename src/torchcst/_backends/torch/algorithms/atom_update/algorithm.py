"""Fallback executes the declared coordinate law, including geometry retraction."""

from dataclasses import dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, SupportResult
from torchcst.operators.atom_update import AtomUpdateContext, AtomUpdateInputs


@dataclass(frozen=True)
class AtomUpdateAlgorithm(Algorithm[DefaultRecipe]):
    id: str = "torch_atom_update"
    revision: str = "v1"
    operation_id: str = "atom_update"
    semantics_id: str = "kernel-coordinate-update-v1"
    recipe_type: type = DefaultRecipe
    input_type: type = AtomUpdateInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not DefaultRecipe:
            raise ValueError("reference atom update has no tunable recipe")

    def supports(self, context, recipe):
        if not isinstance(context, AtomUpdateContext):
            return SupportResult(("requires AtomUpdateContext",))
        if context.execution_mode == "cuda_graph":
            return SupportResult(
                ("general coordinate reference requires eager execution",)
            )
        return SupportResult()

    def workspace_bound(self, context, recipe):
        return None

    @torch.no_grad()
    def execute(self, state, inputs):
        from torchcst._backends.torch.kernels import execution

        binding = state.binding
        proposal = binding.execution_parameters()
        updated = execution.apply_parameter_update(
            binding.kernel,
            *binding.cst_charts(),
            inputs.previous,
            proposal - inputs.previous,
            step_size=inputs.step_size,
        )
        proposal.copy_(updated)
        return proposal.detach()
