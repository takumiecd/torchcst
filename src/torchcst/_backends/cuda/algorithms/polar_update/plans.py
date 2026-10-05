"""Explicit opt-in fused Polar update, independent of the Linear Plan."""

from torchcst._backends.schema import DefaultRecipe, ExecutionPlan

FUSED = ExecutionPlan("cuda_polar_update_fused", "v1", DefaultRecipe())
