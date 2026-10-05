"""Declared Torch Polar update; calculation is loaded only at execution."""

from torchcst._backends.schema import DefaultRecipe, ExecutionPlan

TORCH = ExecutionPlan("torch_polar_update", "v1", DefaultRecipe())
