"""Built-in Torch Linear plans used by the default selector and comparisons."""

from torchcst._backends.schema import DefaultRecipe, ExecutionPlan

TORCH_FACTORED_LINEAR = ExecutionPlan("torch_factored", "v1", DefaultRecipe())
TORCH_MATERIALIZED_LINEAR = ExecutionPlan("torch_materialized", "v1", DefaultRecipe())
TORCH_NORMALIZED_RADIAL_LINEAR = ExecutionPlan(
    "torch_normalized_radial", "v1", DefaultRecipe()
)
