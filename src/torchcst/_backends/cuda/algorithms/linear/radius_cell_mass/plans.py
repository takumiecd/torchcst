"""Explicit research Plans; the automatic selector is unchanged."""

from torchcst._backends.schema import ExecutionPlan

from .recipe import RadiusCellMassRecipe

REFERENCE = ExecutionPlan("cuda_radius_cell_mass", "v1", RadiusCellMassRecipe())
DIRECT = ExecutionPlan("cuda_radius_cell_mass", "v1", RadiusCellMassRecipe("direct"))

# DIRECT above means direct *moment construction*. These plans instead bypass
# moments/coeffs for low levels, while retaining exactly the same cell masses.
HYBRID_SERIAL = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(input_direct_levels=2, output_direct_levels=2),
)
HYBRID_OVERLAP = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2, output_direct_levels=2, schedule="overlap"
    ),
)
BLOCK_LOCAL = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2, output_direct_levels=2, schedule="block_local"
    ),
)

BLOCK_LOCAL_PARALLEL = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2, output_direct_levels=2, schedule="block_local_parallel"
    ),
)
