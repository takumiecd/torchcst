"""Explicit opt-in Plans; none is added to the automatic selector."""

from torchcst._backends.schema import ExecutionPlan

from .recipe import HierarchicalCellMassRecipe

CONSERVATIVE = ExecutionPlan(
    "cuda_cell_mass_hierarchical", "v1", HierarchicalCellMassRecipe()
)
VALIDATED_128 = ExecutionPlan(
    "cuda_cell_mass_hierarchical", "v1", HierarchicalCellMassRecipe("validated_128")
)
VALIDATED_1024 = ExecutionPlan(
    "cuda_cell_mass_hierarchical", "v1", HierarchicalCellMassRecipe("validated_1024")
)
