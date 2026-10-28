"""Explicit research Plans; the automatic selector is unchanged."""

from torchcst._backends.schema import ExecutionPlan

from .recipe import RadiusCellMassRecipe

REFERENCE = ExecutionPlan("cuda_radius_cell_mass", "v1", RadiusCellMassRecipe())
DIRECT = ExecutionPlan("cuda_radius_cell_mass", "v1", RadiusCellMassRecipe("direct"))
