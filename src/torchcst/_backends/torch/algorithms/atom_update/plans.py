"""Reference update for every supported Kernel/Geometry contract."""

from torchcst._backends.schema import DefaultRecipe, ExecutionPlan

REFERENCE = ExecutionPlan("torch_atom_update", "v1", DefaultRecipe())
