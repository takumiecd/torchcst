"""Validated recipes for two ordinary registered Algorithm candidates."""

from torchcst._backends.schema import ExecutionPlan

from .full.recipe import FullRecipe
from .window.recipe import WindowRecipe

FULL = ExecutionPlan("normalized_full", "v1", FullRecipe())
WINDOW = ExecutionPlan("normalized_window", "v1", WindowRecipe())
