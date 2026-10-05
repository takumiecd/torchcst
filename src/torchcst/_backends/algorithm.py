"""Backend-independent processing/state lifecycle; no operator-family hierarchy."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from torch import Tensor

from .schema import Context, SupportResult
from .state import AlgorithmState

RecipeT = TypeVar("RecipeT")


@dataclass(frozen=True)
class Algorithm(ABC, Generic[RecipeT]):
    """Stateless processing; the caller owns each AlgorithmState.

    Concrete algorithms define operation inputs, preparation and execution.
    Registry instances never retain input tensors or mutable learning state.
    """

    id: str
    revision: str
    operation_id: str
    semantics_id: str
    recipe_type: type[RecipeT]
    input_type: type

    def __post_init__(self):
        for name in ("id", "revision", "operation_id", "semantics_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"algorithm {name} must be a nonempty string")
        for name in ("recipe_type", "input_type"):
            if not isinstance(getattr(self, name), type):
                raise TypeError(f"algorithm {name} must be a type")

    @abstractmethod
    def validate_recipe(self, recipe: RecipeT) -> None:
        """Check this implementation's settings without loading compute code."""

    @abstractmethod
    def supports(self, context: Context, recipe: RecipeT) -> SupportResult:
        """Check the complete operation and runtime contract using metadata."""

    @abstractmethod
    def workspace_bound(self, context: Context, recipe: RecipeT) -> int | None:
        """Bound managed scratch; this is not a measured memory peak."""

    def matches(self, context: Context, recipe: RecipeT) -> bool:
        """Routing eligibility; Dispatch still enforces full support after selection."""
        return self.supports(context, recipe).supported

    def validate_inputs(self, state, inputs):
        if type(inputs) is not self.input_type:
            raise TypeError("inputs type differs from algorithm contract")

    def create_state(self, binding, *, recipe, **configuration):
        self.validate_recipe(recipe)
        return AlgorithmState(binding, self, recipe=recipe, **configuration)

    def prepare(self, state):
        state.invalidate()
        state.mark_current()

    def run(self, state, inputs):
        if not isinstance(state, AlgorithmState) or state.algorithm is not self:
            raise ValueError("state belongs to a different algorithm")
        self.validate_inputs(state, inputs)
        self.validate_recipe(state.recipe)
        if not state.is_current():
            self.prepare(state)
        if not state.is_current():
            raise RuntimeError("algorithm did not prepare the current binding state")
        result = self.execute(state, inputs)
        owner = state.atom_state
        return (
            owner.protect_tensor(result)
            if owner is not None and isinstance(result, Tensor)
            else result
        )

    @abstractmethod
    def execute(self, state, inputs):
        """Execute with the bound recipe/configuration and typed invocation data."""
