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

    def __post_init__(self):
        for name in ("id", "revision", "operation_id", "semantics_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"algorithm {name} must be a nonempty string")
        if not isinstance(self.recipe_type, type):
            raise TypeError("algorithm recipe_type must be a type")

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
        """Routing eligibility; Registry still enforces full support after selection."""
        return self.supports(context, recipe).supported

    def validate_inputs(self, context: Context, **inputs):
        """Use the operation's validator, or override for an Algorithm's interface."""
        context.validate_inputs(**inputs)

    def create_state(self, atom_state, **configuration):
        return AlgorithmState(atom_state, self, **configuration)

    def prepare(self, state):
        state.invalidate()
        state.mark_current()

    def run(self, state, **inputs):
        if not isinstance(state, AlgorithmState) or state.algorithm is not self:
            raise ValueError("state belongs to a different algorithm")
        if not state.is_current():
            self.prepare(state)
        if not state.is_current():
            raise RuntimeError("algorithm did not prepare the current AtomState layout")
        result = self._execute_state(state, **inputs)
        return (
            state.atom_state.protect_tensor(result)
            if isinstance(result, Tensor)
            else result
        )

    def _execute_state(self, state, **inputs):
        if "recipe" in state.configuration and state.configuration[
            "recipe"
        ] != inputs.get("recipe"):
            raise ValueError("algorithm state recipe differs")
        return self.execute(state=state, **inputs)

    @abstractmethod
    def execute(self, **inputs):
        """Execute this algorithm's declared operation."""
