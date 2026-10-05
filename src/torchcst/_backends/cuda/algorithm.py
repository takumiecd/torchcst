"""Common algorithm contract shared by dispatch, registry and benchmarks."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from torch import Tensor

from torchcst._backends.algorithm import Algorithm as BaseAlgorithm
from torchcst._backends.cuda.schema import (
    DispatchContext,
    SupportResult,
)
from torchcst.operators.spec import OperatorSpec

RecipeT = TypeVar("RecipeT")


@dataclass(frozen=True)
class Algorithm(BaseAlgorithm, Generic[RecipeT]):
    """Immutable identity and recipe-specific execution interface.

    Construction, recipe validation and support checks must not import GPU
    implementations or read tensor values. ``execute`` owns the forward and
    autograd connection, including per-invocation backward state. An algorithm
    instance must not retain input tensors, dynamic support or gradient buffers.
    Structural caches may live in the lazily loaded implementation.

    The semantic ID fixes the mathematical operator; recipe tuning cannot
    redefine normalization, support or parameter derivatives.
    """

    id: str
    revision: str
    operation_id: str
    semantics_id: str
    recipe_type: type[RecipeT]

    def __post_init__(self) -> None:
        for name in ("id", "revision", "operation_id", "semantics_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"algorithm {name} must be a nonempty string")
        if not isinstance(self.recipe_type, type):
            raise TypeError("algorithm recipe_type must be a type")

    def _execute_state(self, state, **inputs):
        if state.configuration.get("recipe") != inputs.get("recipe"):
            raise ValueError("algorithm state recipe differs")
        operator = state.configuration.get("operator")
        if operator is not None and operator != inputs.get("operator"):
            raise ValueError("algorithm state operator differs")
        # Existing CUDA executors consume their established tensor interfaces.
        # They may specialize create_state/prepare as placement is introduced.
        return self.execute(**inputs)

    @abstractmethod
    def validate_recipe(self, recipe: RecipeT) -> None:
        """Reject invalid or unvalidated settings before execution."""

    @abstractmethod
    def supports(self, context: DispatchContext, recipe: RecipeT) -> SupportResult:
        """Check the full Kernel/Chart mathematical contract and runtime metadata.

        A matching operation ID alone does not establish mathematical support.
        Reject unrecognized revisions, parameterizations and normalization here.
        """

    @abstractmethod
    def workspace_bound(self, context: DispatchContext, recipe: RecipeT) -> int | None:
        """Bound algorithm-managed scratch/state, or return None if unknown.

        This is not a measured allocated/reserved/process GPU memory peak.
        """

    @abstractmethod
    def execute(
        self,
        *,
        x: Tensor,
        parameters: Tensor,
        operator: OperatorSpec,
        recipe: RecipeT,
    ) -> Tensor:
        """Return Y with the recipe's forward/backward autograd connection.

        Registry validates the context and recipe before calling this method.
        Load GPU code here when needed. Backward must use the settings and saved
        tensors from this forward, without selecting another plan. Parameters
        must not be updated during execution; the optimizer owns updates.
        """
