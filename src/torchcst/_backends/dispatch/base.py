"""Replaceable Plan selection, with shared support checks and fallback handling."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torchcst._backends.registry import Registry
from torchcst._backends.schema import (
    Context,
    ExecutionPlan,
)


@dataclass(frozen=True)
class Match:
    plan: ExecutionPlan
    path: tuple[str, ...]
    reason: str
    evidence_ids: tuple[str, ...] = ()


class Selector(ABC):
    """Select using metadata only; never retain tensors or backward state.

    Implementations propose one Plan in ``_match``. All implementations share
    registry validation, workspace enforcement and an explicitly supplied
    fallback. Ranking/training happens before construction, outside this API.
    """

    def __init__(self, *, revision: str, registry: Registry, fallback_plan=None):
        if type(revision) is not str or not revision:
            raise ValueError("selector revision must be a nonempty string")
        if not isinstance(registry, Registry):
            raise TypeError("selector requires a Registry")
        if fallback_plan is not None:
            registry.validate_plan(fallback_plan)
        self._revision = revision
        self._registry = registry
        self._fallback_plan = fallback_plan

    @property
    def revision(self):
        return self._revision

    @property
    def registry(self):
        return self._registry

    @abstractmethod
    def _match(self, context: Context) -> Match | None:
        """Return a proposed Plan, or abstain for an unobserved condition."""

    @property
    def fallback_plan(self):
        return self._fallback_plan

    def select(self, context: Context):
        """Metadata-only convenience; use the shared dispatch validation path."""
        from .runtime import select_plan

        return select_plan(self, context)
