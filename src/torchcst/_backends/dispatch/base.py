"""Replaceable Plan selection, with shared support checks and fallback handling."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torchcst._backends.registry import Registry
from torchcst._backends.schema import (
    Context,
    DispatchDecision,
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

    def select(self, context: Context) -> DispatchDecision:
        match = self._match(context)
        failure = "unobserved condition"
        if match is not None:
            if not isinstance(match, Match):
                raise TypeError("selector must return a Match or None")
            try:
                algorithm = self._registry.validate(match.plan, context)
            except ValueError as error:
                failure = str(error)
                if self._fallback_plan is None:
                    raise
            else:
                return DispatchDecision(
                    plan=match.plan,
                    selector_revision=self.revision,
                    matched_path=match.path,
                    reason=match.reason,
                    evidence_ids=match.evidence_ids,
                    workspace_upper_bound_bytes=algorithm.workspace_bound(
                        context, match.plan.recipe
                    ),
                )
        if self._fallback_plan is None:
            raise ValueError(f"selector has no fallback: {failure}")
        algorithm = self._registry.validate(self._fallback_plan, context)
        return DispatchDecision(
            plan=self._fallback_plan,
            selector_revision=self.revision,
            matched_path=("fallback", self._fallback_plan.algorithm_id),
            reason=f"fallback to {self._fallback_plan.algorithm_id}: {failure}",
            workspace_upper_bound_bytes=algorithm.workspace_bound(
                context, self._fallback_plan.recipe
            ),
        )
