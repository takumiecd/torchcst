"""Plan selection that does not distinguish individual Algorithm families."""

from .base import Match, Selector


class FixedSelector(Selector):
    """An explicitly supplied Plan for bootstrap or direct comparisons."""

    def __init__(self, plan, *, registry, revision="fixed-v1", fallback_plan=None):
        super().__init__(
            revision=revision, registry=registry, fallback_plan=fallback_plan
        )
        registry.validate_plan(plan)
        self._plan = plan

    def _match(self, context):
        return Match(
            plan=self._plan,
            path=("fixed", self._plan.algorithm_id),
            reason="explicit Plan",
        )
