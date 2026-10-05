"""Invocation orchestration; the binding owns all persistent execution state."""

from typing import Protocol

from torchcst._backends.schema import Context, DispatchDecision

from .validation import UnsupportedPlan, validate_plan_context


class ExecutionBinding(Protocol):
    input_type: type

    def validate_inputs(self, inputs) -> None: ...

    def build_context(self, inputs) -> Context: ...

    def algorithm_state(self, algorithm, *, recipe): ...


def select_plan(selector, context):
    match = selector._match(context)
    failure = "unobserved condition"
    if match is not None:
        from .base import Match

        if not isinstance(match, Match):
            raise TypeError("selector must return a Match or None")
        try:
            _, bound = validate_plan_context(selector.registry, match.plan, context)
        except UnsupportedPlan as error:
            failure = str(error)
            if selector.fallback_plan is None:
                raise
        else:
            return DispatchDecision(
                match.plan,
                selector.revision,
                match.path,
                match.reason,
                match.evidence_ids,
                bound,
            )
    if selector.fallback_plan is None:
        raise UnsupportedPlan(f"selector has no fallback: {failure}")
    plan = selector.fallback_plan
    _, bound = validate_plan_context(selector.registry, plan, context)
    return DispatchDecision(
        plan,
        selector.revision,
        ("fallback", plan.algorithm_id),
        f"fallback to {plan.algorithm_id}: {failure}",
        workspace_upper_bound_bytes=bound,
    )


class Dispatcher:
    """One execution entrance, also used when a Plan is explicitly forced.

    Context-only selection is useful for declarations, not proof of input validity.
    No binding, tensor, mutable state or last-call decision is retained here.
    """

    def __init__(self, *, registry=None, selector=None):
        from torchcst._backends.registry import Registry

        if selector is not None:
            if registry is not None and registry is not selector.registry:
                raise ValueError("dispatcher and selector have different registries")
            registry = selector.registry
        if not isinstance(registry, Registry):
            raise TypeError("dispatcher requires a Registry")
        self.registry = registry
        self.selector = selector

    def select(self, context, *, plan=None):
        if plan is None:
            if self.selector is None:
                raise ValueError("dispatcher requires a Plan or Selector")
            return select_plan(self.selector, context)
        _, bound = validate_plan_context(self.registry, plan, context)
        return DispatchDecision(
            plan,
            "forced-v1",
            ("forced", plan.algorithm_id),
            "explicit Plan",
            workspace_upper_bound_bytes=bound,
        )

    def run(self, binding: ExecutionBinding, inputs, *, plan=None):
        if type(inputs) is not binding.input_type:
            raise TypeError("inputs type differs from binding contract")
        binding.validate_inputs(inputs)
        context = binding.build_context(inputs)
        decision = self.select(context, plan=plan)
        algorithm = self.registry.validate_plan(decision.plan)
        if algorithm.input_type is not binding.input_type:
            raise TypeError("algorithm input type differs from binding contract")
        state = binding.algorithm_state(algorithm, recipe=decision.plan.recipe)
        if state.binding is not binding or state.algorithm is not algorithm:
            raise ValueError("algorithm state belongs to a different binding/algorithm")
        if state.recipe != decision.plan.recipe:
            raise ValueError("algorithm state recipe differs")
        state.context = context
        return algorithm.run(state, inputs)
