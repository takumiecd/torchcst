"""Invocation orchestration; the binding owns all persistent execution state."""

from dataclasses import is_dataclass
from typing import Protocol

from torchcst._backends.schema import Context, DispatchDecision
from torchcst._builtin_callbacks import seal_functions as _seal_functions

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

    def _prepare(self, binding, inputs, *, plan=None):
        # An earlier Context never proves this invocation's inputs are valid.
        if type(inputs) is not binding.input_type:
            raise TypeError("inputs type differs from binding contract")
        binding.validate_inputs(inputs)
        context = binding.build_context(inputs)
        from .prepared import PreparedDispatch
        from .select import FixedSelector

        # Declaration validation is cheap and is never certified by equality
        # with a cached Plan (which could invoke attacker-defined __eq__).
        if plan is not None:
            self.registry.validate_plan(plan)
        signature = getattr(binding, "prepared_dispatch_signature", None)
        fixed = type(self.selector) is FixedSelector
        if plan is None and fixed:
            self.registry.validate_plan(self.selector._plan)
            if self.selector.fallback_plan is not None:
                self.registry.validate_plan(self.selector.fallback_plan)
        plans = (
            (plan,)
            if plan is not None
            else ((self.selector._plan, self.selector.fallback_plan) if fixed else ())
        )
        immutable_recipes = all(
            is_dataclass(p.recipe) and type(p.recipe).__dataclass_params__.frozen
            for p in plans
            if p is not None
        )
        eligible = (
            signature is not None
            and immutable_recipes
            and isinstance(getattr(binding, "__dict__", None), dict)
            and is_dataclass(context)
            and type(context).__dataclass_params__.frozen
            and (plan is not None or fixed)
        )
        routing = (
            ("forced", plan)
            if plan is not None
            else (
                "fixed",
                self.selector.revision,
                self.selector._plan,
                self.selector.fallback_plan,
            )
            if fixed
            else None
        )
        placement = signature() if eligible else None
        cached = binding.__dict__.get("_prepared_dispatch") if eligible else None
        if eligible:
            stats = binding.__dict__.setdefault(
                "_prepared_dispatch_stats", {"hits": 0, "misses": 0}
            )
        if (
            cached is not None
            and cached.registry is self.registry
            and cached.recipe is cached.decision.plan.recipe
            and (cached.context is context or cached.context == context)
            and cached.placement == placement
            and (
                (
                    cached.routing[0] == routing[0]
                    and (
                        cached.routing[1] is routing[1]
                        if routing[0] == "forced"
                        else cached.routing[1] == routing[1]
                        and cached.routing[2] is routing[2]
                        and cached.routing[3] is routing[3]
                    )
                )
                or (plan is not None and cached.decision.plan is plan)
            )
        ):
            algorithm = self.registry.validate_plan(cached.decision.plan)
            if algorithm is not cached.algorithm:
                raise ValueError("cached algorithm differs from current Registry")
            if (
                cached.state.binding is not binding
                or cached.state.algorithm is not cached.algorithm
            ):
                raise ValueError(
                    "algorithm state belongs to a different binding/algorithm"
                )
            if cached.state.recipe is not cached.state_recipe:
                raise ValueError("cached algorithm state recipe identity differs")
            stats["hits"] += 1
            # A forced execution may reuse preflight preparation while still
            # describing itself as forced rather than selector-chosen.
            decision = cached.decision
            if plan is not None:
                decision = DispatchDecision(
                    plan,
                    "forced-v1",
                    ("forced", plan.algorithm_id),
                    "explicit Plan",
                    workspace_upper_bound_bytes=decision.workspace_upper_bound_bytes,
                )
            cached.state.context = context
            return decision, cached.algorithm, cached.state
        if eligible:
            stats["misses"] += 1
        decision = self.select(context, plan=plan)
        algorithm = self.registry.validate_plan(decision.plan)
        if algorithm.input_type is not binding.input_type:
            raise TypeError("algorithm input type differs from binding contract")
        state = binding.algorithm_state(algorithm, recipe=decision.plan.recipe)
        if state.binding is not binding or state.algorithm is not algorithm:
            raise ValueError("algorithm state belongs to a different binding/algorithm")
        if (
            type(state.recipe) is not algorithm.recipe_type
            or state.recipe != decision.plan.recipe
        ):
            raise ValueError("algorithm state recipe differs")
        state.context = context
        if eligible:
            binding.__dict__["_prepared_dispatch"] = PreparedDispatch(
                self.registry,
                routing,
                context,
                placement,
                decision,
                algorithm,
                state,
                decision.plan.recipe,
                state.recipe,
            )
        return decision, algorithm, state

    def prepare(self, binding: ExecutionBinding, inputs, *, plan=None):
        """Validate current inputs/configuration and preflight without executing.

        Preparation is binding-owned and bounded to one fixed-Plan entry. A
        later run always validates fresh inputs and reconstructs live Context,
        including after optimizer hooks. Custom selectors are never cached.
        """
        return self._prepare(binding, inputs, plan=plan)[0]

    def run(self, binding: ExecutionBinding, inputs, *, plan=None):
        _, algorithm, state = self._prepare(binding, inputs, plan=plan)
        return algorithm.run(state, inputs)

    def prepare_optimizer_update(self, binding, adapter, site=None):
        """Ask an already-selected update Algorithm for its optional hot path.

        The binding owns the result. Custom/unsupported implementations keep
        the same prepare/run entrance; the Dispatcher stores no model state.
        """
        cached = binding.__dict__.get("_prepared_dispatch")
        if cached is None or cached.registry is not self.registry:
            return None
        factory = getattr(cached.algorithm, "prepare_optimizer_program", None)
        return None if factory is None else factory(self, cached, adapter, site)


# Optional prepared updates may specialize only these original definitions.

_seal_functions(globals())
