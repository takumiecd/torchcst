"""Required inputs, metadata routing and state-free/atom-backed execution boundaries."""

from dataclasses import dataclass, replace

import pytest
import torch

from torchcst import (
    Algorithm,
    DefaultRecipe,
    Dispatcher,
    ExecutionPlan,
    FixedSelector,
    LinearBinding,
    LinearInputs,
    Registry,
    SupportResult,
)
from torchcst._backends.catalog import REGISTRY
from torchcst._backends.dispatch import Match, Selector


@dataclass(frozen=True)
class Inputs:
    values: torch.Tensor
    scale: float = 1


@dataclass(frozen=True)
class Context:
    rows: int
    operation_id: str = "unrelated"
    workspace_limit_bytes: int | None = None


class Binding:
    input_type = Inputs

    def __init__(self):
        self.states = []
        self.limit = None
        self.offset = 0

    def validate_inputs(self, inputs):
        if not isinstance(inputs.values, torch.Tensor) or inputs.values.ndim != 1:
            raise ValueError("values must be a vector")
        if type(inputs.scale) not in (int, float):
            raise TypeError("scale must be numeric")

    def build_context(self, inputs):
        return Context(inputs.values.numel(), workspace_limit_bytes=self.limit)

    def state_signature(self):
        return self.offset

    def algorithm_state(self, algorithm, *, recipe):
        for state in self.states:
            if state.algorithm is algorithm and state.recipe == recipe:
                return state
        state = algorithm.create_state(self, recipe=recipe)
        self.states.append(state)
        return state


class Add(Algorithm):
    def __init__(self, name="add", *, scratch=0, fail=None):
        super().__init__(name, "v1", "unrelated", "add-v1", DefaultRecipe, Inputs)
        object.__setattr__(self, "scratch", scratch)
        object.__setattr__(self, "fail", fail)
        object.__setattr__(self, "preparations", 0)

    def validate_recipe(self, recipe):
        if type(recipe) is not DefaultRecipe:
            raise ValueError("invalid recipe")

    def supports(self, context, recipe):
        if self.fail == "supports":
            raise ValueError("broken support implementation")
        return SupportResult()

    def workspace_bound(self, context, recipe):
        return self.scratch

    def prepare(self, state):
        object.__setattr__(self, "preparations", self.preparations + 1)
        if self.fail == "prepare":
            raise RuntimeError("prepare failed")
        super().prepare(state)

    def execute(self, state, inputs):
        if self.fail == "execute":
            raise RuntimeError("execute failed")
        return inputs.values * inputs.scale + state.binding.offset


def setup(*, scratch=0, fail=None):
    registry = Registry()
    candidate, fallback = Add("fast", scratch=scratch, fail=fail), Add("torch")
    registry.register(candidate)
    registry.register(fallback)
    plans = [
        ExecutionPlan(a.id, a.revision, DefaultRecipe()) for a in (candidate, fallback)
    ]
    selector = FixedSelector(plans[0], registry=registry, fallback_plan=plans[1])
    return Dispatcher(selector=selector), Binding(), candidate, fallback, plans


def test_typed_inputs_are_required_optional_and_not_global_linear_fields():
    dispatcher, binding, _, _, _ = setup()
    with pytest.raises(TypeError):
        Inputs()
    with pytest.raises(TypeError):
        Inputs(torch.ones(2), unknown=1)
    with pytest.raises(TypeError, match="input"):
        dispatcher.run(binding, {"values": torch.ones(2)})
    with pytest.raises(ValueError, match="vector"):
        dispatcher.run(binding, Inputs(torch.ones(2, 2)))
    assert not binding.states
    x = torch.ones(2, requires_grad=True)
    first = dispatcher.run(binding, Inputs(x))
    second = dispatcher.run(binding, Inputs(x, scale=3))
    (first + second).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 4))
    assert binding.states[0].atom_state is None
    assert not hasattr(binding.states[0].context, "validate_inputs")


def test_fallback_only_before_preparation_and_forced_plan_never_falls_back():
    dispatcher, binding, candidate, fallback, plans = setup(scratch=128)
    binding.limit = 0
    x = Inputs(torch.ones(2))
    torch.testing.assert_close(dispatcher.run(binding, x), x.values)
    assert candidate.preparations == 0 and fallback.preparations == 1
    with pytest.raises(ValueError, match="workspace limit"):
        dispatcher.run(binding, x, plan=plans[0])
    assert candidate.preparations == 0


@pytest.mark.parametrize("fail", ["prepare", "execute", "supports"])
def test_failures_are_not_retried_as_another_algorithm(fail):
    dispatcher, binding, _, fallback, _ = setup(fail=fail)
    with pytest.raises((ValueError, RuntimeError), match="failed|broken"):
        dispatcher.run(binding, Inputs(torch.ones(2)))
    assert fallback.preparations == 0


def test_unknown_plan_invalid_workspace_and_wrong_operation_are_errors():
    dispatcher, binding, _, fallback, plans = setup()
    with pytest.raises(ValueError, match="unknown"):
        dispatcher.run(
            binding,
            Inputs(torch.ones(2)),
            plan=replace(plans[0], algorithm_revision="missing"),
        )
    binding.limit = -1
    with pytest.raises(ValueError, match="nonnegative"):
        dispatcher.run(binding, Inputs(torch.ones(2)))
    assert fallback.preparations == 0
    with pytest.raises(ValueError, match="contract"):
        dispatcher.select(Context(2, operation_id="linear"))


def test_invalid_selector_candidate_is_not_hidden_by_fallback():
    dispatcher, binding, _, fallback, plans = setup()

    class Broken(Selector):
        def _match(self, context):
            return Match(replace(plans[0], algorithm_revision="missing"), (), "broken")

    broken = Dispatcher(
        selector=Broken(
            revision="broken", registry=dispatcher.registry, fallback_plan=plans[1]
        )
    )
    with pytest.raises(ValueError, match="unknown"):
        broken.run(binding, Inputs(torch.ones(2)))
    assert not binding.states and fallback.preparations == 0


def test_binding_context_and_configuration_changes_refresh_reused_state():
    dispatcher, binding, candidate, _, _ = setup()
    dispatcher.run(binding, Inputs(torch.ones(2)))
    first = binding.states[0]
    dispatcher.run(binding, Inputs(torch.zeros(2)))
    assert candidate.preparations == 1
    dispatcher.run(binding, Inputs(torch.ones(3)))
    assert candidate.preparations == 2 and binding.states[0] is first
    binding.offset = 5
    torch.testing.assert_close(
        dispatcher.run(binding, Inputs(torch.ones(3))), torch.full((3,), 6.0)
    )
    assert candidate.preparations == 3
    other = Binding()
    dispatcher.run(other, Inputs(torch.ones(3)))
    assert other.states[0] is not first


def test_mismatched_registry_or_reused_state_is_rejected():
    dispatcher, binding, _, _, plans = setup()
    with pytest.raises(ValueError, match="registries"):
        Dispatcher(selector=dispatcher.selector, registry=Registry())
    dispatcher.run(binding, Inputs(torch.ones(2)))
    other = Binding()
    other.states = binding.states
    with pytest.raises(ValueError, match="different binding"):
        dispatcher.run(other, Inputs(torch.ones(2)))
    state = binding.states[0]
    state.recipe = object()
    binding.algorithm_state = lambda algorithm, recipe: state
    with pytest.raises(ValueError, match="recipe"):
        dispatcher.run(binding, Inputs(torch.ones(2)), plan=plans[0])


def test_live_operator_direct_binding_preserves_owner_and_gradients():
    from test_cst_optimizer import site

    model = site()
    binding = LinearBinding(model.operator)
    assert binding.atom_state is model.atom_state
    plan = ExecutionPlan("torch_materialized", "v1", DefaultRecipe())
    x = torch.randn(2, model.in_features, dtype=model.atoms.p.dtype, requires_grad=True)
    y = Dispatcher(registry=REGISTRY).run(binding, LinearInputs(x), plan=plan)
    expected = model.operator.apply(x)
    torch.testing.assert_close(y, expected)
    dy = torch.randn_like(y)
    a = torch.autograd.grad(y, (x, model.atoms.p), dy)
    b = torch.autograd.grad(expected, (x, model.atoms.p), dy)
    for actual, truth in zip(a, b):
        torch.testing.assert_close(actual, truth)
    with pytest.raises(ValueError, match="owner"):
        LinearBinding(model.operator, model.atoms.p.clone())
