"""One registry handles Linear and unrelated operation-specific input interfaces."""

from dataclasses import dataclass, replace

import pytest
import torch

from torchcst import Atoms, AtomState, Dispatcher
from torchcst._backends.algorithm import Algorithm
from torchcst._backends.catalog import REGISTRY
from torchcst._backends.dispatch import (
    ExactEntry,
    ExactSelector,
    FixedSelector,
    OrderedSelector,
    load_selector,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DefaultRecipe, ExecutionPlan, SupportResult


@dataclass(frozen=True)
class DecodeContext:
    operation_id: str = "decode"
    width: int = 2
    workspace_limit_bytes: int | None = None


@dataclass(frozen=True)
class DecodeInputs:
    scale: float = 1


class DecodeBinding:
    input_type = DecodeInputs

    def __init__(self, owner):
        self.atom_state = owner
        self._states = []

    def validate_inputs(self, inputs):
        if self.atom_state.atoms.p.shape[-1] != 2:
            raise ValueError("coordinate width differs from decode contract")

    def build_context(self, inputs):
        return DecodeContext(width=self.atom_state.atoms.p.shape[-1])

    def algorithm_state(self, algorithm, *, recipe):
        for state in self._states:
            if state.algorithm is algorithm and state.recipe == recipe:
                return state
        state = algorithm.create_state(self, recipe=recipe)
        self._states.append(state)
        return state


@dataclass(frozen=True)
class DecodeAlgorithm(Algorithm[DefaultRecipe]):
    id: str = "test_decode"
    revision: str = "v1"
    operation_id: str = "decode"
    semantics_id: str = "polar-decode-test-v1"
    recipe_type: type = DefaultRecipe

    input_type: type = DecodeInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not DefaultRecipe:
            raise ValueError("unsupported recipe")

    def supports(self, context, recipe):
        return (
            SupportResult()
            if isinstance(context, DecodeContext) and context.width == 2
            else SupportResult(("requires decode metadata",))
        )

    def workspace_bound(self, context, recipe):
        return 0

    def execute(self, state, inputs):
        coordinates = state.atom_state.parameters_for_execution()
        # This operation has no Linear input, matrix shape or OperatorSpec.
        radius, angle = coordinates.unbind(-1)
        return (
            torch.stack((radius * angle.cos(), radius * angle.sin()), -1) * inputs.scale
        )


def test_non_linear_algorithm_selection_serialization_state_and_live_values():
    registry = Registry()
    algorithm = DecodeAlgorithm()
    registry.register(algorithm)
    plan = ExecutionPlan(algorithm.id, algorithm.revision, DefaultRecipe())
    restored = registry.loads_plan(registry.dumps_plan(plan))
    selector = OrderedSelector((restored,), registry=registry)
    owner = AtomState.for_atoms(Atoms(torch.tensor([[2.0, 0.0], [3.0, torch.pi / 2]])))
    binding = DecodeBinding(owner)
    state = binding.algorithm_state(algorithm, recipe=plan.recipe)
    context = DecodeContext()
    assert selector.select(context).plan == plan
    exact = ExactSelector.from_entries(
        (ExactEntry(context, plan, ("decode-fixture",)),),
        registry=registry,
        fallback_plan=plan,
        revision="decode-exact-v1",
        score_policy={"id": "fixture", "revision": "v1", "parameters": {}},
        dataset_snapshot="decode-metadata-only",
    )
    loaded = load_selector(exact.dumps(), registry=registry)
    assert loaded.select(context).evidence_ids == ("decode-fixture",)
    assert loaded.select(replace(context, workspace_limit_bytes=0)).evidence_ids == ()
    first = Dispatcher(registry=registry).run(binding, DecodeInputs(), plan=plan)
    torch.testing.assert_close(
        first.detach(), torch.tensor([[2.0, 0.0], [0.0, 3.0]]), atol=1e-6, rtol=0
    )
    first.sum().backward()
    owner.relayout(torch.tensor([1, 0]))
    assert not state.is_current()
    with torch.no_grad():
        owner.atoms.p[:, 0].add_(1)
    second = Dispatcher(registry=registry).run(binding, DecodeInputs(), plan=plan)
    torch.testing.assert_close(
        second.detach(), torch.tensor([[0.0, 4.0], [3.0, 0.0]]), atol=1e-6, rtol=0
    )
    second.sum().backward()
    assert state.is_current()
    with pytest.raises(ValueError, match="coordinate width"):
        bad = DecodeBinding(AtomState.for_atoms(Atoms(torch.ones(3, 5))))
        Dispatcher(registry=registry).run(bad, DecodeInputs(), plan=plan)
    with pytest.raises(ValueError, match="contract"):
        FixedSelector(plan, registry=registry).select(
            replace(context, operation_id="optimizer")
        )
    with pytest.raises(ValueError, match="workspace limit"):
        Dispatcher(registry=registry).select(
            replace(context, workspace_limit_bytes=-1), plan=plan
        )


def test_builtin_torch_and_cuda_share_the_same_registry_contract():
    for name in (
        "torch_materialized",
        "torch_factored",
        "torch_tiled",
        "torch_normalized_radial",
        "normalized_full",
        "normalized_window",
        "cuda_strip_torus_fused",
    ):
        algorithm = REGISTRY.get(name, revision="v1")
        assert isinstance(algorithm, Algorithm)
        plan = ExecutionPlan(name, "v1", algorithm.recipe_type())
        assert REGISTRY.loads_plan(REGISTRY.dumps_plan(plan)) == plan


def test_linear_can_select_a_torch_plan_through_the_common_selector():
    from test_cst_optimizer import site

    model = site()
    x = torch.randn(2, model.in_features, dtype=model.atoms.p.dtype, requires_grad=True)
    plan = ExecutionPlan("torch_materialized", "v1", DefaultRecipe())
    model.selector = FixedSelector(plan, registry=REGISTRY)
    actual = model(x)
    expected = model.operator.apply(x)
    dy = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (x, model.atoms.p), dy)
    expected_grads = torch.autograd.grad(expected, (x, model.atoms.p), dy)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b)
    assert model.algorithm_state(
        REGISTRY.get(plan.algorithm_id, revision="v1"),
        recipe=plan.recipe,
        operator=model.execution_declaration(),
    ).is_current()


def test_linear_uses_the_registry_owned_by_its_selector():
    from test_cst_optimizer import site

    from torchcst._backends.torch.algorithms.materialized import MaterializedAlgorithm

    registry = Registry()
    algorithm = MaterializedAlgorithm(id="custom_materialized")
    registry.register(algorithm)
    plan = ExecutionPlan(algorithm.id, algorithm.revision, DefaultRecipe())
    model = site()
    model.selector = FixedSelector(plan, registry=registry)
    x = torch.randn(2, model.in_features, dtype=model.atoms.p.dtype)
    torch.testing.assert_close(model(x), model.operator.apply(x))
