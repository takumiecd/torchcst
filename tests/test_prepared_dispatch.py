"""Fixed preparation is bounded, binding-owned, and never certifies new inputs."""

import copy
import pickle
from dataclasses import dataclass

import pytest
import torch
from test_cst_optimizer import site
from test_dispatcher import Add, Binding, Inputs

from torchcst import (
    CSTOptimizer,
    DefaultRecipe,
    Dispatcher,
    ExecutionPlan,
    FixedSelector,
    LinearBinding,
    LinearInputs,
    Registry,
)


class PreparedBinding(Binding):
    def prepared_dispatch_signature(self):
        return self.offset


class CountingAdd(Add):
    def __init__(self, name="add"):
        super().__init__(name)
        object.__setattr__(self, "support_calls", 0)

    def supports(self, context, recipe):
        object.__setattr__(self, "support_calls", self.support_calls + 1)
        return super().supports(context, recipe)


def setup():
    registry = Registry()
    a, b = CountingAdd(), CountingAdd("second")
    registry.register(a)
    registry.register(b)
    plan = ExecutionPlan(a.id, a.revision, DefaultRecipe())
    selector = FixedSelector(plan, registry=registry)
    return Dispatcher(selector=selector), PreparedBinding(), a, b, selector


def test_fixed_hot_support_zero_preflight_forced_reuse_and_fresh_inputs():
    dispatcher, binding, algorithm, _, _ = setup()
    inputs = Inputs(torch.ones(2))
    plan = dispatcher.prepare(binding, inputs).plan
    assert algorithm.support_calls == 1
    for _ in range(4):
        torch.testing.assert_close(
            dispatcher.run(binding, inputs, plan=plan), inputs.values
        )
        dispatcher.prepare(binding, inputs)
    assert algorithm.support_calls == 1
    assert binding._prepared_dispatch_stats == {"hits": 8, "misses": 1}
    assert set(vars(dispatcher)) == {"registry", "selector"}
    with pytest.raises(ValueError, match="vector"):
        dispatcher.run(binding, Inputs(torch.ones(2, 2)), plan=plan)
    assert algorithm.support_calls == 1
    binding.offset = 3
    torch.testing.assert_close(dispatcher.run(binding, inputs), inputs.values + 3)
    assert algorithm.support_calls == 2
    assert binding._prepared_dispatch_stats["misses"] == 2


def test_context_workspace_selector_and_parameterless_owner_changes_reprepare():
    dispatcher, binding, a, b, selector = setup()
    dispatcher.run(binding, Inputs(torch.ones(2)))
    dispatcher.run(binding, Inputs(torch.ones(3)))
    assert a.support_calls == 2
    binding.limit = 0
    dispatcher.run(binding, Inputs(torch.ones(3)))
    assert a.support_calls == 3
    selector._plan = ExecutionPlan(b.id, b.revision, DefaultRecipe())
    dispatcher.run(binding, Inputs(torch.ones(3)))
    assert b.support_calls == 1
    assert binding._prepared_dispatch.algorithm is b
    assert len(binding.__dict__["_prepared_dispatch_stats"]) == 2


def test_custom_fixed_selector_subclass_is_called_every_time():
    dispatcher, binding, algorithm, _, selector = setup()

    class Custom(FixedSelector):
        def _match(self, context):
            self.calls += 1
            return super()._match(context)

    custom = Custom(selector._plan, registry=selector.registry)
    custom.calls = 0
    dispatcher = Dispatcher(selector=custom)
    for _ in range(3):
        dispatcher.run(binding, Inputs(torch.ones(2)))
    assert custom.calls == algorithm.support_calls == 3
    assert "_prepared_dispatch" not in vars(binding)


def test_mutable_context_is_not_prepared():
    @dataclass
    class Mutable:
        rows: int = 2
        operation_id: str = "unrelated"
        workspace_limit_bytes: int | None = None

    dispatcher, binding, algorithm, _, _ = setup()
    context = Mutable()
    binding.build_context = lambda _: context
    dispatcher.run(binding, Inputs(torch.ones(2)))
    context.rows = 7
    dispatcher.run(binding, Inputs(torch.ones(2)))
    assert algorithm.support_calls == 2
    assert "_prepared_dispatch" not in vars(binding)


def test_cached_state_wrong_owner_is_rejected():
    dispatcher, binding, _, _, _ = setup()
    dispatcher.run(binding, Inputs(torch.ones(2)))
    binding._prepared_dispatch.state.binding = PreparedBinding()
    with pytest.raises(ValueError, match="different binding"):
        dispatcher.run(binding, Inputs(torch.ones(2)))


@pytest.mark.parametrize("binding_kind", ["module", "direct", "update"])
def test_builtin_caches_clear_on_deepcopy_and_observe_live_mutations(binding_kind):
    from torchcst import AtomUpdateBinding, AtomUpdateInputs
    from torchcst._backends.catalog import get_registry
    from torchcst._backends.torch.algorithms.atom_update.plans import REFERENCE

    model = site()
    if binding_kind == "update":
        binding = AtomUpdateBinding(model.operator)
        plan = REFERENCE
        inputs = AtomUpdateInputs(model.atoms.p.detach().clone(), 0.001)
    else:
        binding = model if binding_kind == "module" else LinearBinding(model.operator)
        plan = ExecutionPlan("torch_materialized", "v1", DefaultRecipe())
        inputs = LinearInputs(
            torch.ones(2, model.in_features, dtype=model.atoms.p.dtype)
        )
    dispatcher = Dispatcher(registry=get_registry())
    dispatcher.prepare(binding, inputs, plan=plan)
    dispatcher.prepare(binding, inputs, plan=plan)
    assert binding._prepared_dispatch_stats == {"hits": 1, "misses": 1}
    with torch.no_grad():
        model.atoms.p.add_(0.001)
    dispatcher.prepare(binding, inputs, plan=plan)
    assert binding._prepared_dispatch_stats["hits"] == 2
    for clone in (copy.deepcopy(binding), pickle.loads(pickle.dumps(binding))):
        assert "_prepared_dispatch" not in vars(clone)
        assert "_prepared_dispatch_stats" not in vars(clone)
    # Even an equal-valued live config write changes its version token.
    scalar = next(model.kernel.buffers())
    scalar.add_(0)
    dispatcher.prepare(binding, inputs, plan=plan)
    assert binding._prepared_dispatch_stats["misses"] == 2


def test_optimizer_hooks_keep_fresh_update_boundary():
    model = site()
    base = torch.optim.AdamW(model.parameters(), lr=0.001)
    optimizer = CSTOptimizer(base, model=model, check_finite=False)
    model.atoms.p.grad = torch.ones_like(model.atoms.p)
    optimizer.step()
    binding = optimizer._update_bindings[model]
    first = binding._prepared_dispatch_stats.copy()

    def mutate_config(*_):
        value = next(model.kernel.buffers())
        value.add_(0)

    handle = base.register_step_pre_hook(mutate_config)
    try:
        model.atoms.p.grad = torch.ones_like(model.atoms.p)
        optimizer.step()
    finally:
        handle.remove()
    assert binding._prepared_dispatch_stats["misses"] == first["misses"] + 1
    assert binding._prepared_dispatch_stats["hits"] == first["hits"] + 1


def test_linear_context_flags_storage_and_layout_are_fresh():
    from torchcst._backends.catalog import get_registry

    model = site()
    dispatcher = Dispatcher(registry=get_registry())
    plan = ExecutionPlan("torch_materialized", "v1", DefaultRecipe())
    x = torch.ones(2, model.in_features, dtype=model.atoms.p.dtype)

    def prepare(value=x):
        return dispatcher.prepare(model, LinearInputs(value), plan=plan)

    prepare()
    prepare(x.requires_grad_())
    assert model._prepared_dispatch_stats["misses"] == 2
    with torch.no_grad():
        prepare()
    assert model._prepared_dispatch_stats["misses"] == 3
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(not previous)
        prepare()
    finally:
        torch.use_deterministic_algorithms(previous)
    assert model._prepared_dispatch_stats["misses"] == 4
    strided = torch.ones(2, model.in_features * 2, dtype=x.dtype)[:, ::2]
    prepare(strided)
    assert model._prepared_dispatch_stats["misses"] == 5
    # Same numerical values and shape, different authoritative storage.
    with torch.no_grad():
        model.atoms.p.set_(model.atoms.p.detach().clone())
    prepare(strided)
    assert model._prepared_dispatch_stats["misses"] == 6
    model.atom_state.relayout(torch.tensor([2, 0, 1]))
    prepare(strided)
    assert model._prepared_dispatch_stats["misses"] == 7


def test_prepared_cache_does_not_add_configuration_signature_walks(monkeypatch):
    from torchcst._backends.catalog import get_registry
    from torchcst.operators import metadata

    model = site()
    plan = ExecutionPlan("torch_materialized", "v1", DefaultRecipe())
    dispatcher = Dispatcher(registry=get_registry())
    inputs = LinearInputs(torch.ones(2, model.in_features, dtype=model.atoms.p.dtype))
    original = metadata.fixed_state_signature
    calls = []

    def signature(roots):
        calls.append(roots)
        return original(roots)

    monkeypatch.setattr(metadata, "fixed_state_signature", signature)
    dispatcher.prepare(model, inputs, plan=plan)
    calls.clear()
    dispatcher.prepare(model, inputs, plan=plan)
    # validate_inputs and build_context each refresh live declaration. The
    # prepared key borrows that signature instead of traversing a third time.
    assert len(calls) == 2
    assert model._prepared_dispatch_stats["hits"] == 1


def test_optimizer_numerical_steps_have_zero_hot_support_checks(monkeypatch):
    from torchcst._backends.catalog import get_registry
    from torchcst._backends.torch.algorithms.atom_update.plans import REFERENCE

    model = site()
    optimizer = CSTOptimizer(
        torch.optim.AdamW(model.parameters()), model=model, check_finite=False
    )
    algorithm = get_registry().validate_plan(REFERENCE)
    original = type(algorithm).supports
    calls = []

    def supports(self, context, recipe):
        calls.append(context)
        return original(self, context, recipe)

    monkeypatch.setattr(type(algorithm), "supports", supports)
    for _ in range(3):
        model.atoms.p.grad = torch.ones_like(model.atoms.p)
        optimizer.step()
    assert len(calls) == 1
    assert optimizer._update_bindings[model]._prepared_dispatch_stats == {
        "hits": 5,
        "misses": 1,
    }


class PretendsEqual:
    def __eq__(self, other):
        return True


def test_warm_cache_rejects_equal_impostor_plan_and_recipe():
    dispatcher, binding, algorithm, _, selector = setup()
    inputs = Inputs(torch.ones(2))
    dispatcher.run(binding, inputs)
    with pytest.raises(TypeError, match="ExecutionPlan"):
        dispatcher.run(binding, inputs, plan=PretendsEqual())
    bad_recipe = ExecutionPlan(algorithm.id, algorithm.revision, PretendsEqual())
    with pytest.raises(TypeError, match="recipe type"):
        dispatcher.run(binding, inputs, plan=bad_recipe)
    selector._plan = PretendsEqual()
    with pytest.raises(TypeError, match="ExecutionPlan"):
        dispatcher.run(binding, inputs)


def test_equivalent_new_plan_is_cold_then_safely_warm():
    dispatcher, binding, algorithm, _, selector = setup()
    inputs = Inputs(torch.ones(2))
    dispatcher.run(binding, inputs)
    new = ExecutionPlan(algorithm.id, algorithm.revision, DefaultRecipe())
    assert new == selector._plan and new is not selector._plan
    dispatcher.run(binding, inputs, plan=new)
    assert algorithm.support_calls == 2
    dispatcher.run(binding, inputs, plan=new)
    assert algorithm.support_calls == 2
    selector._plan = ExecutionPlan(algorithm.id, algorithm.revision, DefaultRecipe())
    dispatcher.run(binding, inputs)
    assert algorithm.support_calls == 3


@pytest.mark.parametrize("corruption", ["schema", "algorithm_id", "recipe"])
@pytest.mark.parametrize("forced", [False, True])
def test_cached_plan_declaration_is_revalidated_after_corruption(corruption, forced):
    dispatcher, binding, _, _, selector = setup()
    plan = selector._plan
    inputs = Inputs(torch.ones(2))
    dispatcher.run(binding, inputs, plan=plan if forced else None)
    field, value = {
        "schema": ("schema_version", 2),
        "algorithm_id": ("algorithm_id", "unknown"),
        "recipe": ("recipe", PretendsEqual()),
    }[corruption]
    object.__setattr__(plan, field, value)
    with pytest.raises((ValueError, TypeError)):
        dispatcher.run(binding, inputs, plan=plan if forced else None)


def test_cached_state_recipe_equal_impostor_is_rejected():
    dispatcher, binding, _, _, _ = setup()
    inputs = Inputs(torch.ones(2))
    dispatcher.run(binding, inputs)
    binding._prepared_dispatch.state.recipe = PretendsEqual()
    with pytest.raises(ValueError, match="recipe identity"):
        dispatcher.run(binding, inputs)


def test_mutable_recipe_never_certifies_later_support():
    @dataclass
    class MutableRecipe:
        value: int = 1

    class MutableAlgorithm(CountingAdd):
        def __init__(self):
            super().__init__()
            object.__setattr__(self, "recipe_type", MutableRecipe)

        def validate_recipe(self, recipe):
            if type(recipe) is not MutableRecipe:
                raise TypeError("expected MutableRecipe")

    registry = Registry()
    algorithm = MutableAlgorithm()
    registry.register(algorithm)
    recipe = MutableRecipe()
    plan = ExecutionPlan(algorithm.id, algorithm.revision, recipe)
    dispatcher = Dispatcher(registry=registry)
    binding = PreparedBinding()
    dispatcher.run(binding, Inputs(torch.ones(2)), plan=plan)
    recipe.value = 7
    dispatcher.run(binding, Inputs(torch.ones(2)), plan=plan)
    assert algorithm.support_calls == 2
    assert "_prepared_dispatch" not in vars(binding)
