"""Radius hierarchy dependencies exclude optimizer-only storage, not placement."""

import copy
from dataclasses import replace

import pytest
import torch
from test_radius_polar_cuda_update import radius_model

from torchcst import CSTOptimizer, LinearBinding, LinearInputs
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    RadiusCellMassAlgorithm,
    RadiusCellMassState,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.recipe import (
    RadiusCellMassRecipe,
)
from torchcst._backends.state import AlgorithmState


def setup(kind="module", algorithm=None):
    model = radius_model()
    binding = model if kind == "module" else LinearBinding(model.operator)
    algorithm = algorithm or RadiusCellMassAlgorithm()
    state = binding.algorithm_state(algorithm, recipe=RadiusCellMassRecipe())
    state.context = binding.build_context(LinearInputs(torch.zeros(4, 16)))
    algorithm.prepare(state)
    assert state.is_current()
    return model, binding, algorithm, state


@pytest.mark.parametrize("kind", ["module", "direct"])
def test_optimizer_reload_and_storage_replacement_do_not_rebuild_hierarchy(kind):
    model, _, algorithm, state = setup(kind)
    optimizer = CSTOptimizer(torch.optim.AdamW(model.parameters()), model=model)
    model.atoms.p.grad = torch.ones_like(model.atoms.p)
    optimizer.step()
    assert state.is_current()
    hierarchy = state.layout
    fields = model.atom_state.optimizer_state.fields
    previous = fields["exp_avg"]
    optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert model.atom_state.optimizer_state.fields["exp_avg"] is not previous
    assert state.is_current()
    assert state.layout is hierarchy
    # The generic lifecycle still tracks optimizer storage for algorithms that
    # may use it. Only this concrete radius implementation declares otherwise.
    generic = AlgorithmState(state.binding, algorithm, recipe=state.recipe)
    generic.context = state.context
    generic.mark_current()
    fields = model.atom_state.optimizer_state.fields
    fields["exp_avg"] = fields["exp_avg"].clone()
    assert not generic.is_current()
    assert state.is_current()


@pytest.mark.parametrize("kind", ["module", "direct"])
@pytest.mark.parametrize(
    "change", ["storage", "replacement", "dtype", "relayout", "input_shape", "config"]
)
def test_parameter_and_context_dependencies_remain_live(kind, change):
    model, binding, _, state = setup(kind)
    if change == "storage":
        with torch.no_grad():
            model.atoms.p.set_(model.atoms.p.detach().clone())
    elif change == "replacement":
        model.atoms.p = torch.nn.Parameter(model.atoms.p.detach().clone())
    elif change == "dtype":
        model.double()
    elif change == "relayout":
        model.atom_state.relayout(torch.arange(model.atom_count - 1, -1, -1))
    else:
        if change == "config":
            model.kernel.scalar("activity_gain").mul_(0.9)
        x = torch.zeros(7 if change == "input_shape" else 4, 16)
        state.context = binding.build_context(LinearInputs(x))
    assert not state.is_current()


@pytest.mark.parametrize("change", ["radius", "chart"])
def test_invalid_configuration_still_rejects_before_state_execution(change):
    model, binding, _, state = setup()
    if change == "radius":
        model.kernel.scalar("radius_min_input").fill_(-1)
    else:
        model.input_chart.geometry.periods.fill_(0)
    with pytest.raises(ValueError):
        binding.build_context(LinearInputs(torch.zeros(4, 16)))
    assert isinstance(state, RadiusCellMassState)


def test_custom_state_signature_callback_is_not_skipped(monkeypatch):
    _, binding, algorithm, state = setup()
    calls = []
    value = [0]

    def signature():
        calls.append(value[0])
        return value[0]

    monkeypatch.setattr(binding, "state_signature", signature, raising=False)
    algorithm.prepare(state)
    assert state.is_current()
    assert calls == [0, 0]
    value[0] = 1
    assert not state.is_current()
    assert calls == [0, 0, 1]


def test_custom_algorithm_subclass_keeps_generic_optimizer_dependencies():
    class Custom(RadiusCellMassAlgorithm):
        pass

    _, _, _, state = setup(algorithm=Custom())
    assert type(state) is AlgorithmState


def test_context_recipe_and_serialization_keep_existing_lifecycle():
    model, _, algorithm, state = setup()
    context = state.context
    state.context = replace(context, deterministic=True)
    assert not state.is_current()
    algorithm.prepare(state)
    assert state.is_current()
    clone = copy.deepcopy(model)
    assert "_algorithm_states" not in vars(clone)
    with pytest.raises(ValueError):
        algorithm.create_state(model, recipe=replace(state.recipe, save_response=False))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("needs_x,needs_p", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("cached", [False, True])
def test_real_cuda_execution_keeps_retained_read_leases(needs_x, needs_p, cached):
    from torchcst import FixedSelector
    from torchcst._backends.catalog import get_registry
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
        BLOCK_LOCAL_CACHED_PARALLEL,
        BLOCK_LOCAL_PARALLEL,
    )
    from torchcst.atoms.optimizer_state import OptimizerFieldSpec

    model = radius_model().cuda()
    model.atoms.p.requires_grad_(needs_p)
    plan = BLOCK_LOCAL_CACHED_PARALLEL if cached else BLOCK_LOCAL_PARALLEL
    model.selector = FixedSelector(plan, registry=get_registry())
    x = torch.randn(3, 16, device="cuda", requires_grad=needs_x)
    y = model(x)
    state = model._prepared_dispatch.state
    assert type(state) is RadiusCellMassState
    owner = model.atom_state
    owner.optimizer_state.declare(
        "test_moment", torch.zeros_like(model.atoms.p), OptimizerFieldSpec(atom_axis=0)
    )
    owner.optimizer_state.fields["test_moment"] = torch.zeros_like(model.atoms.p)
    assert state.is_current()  # Moment storage does not describe the hierarchy.
    selected = tuple(t for t in (x, model.atoms.p) if t.requires_grad)
    for retain in (True, False):
        grads = torch.autograd.grad(y.sum(), selected, retain_graph=retain)
        assert all(torch.isfinite(g).all() for g in grads)
        if retain:
            with pytest.raises(RuntimeError, match="outstanding|retained"):
                owner.relayout(torch.arange(model.atom_count - 1, -1, -1))
    owner.relayout(torch.arange(model.atom_count - 1, -1, -1))
    assert not state.is_current()
