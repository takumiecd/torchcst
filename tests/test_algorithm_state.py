import copy
from dataclasses import dataclass

import pytest
import torch
from test_cst_optimizer import site

from torchcst import CSTOptimizer
from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, SupportResult
from torchcst._backends.state import AlgorithmState


@dataclass(frozen=True)
class SumInputs:
    scale: float = 1


class SumAlgorithm(Algorithm):
    def __init__(self):
        super().__init__("test_sum", "v1", "sum", "sum-v1", DefaultRecipe, SumInputs)
        object.__setattr__(self, "preparations", 0)

    def validate_recipe(self, recipe):
        pass

    def supports(self, context, recipe):
        return SupportResult()

    def workspace_bound(self, context, recipe):
        return 0

    def prepare(self, state):
        object.__setattr__(self, "preparations", self.preparations + 1)
        state.invalidate()
        state.layout = state.atom_state.row_to_id
        state.mark_current()

    def execute(self, state, inputs):
        return state.atom_state.parameters_for_execution().sum() * inputs.scale


def test_algorithm_rebuilds_only_placement_and_reads_current_values():
    model = site()
    algorithm = SumAlgorithm()
    state = model.algorithm_state(algorithm, recipe=DefaultRecipe())
    assert isinstance(state, AlgorithmState)
    assert model.algorithm_state(algorithm, recipe=DefaultRecipe()) is state
    first = algorithm.run(state, SumInputs()).detach()
    with torch.no_grad():
        model.atoms.p.add_(1)
    second = algorithm.run(state, SumInputs()).detach()
    torch.testing.assert_close(second - first, first.new_tensor(model.atoms.p.numel()))
    assert algorithm.preparations == 1
    model.atom_state.relayout(torch.tensor([2, 0, 1]))
    assert not state.is_current()
    algorithm.run(state, SumInputs()).sum().backward()
    assert algorithm.preparations == 2
    assert state.is_current()
    model.float()
    assert not state.is_current()
    algorithm.run(state, SumInputs()).sum().backward()
    assert algorithm.preparations == 3


def test_algorithm_states_are_per_owner_and_transient():
    first, second = site(), site()
    algorithm = SumAlgorithm()
    a, b = (
        first.algorithm_state(algorithm, recipe=DefaultRecipe()),
        second.algorithm_state(algorithm, recipe=DefaultRecipe()),
    )
    assert a is not b and a.atom_state is not b.atom_state
    before = set(first.state_dict())
    algorithm.run(a, SumInputs()).sum().backward()
    assert set(first.state_dict()) == before
    clone = copy.deepcopy(first)
    assert "_algorithm_states" not in clone.__dict__
    assert clone.atom_state.atoms.__dict__["_atom_state_owner"] is clone.atom_state
    first.load_state_dict(copy.deepcopy(first.state_dict()))
    assert not a.is_current()


def test_optimizer_reload_invalidates_cached_storage_without_relayout():
    model = site()
    optimizer = CSTOptimizer(torch.optim.Adam(model.parameters()), model=model)
    model.atoms.p.grad = torch.ones_like(model.atoms.p)
    optimizer.step()
    algorithm = SumAlgorithm()
    state = model.algorithm_state(algorithm, recipe=DefaultRecipe())
    algorithm.run(state, SumInputs()).backward()
    assert state.is_current()
    version = model.atom_state.layout_version
    old_moment = model.atom_state.optimizer_state.get("exp_avg")
    optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert model.atom_state.layout_version == version
    assert model.atom_state.optimizer_state.get("exp_avg") is not old_moment
    assert not state.is_current()
    algorithm.run(state, SumInputs()).backward()
    assert state.is_current()
    with pytest.raises(ValueError, match="different algorithm"):
        SumAlgorithm().run(state, SumInputs())
