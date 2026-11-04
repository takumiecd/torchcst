"""Ordered execution edges must remain visible even when Modules are aliased."""

import os

import pytest
import torch
from test_full_step_graph import cohort  # noqa: F401
from torch import nn

from benchmarks.training.full_step_graph import PreparedStep
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans


def cpu_signature_cohort():
    model = nn.Sequential(
        nn.Linear(1, 1, bias=False),
        nn.ReLU(),
        nn.Linear(1, 1, bias=False),
        nn.ReLU(),
        nn.Linear(1, 2, bias=False),
    )
    with torch.no_grad():
        model[0].weight.fill_(1)
        model[2].weight.fill_(1)
        model[4].weight.copy_(torch.tensor([[-1.0], [1.0]]))
    program = PreparedStep.__new__(PreparedStep)
    program.model = model
    program.optimizer = torch.optim.AdamW(model.parameters())
    program.params = tuple(model.parameters())
    program.graph = None
    program.site = None
    program.periods = None
    program.tx = torch.ones(1, 1)
    program.ty = torch.zeros(1, dtype=torch.long)
    return model, program


def test_existing_relu_alias_changes_eager_semantics_and_signature():
    model, program = cpu_signature_cohort()
    before = program._signature()
    assert torch.equal(model(program.tx), torch.tensor([[-1.0, 1.0]]))
    model.add_module("aliased_relu", model[1])
    assert torch.equal(model(program.tx), torch.tensor([[0.0, 1.0]]))
    assert program._signature() != before


def test_none_execution_edge_is_not_hidden_by_module_deduplication():
    model, program = cpu_signature_cohort()
    before = program._signature()
    model.add_module("missing_stage", None)
    with pytest.raises(TypeError):
        model(program.tx)
    assert program._signature() != before


def test_topology_guard_still_allows_learned_parameter_value_updates():
    model, program = cpu_signature_cohort()
    before = program._signature()
    with torch.no_grad():
        model[0].weight.add_(0.25)
    assert program._signature() == before


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("CST_GRAPH_INPUTS"),
    reason="requires CUDA and frozen N1024/A52428/B4 inputs",
)
def test_alias_falls_back_before_graph_launch(monkeypatch, request):
    monkeypatch.setattr(
        plans,
        "GLOBAL_OWNER_WINDOW_RECOMPUTE",
        plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C,
    )
    model, opt, _, _, indices, program, _ = request.getfixturevalue("cohort")
    original = program.graph
    calls = []

    class RejectReplay:
        def replay(self):
            calls.append("replay")
            raise AssertionError("changed topology reached Graph replay")

    program.graph = RejectReplay()
    try:
        assert program.eligible()
        model.add_module("aliased_relu", model[1])
        assert not program.eligible()
        program.step(indices)
        assert program.invalidated and calls == []
        assert int(opt.state[model[2].atoms.p]["step"].item()) == 9
    finally:
        program.graph = original
