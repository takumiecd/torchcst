"""Ordered execution edges must remain visible even when Modules are aliased."""

import gc
import os
import weakref

import pytest
import torch
from test_full_step_graph import cohort  # noqa: F401
from torch import nn

from benchmarks.training.full_step_graph import (
    PreparedStep,
    RootTopology,
    exact,
    ordinary_step,
    restore,
    snapshot,
)
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
    program._root_topology = RootTopology(model)
    return model, program


def test_existing_relu_alias_changes_eager_semantics_and_root_edges():
    model, program = cpu_signature_cohort()
    before = program._signature()
    assert torch.equal(model(program.tx), torch.tensor([[-1.0, 1.0]]))
    model.add_module("aliased_relu", model[1])
    assert torch.equal(model(program.tx), torch.tensor([[0.0, 1.0]]))
    # Module traversal deduplicates the alias. Ordered execution edges do not.
    assert program._signature() == before
    assert not program._root_topology.unchanged()


def test_none_execution_edge_is_not_hidden_by_module_deduplication():
    model, program = cpu_signature_cohort()
    before = program._signature()
    model.add_module("missing_stage", None)
    with pytest.raises(TypeError):
        model(program.tx)
    assert program._signature() == before
    assert not program._root_topology.unchanged()


@pytest.mark.parametrize("change", ["delete", "reorder", "rebind", "rename"])
def test_root_registration_mutations_are_rejected(change):
    model, program = cpu_signature_cohort()
    assert program._root_topology.unchanged()
    if change == "delete":
        del model._modules["1"]
    elif change == "reorder":
        model._modules["1"] = model._modules.pop("1")
    elif change == "rebind":
        model[1] = nn.ReLU()
    else:
        model._modules = {
            "renamed" if k == "1" else k: v for k, v in model._modules.items()
        }
    assert not program._root_topology.unchanged()


def test_topology_observer_does_not_retain_removed_module_or_root():
    model = nn.Sequential(nn.ReLU())
    child = weakref.ref(model[0])
    root = weakref.ref(model)
    observer = RootTopology(model)
    model[0] = nn.ReLU()
    gc.collect()
    assert child() is None
    assert not observer.unchanged()
    del model
    gc.collect()
    assert root() is None
    assert not observer.unchanged()


def test_topology_guard_still_allows_learned_parameter_value_updates():
    model, program = cpu_signature_cohort()
    before = program._signature()
    with torch.no_grad():
        model[0].weight.add_(0.25)
    assert program._signature() == before
    assert program._root_topology.unchanged()


def state_and_gradients(model, opt):
    return {
        **snapshot(model, opt),
        **{
            "grad/" + name: p.grad.detach().cpu().clone()
            for name, p in model.named_parameters()
        },
    }


def version_signature(model, opt, program):
    return tuple(
        value._version
        for value in (
            *model.parameters(),
            *model.buffers(),
            *(p.grad for p in model.parameters()),
            *(
                v
                for s in opt.state.values()
                for v in s.values()
                if isinstance(v, torch.Tensor)
            ),
            *program.totals,
        )
    )


def fp32_cohort(monkeypatch, request):
    monkeypatch.setattr(
        plans,
        "GLOBAL_OWNER_WINDOW_RECOMPUTE",
        plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C,
    )
    return request.getfixturevalue("cohort")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("CST_GRAPH_INPUTS"),
    reason="requires CUDA and frozen N1024/A52428/B4 inputs",
)
def test_alias_guard_rejection_does_not_change_numeric_state(monkeypatch, request):
    model, opt, _, _, _, program, _ = fp32_cohort(monkeypatch, request)
    assert program.eligible()
    before = state_and_gradients(model, opt)
    metrics = tuple(t.detach().cpu().clone() for t in program.totals)
    versions = version_signature(model, opt, program)
    signature = program._signature()
    model.add_module("aliased_relu", model[1])
    assert not program.eligible()
    assert not program.invalidated  # Eligibility is a read-only probe.
    if getattr(program, "_use_flat", False):
        # G1 also certifies the complete ordered edge set in its model leaves.
        # It deliberately invalidates these columns on the added alias.
        assert program._signature() != signature
    else:
        # The recursive columns deduplicate aliases; the common root guard is
        # what catches this case in qualified G0.
        assert program._signature() == signature
    assert exact(before, state_and_gradients(model, opt))["all_byte_equal"]
    assert all(torch.equal(t.cpu(), saved) for t, saved in zip(program.totals, metrics))
    assert versions == version_signature(model, opt, program)
    assert int(opt.state[model[2].atoms.p]["step"].item()) == 8


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("CST_GRAPH_INPUTS"),
    reason="requires CUDA and frozen N1024/A52428/B4 inputs",
)
def test_alias_falls_back_before_graph_launch(monkeypatch, request):
    model, opt, tx, ty, indices, program, _ = fp32_cohort(monkeypatch, request)
    model.add_module("aliased_relu", model[1])
    assert not program.eligible()
    start = snapshot(model, opt)
    start_totals = tuple(t.detach().cpu().clone() for t in program.totals)
    y, loss = ordinary_step(model, opt, tx, ty, indices, program.totals)
    expected = {
        **state_and_gradients(model, opt),
        "logits": y.detach().cpu().clone(),
        "loss": loss.detach().cpu().clone(),
        **{
            f"total/{i}": t.detach().cpu().clone() for i, t in enumerate(program.totals)
        },
    }
    del y, loss
    gc.collect()
    torch.cuda.synchronize()
    restore(model, opt, start)
    for target, saved in zip(program.totals, start_totals):
        target.copy_(saved)
    original = program.graph
    calls = []

    class RejectReplay:
        def replay(self):
            calls.append("replay")
            raise AssertionError("changed topology reached Graph replay")

    program.graph = RejectReplay()
    try:
        assert not program.eligible()
        y, loss = program.step(indices)
        actual = {
            **state_and_gradients(model, opt),
            "logits": y,
            "loss": loss,
            **{f"total/{i}": t for i, t in enumerate(program.totals)},
        }
        assert exact(expected, actual)["all_byte_equal"]
        assert program.invalidated and calls == []
        assert int(opt.state[model[2].atoms.p]["step"].item()) == 9
    finally:
        program.graph = original
