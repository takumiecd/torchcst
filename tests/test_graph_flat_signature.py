"""CPU whole-signature implication; CUDA APIs are mocked, never GPU proof."""

import os
from contextlib import nullcontext

import pytest
import torch
from test_graph_capture_qualification import (
    cpu_cuda_api,  # noqa: F401
    dense_cpu_inputs,
    install_virtual_body,
)

from benchmarks.training.flat_step_graph import FlatPreparedStep
from benchmarks.training.full_step_graph import PreparedStep, exact, snapshot

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    and os.environ.get("CST_TEST_NATIVE_FUNCTION_GUARD") != "1",
    reason="explicit CPU native-host qualification or CUDA required",
)


@pytest.fixture
def pair(monkeypatch, request):
    request.getfixturevalue("cpu_cuda_api")
    result = []
    for cls in (PreparedStep, FlatPreparedStep):
        torch.manual_seed(41)
        inputs = dense_cpu_inputs()
        inputs.model.register_buffer("setting", torch.ones(2))
        program = cls(inputs.model, inputs.optimizer, inputs.tx, inputs.ty, "dense")
        assert not program.invalidated, program.qualification_error
        install_virtual_body(monkeypatch, inputs, program)
        assert program.capture(preserve_totals=True)
        result.append((inputs, program))
    assert exact(
        snapshot(result[0][0].model, result[0][0].optimizer),
        snapshot(result[1][0].model, result[1][0].optimizer),
    )["all_byte_equal"]
    return result


def mutate(inputs, change):
    model, optimizer = inputs.model, inputs.optimizer
    if change == "values":
        with torch.no_grad():
            for p in inputs.params:
                p.add_(0.125)
                p.grad.add_(1)
                for v in optimizer.state[p].values():
                    v.add_(1)
    elif change == "storage":
        with torch.no_grad():
            inputs.params[0].set_(inputs.params[0].detach().clone())
    elif change == "grad":
        inputs.params[0].grad = inputs.params[0].grad.clone()
    elif change == "constant":
        model.setting.add_(1)
    elif change == "training":
        model[0].training = False
    elif change == "params-reverse":
        optimizer.param_groups[0]["params"].reverse()
    elif change == "groups-reverse":
        optimizer.param_groups.reverse()
    elif change == "nested-option":
        optimizer.param_groups[0]["extra"]["v"].append(2)
    elif change == "nested-state":
        optimizer.state[inputs.params[0]]["extra"]["v"].append(2)
    elif change == "root-alias":
        model.add_module("alias", model[1])
    elif change == "none-edge":
        model.add_module("none", None)
    elif change == "child-rebind":
        model[1] = torch.nn.ReLU()
    elif change == "parameter-alias":
        model.register_parameter("alias", inputs.params[0])
    elif change == "buffer-alias":
        model.register_buffer("alias", model.setting)
    else:
        raise AssertionError(change)


@pytest.mark.parametrize(
    "change",
    [
        "values",
        "storage",
        "grad",
        "constant",
        "training",
        "params-reverse",
        "groups-reverse",
        "nested-option",
        "nested-state",
        "root-alias",
        "none-edge",
        "child-rebind",
        "parameter-alias",
        "buffer-alias",
    ],
)
def test_flat_acceptance_implies_full_qualified_reference(pair, change):
    # Nested repr inputs are anchored before mutation, just as legacy did.
    if change.startswith("nested"):
        for inputs, program in pair:
            target = (
                inputs.optimizer.param_groups[0]
                if change == "nested-option"
                else inputs.optimizer.state[inputs.params[0]]
            )
            target["extra"] = {"v": [1]}
            program.signature = program._signature()
    for inputs, _ in pair:
        mutate(inputs, change)
    old, flat = (program.eligible() for _, program in pair)
    assert not flat or old
    if change == "values":
        assert old and flat
    elif change in ("parameter-alias", "buffer-alias"):
        assert not flat  # A conservative extra raw edge rejection is allowed.
    else:
        assert not old and not flat


def test_rebinding_the_prepared_model_cannot_keep_old_flat_root(pair):
    for inputs, program in pair:
        replacement = dense_cpu_inputs().model
        # Keep the old model alive: a dead weakref is not the only detection.
        assert inputs.model is program.model
        program.model = replacement
        assert not program.eligible()


def test_flat_shares_numeric_update_and_all_non_model_checks():
    for name in (
        "_signature",
        "_grad_signature",
        "eligible",
        "capture",
        "body",
        "step",
    ):
        assert getattr(FlatPreparedStep, name) is getattr(PreparedStep, name)


def test_guard_rejection_never_copies_indices_replays_or_updates_state(pair):
    for inputs, program in pair:
        before = snapshot(inputs.model, inputs.optimizer)
        indices = program.indices.clone()
        inputs.model.add_module("alias", inputs.model[1])
        assert not program.eligible()
        assert torch.equal(program.indices, indices)
        assert exact(before, snapshot(inputs.model, inputs.optimizer))["all_byte_equal"]
        # The fixture's CpuGraph.replay raises, so no unchecked replay is hidden.


@pytest.mark.parametrize("mode", ["no-grad", "inference", "saved-hooks", "precision"])
def test_retained_tls_and_live_precision_checks_reject_both(pair, monkeypatch, mode):
    if mode == "precision":
        monkeypatch.setattr(
            torch.backends.cuda.matmul,
            "allow_tf32",
            not torch.backends.cuda.matmul.allow_tf32,
        )
        context = nullcontext()
    elif mode == "saved-hooks":
        context = torch.autograd.graph.saved_tensors_hooks(lambda v: v, lambda v: v)
    elif mode == "inference":
        context = torch.inference_mode()
    else:
        context = torch.no_grad()
    with context:
        assert all(not program.eligible() for _, program in pair)


@pytest.mark.parametrize("role", ["grad", "state", "input", "period"])
@pytest.mark.parametrize("accessor", ["data_ptr", "stride"])
def test_unregistered_metadata_provider_is_rejected_without_invocation(
    pair, monkeypatch, role, accessor
):
    for inputs, program in pair:
        if role == "period":
            program.periods = (torch.ones(1),)
            program.signature = program._signature()
        value = {
            "grad": inputs.params[0].grad,
            "state": inputs.optimizer.state[inputs.params[0]]["exp_avg"],
            "input": program.tx,
            "period": program.periods[0] if program.periods else None,
        }[role]
        calls = []

        def unexpected_provider(calls=calls):
            calls.append(True)
            raise AssertionError("unqualified metadata provider executed")

        monkeypatch.setattr(value, accessor, unexpected_provider)
        versions = tuple(p._version for p in inputs.params)
        before = snapshot(inputs.model, inputs.optimizer)
        assert not program.eligible()
        assert calls == []
        assert versions == tuple(p._version for p in inputs.params)
        assert exact(before, snapshot(inputs.model, inputs.optimizer))["all_byte_equal"]
