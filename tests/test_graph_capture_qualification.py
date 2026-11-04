"""CPU control-flow tests of capture qualification, not CUDA Graph proof.

Only CUDA's stream/graph/synchronization API is mocked. PreparedStep's actual
constructor, qualification, signatures, capture rollback and eligibility run on
CPU. A deterministic CPU body stands in for the four virtual numeric updates;
these tests do not establish CUDA capture/replay or GPU memory correctness.
"""

import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from benchmarks.training import full_step_graph as graph_module


@pytest.fixture
def cpu_cuda_api(monkeypatch):
    events = []
    owner = SimpleNamespace(cuda_stream=913)

    class CpuGraph:
        def __init__(self):
            events.append("graph-created")

        def reset(self):
            events.append("graph-retired")

        def replay(self):
            raise AssertionError("a CPU capture qualification test reached replay")

    @contextmanager
    def stream_context(stream):
        assert stream is owner
        yield

    @contextmanager
    def graph_context(graph, *, stream):
        assert isinstance(graph, CpuGraph)
        assert stream is owner
        events.append("capture-enter")
        yield
        events.append("capture-exit")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: owner)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "graph", graph_context)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", CpuGraph)
    return events


def dense_cpu_inputs():
    model = nn.Sequential(
        nn.Linear(1, 1),
        nn.ReLU(),
        nn.Linear(1, 1, bias=False),
        nn.ReLU(),
        nn.Linear(1, 2),
    )
    params = tuple(model.parameters())
    ordinary = [*model[0].parameters(), *model[4].parameters()]
    optimizer = torch.optim.AdamW(
        [
            {"params": ordinary, "role": "ordinary"},
            {"params": [model[2].weight], "role": "middle"},
        ],
        fused=True,
        capturable=False,
    )
    for index, parameter in enumerate(params):
        parameter.grad = torch.full_like(parameter, 0.125 + index)
        optimizer.state[parameter] = {
            "step": torch.tensor(8.0),
            "exp_avg": torch.full_like(parameter, 0.25 + index),
            "exp_avg_sq": torch.full_like(parameter, 0.5 + index),
        }
    return SimpleNamespace(
        model=model,
        optimizer=optimizer,
        params=params,
        tx=torch.ones(8, 1),
        ty=torch.zeros(8, dtype=torch.long),
    )


def construct(inputs):
    return graph_module.PreparedStep(
        inputs.model, inputs.optimizer, inputs.tx, inputs.ty, "dense"
    )


def numeric_fields(inputs, program):
    # Use previously recorded Parameters, never a possibly overridden traversal.
    fields = [*inputs.params]
    fields.extend(
        value
        for parameter in inputs.params
        for value in inputs.optimizer.state[parameter].values()
    )
    return [*fields, *program.totals]


def numeric_snapshot(inputs, program):
    return tuple(value.detach().clone() for value in numeric_fields(inputs, program))


def assert_numeric_restored(inputs, program, expected):
    actual = numeric_fields(inputs, program)
    assert len(actual) == len(expected)
    assert all(torch.equal(value, saved) for value, saved in zip(actual, expected))
    assert all(
        inputs.optimizer.state[parameter]["step"].item() == 8
        for parameter in inputs.params
    )


def grad_and_versions(inputs, program):
    return (
        tuple(
            (id(parameter.grad), parameter.grad.detach().clone())
            for parameter in inputs.params
        ),
        tuple(
            value._version
            for value in (
                *numeric_fields(inputs, program),
                *(parameter.grad for parameter in inputs.params),
            )
        ),
    )


def assert_grad_and_versions_unchanged(inputs, program, expected):
    grads, versions = expected
    for parameter, (identity, value) in zip(inputs.params, grads):
        assert id(parameter.grad) == identity
        assert torch.equal(parameter.grad, value)
    assert grad_and_versions(inputs, program)[1] == versions


def change_callback(monkeypatch, change):
    if change == "ce-defaults":
        function = nn.functional.cross_entropy
        defaults = list(function.__defaults__)
        defaults[-2] = "sum"
        monkeypatch.setattr(function, "__defaults__", tuple(defaults))
    elif change == "relu-class":

        def changed_relu(self, value):
            return value * 0

        monkeypatch.setattr(nn.ReLU, "forward", changed_relu)
    else:

        def forbidden_traversal(*args, **kwargs):
            raise AssertionError("unqualified traversal executed")

        monkeypatch.setattr(nn.Module, "named_parameters", forbidden_traversal)


def install_virtual_body(monkeypatch, inputs, program, *, mutate=None, fail_at=None):
    calls = []
    temporary_refs = []

    def body():
        calls.append(len(calls) + 1)
        with torch.no_grad():
            for parameter in inputs.params:
                parameter.add_(1)
                parameter.grad = torch.full_like(parameter, 30 + len(calls))
                state = inputs.optimizer.state[parameter]
                state["step"].add_(1)
                state["exp_avg"].add_(2)
                state["exp_avg_sq"].add_(3)
            program.totals[0].add_(17)
            program.totals[1].add_(1)
        program.logits = torch.full((4, 2), float(len(calls)))
        program.loss = torch.tensor(float(len(calls)))
        temporary_refs.extend((weakref.ref(program.logits), weakref.ref(program.loss)))
        if mutate is not None:
            mutate(len(calls))
        if len(calls) == fail_at:
            raise RuntimeError("virtual CPU body failed")

    monkeypatch.setattr(program, "body", body)
    return calls, temporary_refs


@pytest.mark.parametrize("phase", ["before-construction", "before-capture"])
@pytest.mark.parametrize("change", ["ce-defaults", "relu-class", "traversal"])
def test_unqualified_callback_stops_before_first_virtual_body(
    monkeypatch, cpu_cuda_api, phase, change
):
    inputs = dense_cpu_inputs()
    if phase == "before-construction":
        change_callback(monkeypatch, change)
    program = construct(inputs)
    if phase == "before-capture":
        change_callback(monkeypatch, change)
    calls, _ = install_virtual_body(monkeypatch, inputs, program)
    before = numeric_snapshot(inputs, program)
    unchanged = grad_and_versions(inputs, program)
    assert program.capture(preserve_totals=True) is False
    assert calls == []
    assert program.invalidated and program.graph is None
    assert_numeric_restored(inputs, program, before)
    assert_grad_and_versions_unchanged(inputs, program, unchanged)
    assert "graph-created" not in cpu_cuda_api


def test_non_grad_drift_is_checked_before_first_virtual_body(monkeypatch, cpu_cuda_api):
    inputs = dense_cpu_inputs()
    program = construct(inputs)
    inputs.optimizer.param_groups[0]["lr"] *= 2
    calls, _ = install_virtual_body(monkeypatch, inputs, program)
    before = numeric_snapshot(inputs, program)
    unchanged = grad_and_versions(inputs, program)
    assert program.capture(preserve_totals=True) is False
    assert calls == []
    assert program.invalidated and program.graph is None
    assert_numeric_restored(inputs, program, before)
    assert_grad_and_versions_unchanged(inputs, program, unchanged)


def test_four_virtual_updates_restore_clock_metrics_and_refresh_only_grad(
    monkeypatch, cpu_cuda_api
):
    inputs = dense_cpu_inputs()
    program = construct(inputs)
    program.totals[0].fill_(23)
    program.totals[1].fill_(7)
    before = numeric_snapshot(inputs, program)
    old_grads = tuple(weakref.ref(parameter.grad) for parameter in inputs.params)
    non_grad_signature = program._signature(include_grad=False)
    calls, _ = install_virtual_body(monkeypatch, inputs, program)
    assert program.capture(preserve_totals=True) is True
    assert calls == [1, 2, 3, 4]
    assert cpu_cuda_api.count("capture-enter") == 1
    assert not program.invalidated and program.graph is not None
    assert_numeric_restored(inputs, program, before)
    assert all(reference() is None for reference in old_grads)
    assert all(torch.all(parameter.grad == 34) for parameter in inputs.params)
    assert program._signature(include_grad=False) == non_grad_signature
    assert program.signature == program._signature()
    assert program.eligible()


@pytest.mark.parametrize("change", ["group-option", "ce-defaults"])
def test_mid_capture_mutation_is_restored_but_never_reanchored(
    monkeypatch, cpu_cuda_api, change
):
    inputs = dense_cpu_inputs()
    program = construct(inputs)
    program.totals[0].fill_(23)
    program.totals[1].fill_(7)
    before = numeric_snapshot(inputs, program)
    original_signature = program.signature

    def mutate(call):
        if call == 2:
            if change == "group-option":
                inputs.optimizer.param_groups[0]["lr"] *= 2
            else:
                change_callback(monkeypatch, "ce-defaults")

    calls, temporary_refs = install_virtual_body(
        monkeypatch, inputs, program, mutate=mutate
    )
    assert program.capture(preserve_totals=True) is False
    assert len(calls) >= 2
    assert program.invalidated and program.graph is None
    assert_numeric_restored(inputs, program, before)
    assert program.signature == original_signature
    assert not program.eligible()
    assert all(reference() is None for reference in temporary_refs)


def test_grad_refresh_cannot_hide_simultaneous_last_body_non_grad_mutation(
    monkeypatch, cpu_cuda_api
):
    inputs = dense_cpu_inputs()
    program = construct(inputs)
    before = numeric_snapshot(inputs, program)
    original_signature = program.signature

    def mutate(call):
        if call == 4:
            inputs.model[0].training = not inputs.model[0].training

    calls, _ = install_virtual_body(monkeypatch, inputs, program, mutate=mutate)
    assert program.capture(preserve_totals=True) is False
    assert calls == [1, 2, 3, 4]
    assert_numeric_restored(inputs, program, before)
    assert program.signature == original_signature
    assert program.invalidated and program.graph is None
    assert not program.eligible()


def test_mid_capture_fixed_buffer_change_is_not_rolled_back_as_numeric_state(
    monkeypatch, cpu_cuda_api
):
    inputs = dense_cpu_inputs()
    inputs.model.register_buffer("fixed_setting", torch.tensor(3.0))
    program = construct(inputs)
    program.totals[0].fill_(23)
    before = numeric_snapshot(inputs, program)
    original_signature = program.signature

    def mutate(call):
        if call == 2:
            inputs.model.fixed_setting.add_(1)

    calls, temporary_refs = install_virtual_body(
        monkeypatch, inputs, program, mutate=mutate
    )
    assert program.capture(preserve_totals=True) is False
    assert len(calls) >= 2
    assert_numeric_restored(inputs, program, before)
    assert inputs.model.fixed_setting.item() == 4
    assert inputs.model.fixed_setting._version == 1
    assert program.signature == original_signature
    assert program.invalidated and program.graph is None
    assert not program.eligible()
    assert all(reference() is None for reference in temporary_refs)


@pytest.mark.parametrize("hook_kind", ["module-forward", "optimizer-pre"])
def test_last_virtual_body_hook_rejected_before_grad_signature_refresh(
    monkeypatch, cpu_cuda_api, hook_kind
):
    inputs = dense_cpu_inputs()
    program = construct(inputs)
    program.totals[0].fill_(23)
    program.totals[1].fill_(7)
    before = numeric_snapshot(inputs, program)
    original_signature = program.signature
    handles = []

    def mutate(call):
        if call == 4:
            if hook_kind == "module-forward":
                handle = inputs.model[0].register_forward_hook(lambda *args: None)
            else:
                handle = inputs.optimizer.register_step_pre_hook(lambda *args: None)
            handles.append(handle)

    calls, temporary_refs = install_virtual_body(
        monkeypatch, inputs, program, mutate=mutate
    )
    try:
        assert program.capture(preserve_totals=True) is False
        assert calls == [1, 2, 3, 4]
        assert len(handles) == 1
        assert_numeric_restored(inputs, program, before)
        assert program.signature == original_signature
        assert program.invalidated and program.graph is None
        assert not program.eligible()
        assert all(reference() is None for reference in temporary_refs)
    finally:
        for handle in handles:
            handle.remove()
    assert not inputs.model[0]._forward_hooks
    assert not inputs.optimizer._optimizer_step_pre_hooks


def test_body_exception_restores_numeric_state_or_stops_cohort(
    monkeypatch, cpu_cuda_api
):
    inputs = dense_cpu_inputs()
    program = construct(inputs)
    program.totals[0].fill_(23)
    before = numeric_snapshot(inputs, program)
    calls, temporary_refs = install_virtual_body(
        monkeypatch, inputs, program, fail_at=2
    )
    try:
        result = program.capture(preserve_totals=True)
    except RuntimeError as error:
        assert "virtual CPU body failed" in str(error)
    else:
        assert result is False
    assert calls == [1, 2]
    assert_numeric_restored(inputs, program, before)
    assert program.invalidated and program.graph is None
    assert all(reference() is None for reference in temporary_refs)


def test_restore_failure_raises_and_cannot_leave_replay_eligible(
    monkeypatch, cpu_cuda_api
):
    inputs = dense_cpu_inputs()
    program = construct(inputs)
    install_virtual_body(monkeypatch, inputs, program)

    def failed_restore(*args, **kwargs):
        raise RuntimeError("CPU rollback deliberately unavailable")

    monkeypatch.setattr(graph_module, "restore_capture_state", failed_restore)
    with pytest.raises(RuntimeError, match="rollback deliberately unavailable"):
        program.capture(preserve_totals=True)
    assert program.invalidated and program.graph is None
    assert not program.eligible()
    assert program.restoration_failed
    stopped_state = numeric_snapshot(inputs, program)
    with pytest.raises(RuntimeError, match="cohort updates are disabled"):
        program.step(torch.arange(4))
    assert all(
        torch.equal(value, saved)
        for value, saved in zip(numeric_fields(inputs, program), stopped_state)
    )


@pytest.mark.parametrize("kind", ["module", "optimizer"])
def test_cold_hooks_select_eager_without_virtual_capture(
    monkeypatch, cpu_cuda_api, kind
):
    inputs = dense_cpu_inputs()
    calls = []
    handle = (
        inputs.model[0].register_forward_hook(lambda *_: calls.append("forward"))
        if kind == "module"
        else inputs.optimizer.register_step_pre_hook(
            lambda *_: calls.append("optimizer")
        )
    )
    try:
        program = construct(inputs)
        virtual, _ = install_virtual_body(monkeypatch, inputs, program)
        before = numeric_snapshot(inputs, program)
        assert program.capture() is False
        assert virtual == [] and calls == []
        assert_numeric_restored(inputs, program, before)
        program.step(torch.arange(4))
        assert len(calls) == 1
        assert all(inputs.optimizer.state[p]["step"].item() == 9 for p in inputs.params)
    finally:
        handle.remove()


def test_numeric_state_rebind_during_capture_stops_updates(monkeypatch, cpu_cuda_api):
    inputs = dense_cpu_inputs()
    program = construct(inputs)
    original_moment = inputs.optimizer.state[inputs.params[0]]["exp_avg"]
    before = original_moment.clone()

    def mutate(call):
        if call == 2:
            state = inputs.optimizer.state[inputs.params[0]]
            state["exp_avg"] = state["exp_avg"].clone()

    _, temporary_refs = install_virtual_body(
        monkeypatch, inputs, program, mutate=mutate
    )
    with pytest.raises(RuntimeError, match="Adam state bindings"):
        program.capture()
    assert torch.equal(original_moment, before)
    assert program.restoration_failed and program.graph is None
    assert all(reference() is None for reference in temporary_refs)
    with pytest.raises(RuntimeError, match="cohort updates are disabled"):
        program.step(torch.arange(4))


def test_postflight_exception_also_retires_captured_temporaries(
    monkeypatch, cpu_cuda_api
):
    inputs = dense_cpu_inputs()
    program = construct(inputs)
    before = numeric_snapshot(inputs, program)

    calls, temporary_refs = install_virtual_body(monkeypatch, inputs, program)
    signature = program._signature

    def unavailable_postflight(*args, **kwargs):
        if len(calls) == 4:
            raise RuntimeError("postflight metadata unavailable")
        return signature(*args, **kwargs)

    monkeypatch.setattr(program, "_signature", unavailable_postflight)
    with pytest.raises(RuntimeError, match="postflight metadata unavailable"):
        program.capture()
    assert_numeric_restored(inputs, program, before)
    assert program.invalidated and program.graph is None
    assert all(reference() is None for reference in temporary_refs)


def test_changed_metadata_declaration_cannot_run_during_rollback(
    monkeypatch, cpu_cuda_api
):
    inputs = dense_cpu_inputs()
    program = construct(inputs)
    calls = []

    def forbidden_stride(*args):
        calls.append(True)
        raise AssertionError("changed metadata declaration was invoked")

    def mutate(call):
        if call == 2:
            monkeypatch.setattr(torch.Tensor, "stride", forbidden_stride)

    install_virtual_body(monkeypatch, inputs, program, mutate=mutate)
    with pytest.raises(RuntimeError, match="metadata declarations changed"):
        program.capture()
    assert calls == []
    assert program.restoration_failed and program.graph is None
    with pytest.raises(RuntimeError, match="cohort updates are disabled"):
        program.step(torch.arange(4))
