"""A prepared Linear entrance must reject changes before numerical execution."""

import copy
import gc
import importlib
import os
import weakref
from dataclasses import replace

import pytest
import torch

from benchmarks.cuda.linear.manifest import load_run
from benchmarks.cuda.linear.radius_cell_mass import configure, fixture
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import prepared
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.dispatch.prepared import PreparedDispatch, placement_signature
from torchcst._backends.dispatch.runtime import Dispatcher
from torchcst._backends.schema import DispatchDecision
from torchcst.operators.execution import LinearInputs


@pytest.fixture(scope="module", autouse=True)
def native_checks():
    if (
        not torch.cuda.is_available()
        and os.getenv("CST_TEST_NATIVE_FUNCTION_GUARD") != "1"
    ):
        pytest.skip("explicit CPU native-check build or CUDA required")
    importlib.import_module("torch._dynamo.guards")

    from torchcst._backends.cuda.algorithms.polar_update.function_metadata import (
        get_factory,
    )

    assert get_factory() is not None


def case():
    # Only metadata is prepared on CPU; this does not claim CUDA support or
    # execute a CUDA Algorithm. Real module calls are checked separately below.
    run = load_run(
        "benchmarks/cuda/linear/cases/radius-cell-mass-128-native-initial.json",
        "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )
    model, x, *_ = fixture(run.case)
    plan = run.plans[0].plan
    model = configure(model, plan)
    dispatcher = Dispatcher(selector=model.selector)
    algorithm = dispatcher.registry.validate_plan(plan)
    inputs = LinearInputs(x.requires_grad_())
    context = model.build_context(inputs)
    state = model.algorithm_state(algorithm, recipe=plan.recipe)
    state.context = context
    state.layout = hierarchy_from_operator(model.execution_declaration())
    state.mark_current()
    cached = PreparedDispatch(
        dispatcher.registry,
        ("fixed", model.selector.revision, plan, None),
        context,
        placement_signature(model),
        DispatchDecision(plan, "fixed-v1", ("fixed", plan.algorithm_id), "test"),
        algorithm,
        state,
        plan.recipe,
        state.recipe,
    )
    model._prepared_dispatch = cached
    model._prepared_dispatch_stats = {"hits": 0, "misses": 1}
    program = prepared.prepare_program(dispatcher, cached, inputs)
    assert program is not None
    assert check(program, dispatcher, inputs)
    return model, dispatcher, inputs, cached, program


def check(program, dispatcher, inputs):
    roots = (program.template[0], dispatcher, inputs, *program.template[3:])
    return program.callback_guard.check(roots) and program.guard.check(roots)


def test_new_input_storage_is_allowed_and_initial_tensor_is_not_retained():
    _, dispatcher, inputs, _, program = case()
    reference = weakref.ref(inputs.x)
    another = LinearInputs(torch.zeros_like(inputs.x, requires_grad=True))
    assert check(program, dispatcher, another)
    del inputs
    gc.collect()
    assert reference() is None
    assert check(program, dispatcher, another)


@pytest.mark.parametrize("change", ["shape", "stride", "dtype", "grad", "subclass"])
def test_changed_input_metadata_uses_the_ordinary_entrance(change):
    _, dispatcher, inputs, _, program = case()
    if change == "shape":
        value = torch.ones(5, 128, requires_grad=True)
    elif change == "stride":
        value = torch.ones(128, 4).T.requires_grad_()
    elif change == "dtype":
        value = inputs.x.double()
    elif change == "grad":
        value = inputs.x.detach()
    else:

        class Custom(torch.Tensor):
            pass

        value = inputs.x.as_subclass(Custom)
    assert not check(program, dispatcher, LinearInputs(value))


@pytest.mark.parametrize(
    "change", ["storage", "optimizer", "layout", "buffer", "recipe", "state"]
)
def test_changed_live_metadata_cannot_reuse_the_prepared_program(change):
    model, dispatcher, inputs, cached, program = case()
    if change == "storage":
        model.atoms.p.data = model.atoms.p.detach().clone()
    elif change == "optimizer":
        model.atom_state.optimizer_state.fields["late"] = torch.ones(2)
    elif change == "layout":
        model.atom_state.layout_version += 1
    elif change == "buffer":
        with torch.no_grad():
            value = next(model.kernel.buffers())
            value.copy_(value)  # Same values still require declaration refresh.
    elif change == "recipe":
        object.__setattr__(cached.state.recipe, "output_tile_sites", 16)
    else:
        cached.state.recipe = replace(cached.state.recipe, output_tile_sites=16)
    assert not check(program, dispatcher, inputs)


def test_grad_mode_changes_are_not_hidden_by_a_cached_context():
    _, dispatcher, inputs, _, program = case()
    with torch.no_grad():
        assert not check(program, dispatcher, inputs)
    assert check(program, dispatcher, inputs)


def test_replaced_class_accessor_is_rejected_before_it_is_called(monkeypatch):
    model, dispatcher, inputs, _, program = case()
    calls = []
    monkeypatch.setattr(
        type(model), "input_type", property(lambda self: calls.append(1))
    )
    assert not check(program, dispatcher, inputs)
    assert calls == []


def test_changed_function_code_and_instance_callbacks_request_fallback(monkeypatch):
    model, dispatcher, inputs, cached, program = case()
    fn = type(model).validate_inputs
    monkeypatch.setattr(fn, "__code__", (lambda self, inputs: None).__code__)
    assert not check(program, dispatcher, inputs)
    assert prepared.prepare_program(dispatcher, cached, inputs) is None


def test_instance_override_and_registry_override_are_not_certified(monkeypatch):
    model, dispatcher, inputs, cached, _ = case()
    model.validate_inputs = lambda inputs: None
    assert prepared.prepare_program(dispatcher, cached, inputs) is None
    del model.validate_inputs
    monkeypatch.setattr(
        type(dispatcher.registry), "get", lambda *a, **k: cached.algorithm
    )
    assert prepared.prepare_program(dispatcher, cached, inputs) is None


def test_program_and_attempt_stamp_are_removed_from_model_copies():
    model, _, _, cached, program = case()
    model._execution_program = program
    model._execution_program_attempt = (cached, cached.state._storage)
    copied = copy.deepcopy(model)
    assert "_execution_program" not in copied.__dict__
    assert "_execution_program_attempt" not in copied.__dict__


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def gpu_case(*, native_entry=True):
    from benchmarks.cuda.linear.radius_cell_mass import optimizer, train

    run = load_run(
        "benchmarks/cuda/linear/cases/radius-cell-mass-128-native-initial.json",
        "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )
    model, x, target, _ = fixture(run.case)
    plan = run.plans[0].plan
    plan = replace(plan, recipe=replace(plan.recipe, native_entry=native_entry))
    model = configure(model.cuda(), plan)
    x, target = x.cuda().requires_grad_(), target.cuda()
    opt = optimizer(model, run.case)
    for _ in range(2):
        train(model, opt, x, target)
    if native_entry:
        assert model._execution_program is not None
    return model, x, target, opt


@GPU
def test_public_prepared_module_has_bit_identical_five_step_states():
    runs = []
    for native_entry in (False, True):
        model, x, target, opt = gpu_case(native_entry=native_entry)
        values = {}
        for step in range(5):
            opt.zero_grad(set_to_none=True)
            x.grad = None
            y = model(x)
            (y - target).square().mean().backward()
            for name, value in (("Y", y), ("dX", x.grad), ("dP", model.atoms.p.grad)):
                values[step, name] = value.detach().cpu().clone()
            opt.step()
            values[step, "P"] = model.atoms.p.detach().cpu().clone()
            for index, state in enumerate(opt.state.values()):
                for name, value in state.items():
                    if isinstance(value, torch.Tensor):
                        values[step, index, name] = value.detach().cpu().clone()
        runs.append(values)
    assert runs[0].keys() == runs[1].keys()
    for name in runs[0]:
        torch.testing.assert_close(runs[0][name], runs[1][name], rtol=0, atol=0)


@GPU
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (True, False), (False, True)]
)
def test_prepared_read_lease_retained_backward_and_offload(offload, needs_x, needs_p):
    from contextlib import nullcontext

    model, x, _, _ = gpu_case()
    x.requires_grad_(needs_x)
    model.atoms.p.requires_grad_(needs_p)

    def pack(value):
        return value.cpu(), value.device

    def unpack(saved):
        return saved[0].to(saved[1])

    with (
        torch.autograd.graph.saved_tensors_hooks(pack, unpack)
        if offload
        else nullcontext()
    ):
        y = model(x)
    selected = tuple(v for v in (x, model.atoms.p) if v.requires_grad)
    torch.autograd.grad(y, selected, torch.ones_like(y), retain_graph=True)
    with pytest.raises(RuntimeError, match="outstanding|retained"):
        model.atom_state._require_boundary()
    torch.autograd.grad(y, selected, torch.ones_like(y))
    model.atom_state._require_boundary()


@GPU
def test_prepared_input_and_plan_changes_reprepare_and_dtype_errors_are_preserved():
    model, x, _, _ = gpu_case()
    old = model._execution_program
    with pytest.raises(ValueError, match="dtype"):
        model(x.double())
    another = torch.randn(5, 128, device="cuda", requires_grad=True)
    y = model(another)
    assert y.shape == (5, 128)
    assert model._execution_program is not old
    selected = model.selector._plan
    model.selector._plan = replace(
        selected, recipe=replace(selected.recipe, output_tile_sites=16)
    )
    reference = y.detach()
    torch.testing.assert_close(model(another), reference, rtol=1e-6, atol=1e-7)


@GPU
def test_changed_buffer_and_parameter_storage_rebuild_the_selected_layout():
    model, x, _, _ = gpu_case()
    program = model._execution_program
    with torch.no_grad():
        model.kernel.scalar("radius_min_input").mul_(1.1)
    actual = model(x)
    assert model._execution_program is not program
    reference = copy.deepcopy(model).cpu().double()
    reference.selector = None
    torch.testing.assert_close(
        actual.cpu().double(), reference(x.cpu().double()), rtol=5e-5, atol=5e-6
    )
    program = model._execution_program
    model.atoms.p.data = model.atoms.p.detach().clone()
    torch.testing.assert_close(model(x), actual, rtol=0, atol=0)
    assert model._execution_program is not program


@GPU
def test_instance_validation_callback_runs_once_on_fallback():
    model, x, _, _ = gpu_case()
    calls = []
    original = model.validate_inputs

    def validate(inputs):
        calls.append(inputs)
        return original(inputs)

    model.validate_inputs = validate
    assert torch.isfinite(model(x)).all()
    assert len(calls) == 1
