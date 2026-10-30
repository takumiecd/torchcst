"""Optional host C++ checks preserve callback and dictionary mutation gates."""

import gc
import importlib
import os
import weakref
from types import FunctionType, SimpleNamespace

import pytest
import torch
from test_prepared_optimizer_update import example

from torchcst._backends.cuda import _host_function_metadata as function_metadata
from torchcst.operators._native_metadata_guard import guard_api


@pytest.fixture(scope="module")
def native_factory():
    if (
        not torch.cuda.is_available()
        and os.environ.get("CST_TEST_NATIVE_FUNCTION_GUARD") != "1"
    ):
        pytest.skip("explicit native-host compilation or CUDA required")
    factory = function_metadata.get_factory()
    assert factory is not None, "native host helper qualification failed"
    return factory


def definition():
    def template(value=1, *, scale=2):
        return value * scale

    fn = FunctionType(template.__code__, {}, argdefs=template.__defaults__)
    fn.__kwdefaults__ = dict(template.__kwdefaults__)
    return fn


def combined_guard(factory, functions):
    importlib.import_module("torch._dynamo.guards")
    api = guard_api()
    if api is None:
        pytest.skip("optional native guard API unavailable")
    checker, dictionaries = function_metadata.prepare_checks(factory, functions, ())
    assert checker is not None
    cls, kind = api
    guard = cls()
    guard.add_lambda_guard(checker, ["function slots"], None)
    for index, dictionary in enumerate(dictionaries):
        node = guard.tuple_getitem_manager(index, "dictionary", dictionary, kind)
        node.add_dict_version_guard(dictionary, ["function dictionary"], None)
    assert guard.check(dictionaries)
    return guard, dictionaries


@pytest.mark.parametrize(
    "change",
    [
        "code",
        "defaults",
        "defaults_none",
        "keywords",
        "keywords_none",
        "function_dictionary",
        "function_dictionary_write",
        "keywords_write",
        "globals_write",
    ],
)
def test_slot_and_dictionary_mutations_remain_observed(native_factory, change):
    fn = definition()
    guard, roots = combined_guard(native_factory, (fn,))
    if change == "code":
        fn.__code__ = (lambda value: value).__code__
    elif change == "defaults":
        fn.__defaults__ = (3,)
    elif change == "defaults_none":
        fn.__defaults__ = None
    elif change == "keywords":
        fn.__kwdefaults__ = {"scale": 3}
    elif change == "keywords_none":
        fn.__kwdefaults__ = None
    elif change == "function_dictionary":
        fn.__dict__ = {"marker": 1}
    elif change == "function_dictionary_write":
        fn.marker = 1
    elif change == "keywords_write":
        fn.__kwdefaults__["scale"] = 3
    else:
        fn.__globals__["changed"] = 1
    assert not guard.check(roots)


@pytest.mark.parametrize("empty", [False, True])
def test_closure_content_changes_are_observed(native_factory, empty):
    value = object()

    def fn():
        return value

    guard, roots = combined_guard(native_factory, (fn,))
    if empty:
        del fn.__closure__[0].cell_contents
    else:
        fn.__closure__[0].cell_contents = object()
    assert not guard.check(roots)


def test_none_slots_and_unsupported_inputs(native_factory):
    fn = lambda: None
    checker, dictionaries = function_metadata.prepare_checks(native_factory, (fn,), ())
    assert checker is not None and checker(None)
    assert dictionaries
    assert function_metadata.prepare_checks(native_factory, (len,), ()) == (None, ())
    value = object()

    def empty():
        return value

    del empty.__closure__[0].cell_contents
    assert function_metadata.prepare_checks(native_factory, (empty,), ()) == (None, ())


def test_native_mro_check_keeps_the_tuple_alive_and_observes_replacement(
    native_factory,
):
    class Base:
        pass

    class Alternate:
        pass

    class Owner(Base):
        pass

    checker, _ = function_metadata.prepare_checks(
        native_factory, (definition(),), (Owner,), with_class_mro=True
    )
    assert checker is not None and checker(None)
    old_mro = Owner.__mro__
    Owner.__bases__ = (Alternate,)
    assert Owner.__mro__ is not old_mro
    assert not checker(None)
    Owner.__bases__ = (Base,)
    # Even restoring the same bases creates a new tuple and must requalify.
    assert Owner.__mro__ == old_mro and Owner.__mro__ is not old_mro
    assert not checker(None)


@pytest.mark.parametrize("target", ["kernel", "chart", "geometry", "tensor"])
def test_native_mro_sharing_keeps_mutable_class_namespaces_live(
    native_factory, monkeypatch, target
):
    model, _opt, _dispatcher, _cached, (guard, roots) = example(
        function_factory=native_factory
    )
    owners = {
        "kernel": type(model.kernel),
        "chart": type(model.input_chart),
        "geometry": type(model.input_chart.geometry),
        "tensor": torch.Tensor,
    }
    assert guard.check(roots)
    monkeypatch.setattr(owners[target], "_test_guard_revision", object(), raising=False)
    assert not guard.check(roots)


def test_model_is_not_retained_by_the_native_function_records(native_factory):
    model, opt, dispatcher, cached, native = example(function_factory=native_factory)
    guard, roots = native
    assert len(roots) == 8
    refs = [
        weakref.ref(model),
        weakref.ref(model.atoms.p),
        weakref.ref(cached.state.binding),
    ]
    del roots, native, model, opt, dispatcher, cached
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert guard is not None


def _runtime_dictionary_guard(factory, builder, previous=None):
    importlib.import_module("torch._dynamo.guards")
    if builder == "linear":
        from test_radius_prepared_entry import case

        from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import prepared
        from torchcst.operators.execution import LinearInputs

        if previous is None:
            model, dispatcher, inputs, cached, program = case()
        else:
            model, dispatcher, cached = (
                previous["model"],
                previous["dispatcher"],
                previous["cached"],
            )
            inputs = LinearInputs(previous["input"])
            program = prepared.prepare_program(dispatcher, cached, inputs)
            assert program is not None
        roots = (model, dispatcher, inputs.x, *program.template[3:])
        guards = (program.callback_guard, program.guard)
        value = inputs.x
    else:
        if previous is None:
            model, _opt, dispatcher, cached, (guard, roots) = example(
                function_factory=factory
            )
        else:
            from torchcst._backends.cuda.algorithms.polar_update.optimizer_program import (
                _native_guard,
            )

            model, dispatcher, cached = (
                previous["model"],
                previous["dispatcher"],
                previous["cached"],
            )
            callbacks = tuple((owner, ()) for owner in previous["roots"][3])
            guard, roots = _native_guard(
                cached.state.binding,
                dispatcher,
                cached,
                callbacks,
                previous["roots"][4],
                model,
                function_factory=factory,
            )
        guards = (guard,)
        value = model.atoms.p.grad
    return {
        "model": model,
        "dispatcher": dispatcher,
        "cached": cached,
        "roots": roots,
        "guards": guards,
        "input": value,
    }


@pytest.mark.parametrize("builder", ["linear", "polar"])
@pytest.mark.parametrize("force_none", [False, True])
def test_runtime_builders_capture_optional_epochs_and_keep_tuple_identity_guards(
    native_factory, monkeypatch, builder, force_none
):
    original = function_metadata.prepare_dictionary_checks
    probe = {}
    watcher_available = original(native_factory, (probe,)) is not None
    captures = []

    def capture(factory, dictionaries):
        checker = None if force_none else original(factory, dictionaries)
        captures.append((tuple(id(value) for value in dictionaries), checker))
        return checker

    monkeypatch.setattr(function_metadata, "prepare_dictionary_checks", capture)
    actual = _runtime_dictionary_guard(native_factory, builder)
    roots, guards = actual["roots"], actual["guards"]
    assert all(guard.check(roots) for guard in guards)
    expected = roots[7] + (roots[8] if builder == "linear" else ())
    assert captures[-1][0] == tuple(id(value) for value in expected)
    assert (captures[-1][1] is not None) == (watcher_available and not force_none)
    assert len(roots) == (9 if builder == "linear" else 8)
    for index in (7, 8) if builder == "linear" else (7,):
        replacement = (*roots[index],)
        assert replacement is not roots[index]
        changed = (*roots[:index], replacement, *roots[index + 1 :])
        assert not all(guard.check(changed) for guard in guards)


@pytest.mark.parametrize("builder", ["linear", "polar"])
def test_runtime_builders_install_returned_native_checker_on_older_python(
    native_factory, monkeypatch, builder
):
    # Exercise the non-None branch even where dictionary watchers are absent.
    # This independent native slot checker changes without touching any actual
    # runtime dictionary, so only installation of the returned checker rejects.
    function = definition()
    checker = native_factory((function,), ())
    monkeypatch.setattr(
        function_metadata, "prepare_dictionary_checks", lambda *args: checker
    )
    actual = _runtime_dictionary_guard(native_factory, builder)
    roots, guards = actual["roots"], actual["guards"]
    assert all(guard.check(roots) for guard in guards)
    function.__defaults__ = (2,)
    assert not all(guard.check(roots) for guard in guards)


@pytest.mark.parametrize("builder", ["linear", "polar"])
@pytest.mark.parametrize("force_none", [False, True])
def test_runtime_dictionary_mutations_stay_invalid_until_resnapshot(
    native_factory, monkeypatch, builder, force_none
):
    if force_none:
        monkeypatch.setattr(
            function_metadata, "prepare_dictionary_checks", lambda *args: None
        )
    actual = _runtime_dictionary_guard(native_factory, builder)
    roots, guards = actual["roots"], actual["guards"]
    dictionary = (
        actual["cached"].context.__dict__
        if builder == "linear"
        else roots[4][0].__dict__
    )
    assert all(guard.check(roots) for guard in guards)
    dictionary["_test_epoch_event"] = object()
    del dictionary["_test_epoch_event"]
    assert not all(guard.check(roots) for guard in guards)
    fresh = _runtime_dictionary_guard(native_factory, builder, previous=actual)
    assert all(guard.check(fresh["roots"]) for guard in fresh["guards"])
    assert not all(guard.check(roots) for guard in guards)


@pytest.mark.parametrize("builder", ["linear", "polar"])
@pytest.mark.parametrize("force_none", [False, True])
@pytest.mark.parametrize("change", ["dictionary_pointer", "mro"])
def test_runtime_epochs_keep_function_slots_and_complete_mro_checks(
    native_factory, monkeypatch, builder, force_none, change
):
    if force_none:
        monkeypatch.setattr(
            function_metadata, "prepare_dictionary_checks", lambda *args: None
        )
    actual = _runtime_dictionary_guard(native_factory, builder)
    roots, guards = actual["roots"], actual["guards"]
    assert all(guard.check(roots) for guard in guards)
    if change == "dictionary_pointer":
        function = roots[5 if builder == "linear" else 4][0]
        monkeypatch.setattr(function, "__dict__", dict(function.__dict__))
        assert not all(guard.check(roots) for guard in guards)
    else:

        class AlternateModule(torch.nn.Module):
            pass

        owner = type(actual["model"].kernel)
        bases = owner.__bases__
        try:
            owner.__bases__ = (AlternateModule,)
            assert not all(guard.check(roots) for guard in guards)
        finally:
            owner.__bases__ = bases


@pytest.mark.parametrize("builder", ["linear", "polar"])
@pytest.mark.parametrize("force_none", [False, True])
def test_runtime_epoch_and_classic_guards_retain_no_model_or_input(
    native_factory, monkeypatch, builder, force_none
):
    original = function_metadata.prepare_dictionary_checks
    epochs = []

    def capture(factory, dictionaries):
        checker = None if force_none else original(factory, dictionaries)
        epochs.append(checker)
        return checker

    monkeypatch.setattr(function_metadata, "prepare_dictionary_checks", capture)
    actual = _runtime_dictionary_guard(native_factory, builder)
    refs = [
        weakref.ref(actual["model"]),
        weakref.ref(actual["model"].atoms.p),
        weakref.ref(actual["input"]),
    ]
    guards = actual["guards"]
    del actual
    gc.collect()
    assert all(reference() is None for reference in refs)
    assert guards and epochs  # Both native guard objects/checkers remain alive.


def test_batched_storage_checks_track_addresses_not_numeric_versions(native_factory):
    tensor = torch.ones(4)
    parameter = torch.nn.Parameter(torch.ones(4))
    checker = function_metadata.prepare_storage_checks(
        native_factory, (tensor, parameter)
    )
    assert checker is not None and checker(None)
    with torch.no_grad():
        tensor.add_(1)
        parameter.add_(1)
    assert checker(None)
    with torch.no_grad():
        parameter.data = parameter.data.clone()
    assert not checker(None)


def test_batched_storage_check_owns_no_tensor(native_factory):
    tensor = torch.ones(4)
    reference = weakref.ref(tensor)
    checker = function_metadata.prepare_storage_checks(native_factory, (tensor,))
    assert checker is not None and checker(None)
    del tensor
    gc.collect()
    assert reference() is None
    assert not checker(None)


@pytest.mark.parametrize(
    "target", ["instance", "class", "builtin_instance", "property", "getattribute"]
)
def test_storage_accessor_replacement_is_detected_without_calling_it(
    native_factory, monkeypatch, target
):
    tensor = torch.ones(4)
    checker = function_metadata.prepare_storage_checks(native_factory, (tensor,))
    assert checker is not None and checker(None)
    calls = []

    def custom(*args):
        calls.append(1)
        return 0

    if target == "instance":
        tensor.data_ptr = custom
    elif target == "builtin_instance":
        tensor.data_ptr = tensor.numel
    elif target == "property":
        monkeypatch.setattr(torch.Tensor, "data_ptr", property(custom))
    elif target == "getattribute":
        monkeypatch.setattr(torch.Tensor, "__getattribute__", custom)
    else:
        monkeypatch.setattr(torch.Tensor, "data_ptr", custom)
    assert not checker(None)
    assert not calls
    assert function_metadata.prepare_storage_checks(native_factory, (tensor,)) is None
    assert not calls


def test_native_program_uses_batched_storage_and_keeps_post_hook_gate(
    native_factory, monkeypatch
):
    calls = []
    original = function_metadata.prepare_storage_checks

    def capture(factory, tensors):
        result = original(factory, tensors)
        calls.append(result)
        return result

    monkeypatch.setattr(function_metadata, "prepare_storage_checks", capture)
    model, _opt, _dispatcher, _cached, (guard, roots) = example(
        function_factory=native_factory
    )
    assert len(roots) == 8 and calls[-1] is not None
    assert guard.check(roots)
    with torch.no_grad():
        fields = model.atom_state.optimizer_state.fields
        fields["exp_avg"].set_(fields["exp_avg"].clone())
    assert not guard.check(roots)


def test_unsupported_interpreter_keeps_existing_guard(monkeypatch):
    function_metadata.get_factory.cache_clear()
    try:
        monkeypatch.setattr(
            function_metadata,
            "sys",
            SimpleNamespace(implementation=SimpleNamespace(name="pypy")),
        )
        assert function_metadata.get_factory() is None
        assert function_metadata.prepare_checks(None, (definition(),), ()) == (None, ())
    finally:
        function_metadata.get_factory.cache_clear()


def test_compiler_failure_is_optional_and_process_cached(monkeypatch):
    import torch.utils.cpp_extension

    calls = []

    def fail(**kwargs):
        calls.append(1)
        raise RuntimeError("unavailable compiler")

    function_metadata.get_factory.cache_clear()
    try:
        monkeypatch.setattr(torch.utils.cpp_extension, "load", fail)
        assert function_metadata.get_factory() is None
        assert function_metadata.get_factory() is None
        assert calls == [1]
    finally:
        function_metadata.get_factory.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cold_native_preparation_rechecks_callbacks_after_the_loader(
    monkeypatch, native_factory
):
    from test_optimizer_amplitude_decay import gpu_optimizer
    from test_radius_polar_cuda_update import radius_model

    from torchcst._backends.torch.profiles import execution as profiles

    model = radius_model().cuda()
    opt = gpu_optimizer(model, 0.01)
    model.atoms.p.grad = torch.ones_like(model.atoms.p) * 0.01
    original = profiles.project_gradient
    loads, projections = [], []

    def projected(*args):
        projections.append(1)
        return original(*args) * 0.9

    def load():
        loads.append(1)
        monkeypatch.setattr(profiles, "project_gradient", projected)
        return native_factory

    monkeypatch.setattr(function_metadata, "get_factory", load)
    opt.step()
    assert loads == [1]
    assert opt._update_bindings[model].__dict__.get("_optimizer_program") is None
    opt.step()
    assert projections
