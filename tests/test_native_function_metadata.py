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
