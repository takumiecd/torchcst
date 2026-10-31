"""Optional CPython watcher epochs complement function-slot metadata guards."""

import gc
import importlib
import os
import subprocess
import sys
import weakref
from types import SimpleNamespace

import pytest
import test_native_function_metadata as native_tests
import torch

from torchcst._backends.cuda import _host_function_metadata as metadata
from torchcst.operators._native_metadata_guard import guard_api

native_factory = native_tests.native_factory


@pytest.fixture
def watcher_factory(native_factory):
    dictionary = {}
    checker = metadata.prepare_dictionary_checks(native_factory, (dictionary,))
    if checker is None:
        pytest.skip("optional CPython 3.12+ dictionary watcher unavailable")
    assert checker(None)
    return native_factory


def watch(factory, *dictionaries):
    checker = metadata.prepare_dictionary_checks(factory, dictionaries)
    assert checker is not None and checker(None)
    return checker


def test_older_interpreters_compile_without_watcher_symbols(native_factory):
    if sys.version_info >= (3, 12):
        pytest.skip("old-interpreter fallback only")
    assert metadata.prepare_dictionary_checks(native_factory, ({},)) is None
    assert metadata._loaded_dictionary_factory(({},)) is None
    assert metadata._loaded_dictionary_stats() == {
        "supported": False,
        "watcher": -1,
        "dictionaries": 0,
        "registrations": 0,
    }


@pytest.mark.parametrize("failure", ["missing", "exception", "invalid", "unsupported"])
def test_optional_factory_failures_retain_fallback(monkeypatch, failure):
    factory = object()
    monkeypatch.setattr(metadata, "sys", SimpleNamespace(version_info=(3, 13)))
    monkeypatch.setattr(metadata, "_loaded_function_factory", factory)

    def native(dictionaries):
        if failure == "exception":
            raise RuntimeError("watcher registration unavailable")
        if failure == "unsupported":
            return None
        return lambda _: failure != "invalid"

    monkeypatch.setattr(
        metadata, "_loaded_dictionary_factory", None if failure == "missing" else native
    )
    assert metadata.prepare_dictionary_checks(factory, ({},)) is None
    assert metadata.prepare_dictionary_checks(None, ({},)) is None
    assert metadata.prepare_dictionary_checks(object(), ({},)) is None


def test_wrapper_qualifies_exact_dicts_before_native_call(monkeypatch):
    factory = object()
    calls = []
    monkeypatch.setattr(metadata, "sys", SimpleNamespace(version_info=(3, 13)))
    monkeypatch.setattr(metadata, "_loaded_function_factory", factory)

    def native(dictionaries):
        calls.append(dictionaries)
        return lambda _: True

    monkeypatch.setattr(metadata, "_loaded_dictionary_factory", native)

    class Custom(dict):
        pass

    assert metadata.prepare_dictionary_checks(factory, (Custom(),)) is None
    assert not calls
    dictionary = {}
    checker = metadata.prepare_dictionary_checks(factory, iter((dictionary,)))
    assert checker(None)
    assert len(calls) == 1 and calls[0][0] is dictionary


@pytest.mark.parametrize(
    "change",
    [
        "set",
        "update",
        "delete",
        "clear",
        "ior",
        "setdefault",
        "pop",
        "popitem",
        "restore",
    ],
)
def test_each_dictionary_event_stickily_invalidates(watcher_factory, change):
    original = object()
    dictionary = {"key": original}
    checker = watch(watcher_factory, dictionary)
    if change == "set":
        dictionary["key"] = object()
    elif change == "update":
        dictionary.update(key=object())
    elif change == "delete":
        del dictionary["key"]
    elif change == "clear":
        dictionary.clear()
    elif change == "ior":
        dictionary |= {"key": object()}
    elif change == "setdefault":
        dictionary.setdefault("other", object())
    elif change == "pop":
        dictionary.pop("key")
    elif change == "popitem":
        dictionary.popitem()
    else:
        dictionary["key"] = object()
        dictionary["key"] = original
    assert not checker(None)


def test_empty_dict_merge_clone_event_invalidates(watcher_factory):
    dictionary = {}
    checker = watch(watcher_factory, dictionary)
    dictionary.update({"first": object(), "second": object()})
    assert not checker(None)


def test_root_guard_observes_sticky_epoch_after_value_restoration(watcher_factory):
    importlib.import_module("torch._dynamo.guards")
    api = guard_api()
    if api is None:
        pytest.skip("optional RootGuard API unavailable")
    original = object()
    dictionary = {"key": original}
    checker = watch(watcher_factory, dictionary)
    root = api[0]()
    root.add_lambda_guard(checker, ["dictionary epoch"], None)
    assert root.check(None)
    dictionary["key"] = object()
    dictionary["key"] = original
    assert not root.check(None)
    fresh = watch(watcher_factory, dictionary)
    assert fresh(None) and not root.check(None)


@pytest.mark.parametrize(
    "dictionary_kind", ["keywords", "function", "globals", "class", "module"]
)
def test_callback_and_owner_dictionary_events(watcher_factory, dictionary_kind):
    fn = native_tests.definition()
    if dictionary_kind == "keywords":
        dictionary = fn.__kwdefaults__
        mutate = lambda: dictionary.update(scale=3)
    elif dictionary_kind == "function":
        dictionary = fn.__dict__
        mutate = lambda: setattr(fn, "marker", 1)
    elif dictionary_kind == "globals":
        dictionary = fn.__globals__
        mutate = lambda: dictionary.update(marker=1)
    elif dictionary_kind == "class":

        class Owner:
            pass

        dictionary = next(
            value for value in gc.get_referents(Owner.__dict__) if type(value) is dict
        )
        mutate = lambda: setattr(Owner, "marker", 1)
    else:
        import types

        owner = types.ModuleType("dictionary_epoch_fixture")
        dictionary = owner.__dict__
        mutate = lambda: setattr(owner, "marker", 1)
    checker = watch(watcher_factory, dictionary)
    mutate()
    assert not checker(None)


def test_dictionary_replacement_still_requires_function_slot_checker(watcher_factory):
    fn = native_tests.definition()
    slots, dictionaries = metadata.prepare_checks(watcher_factory, (fn,), ())
    epoch = watch(watcher_factory, *dictionaries)
    fn.__dict__ = {"replacement": True}
    # The old dictionary did not change. Only the existing slot checker can
    # detect that the owner now points to a different dictionary.
    assert epoch(None)
    assert not slots(None)


def test_multiple_epochs_and_duplicate_dictionaries(watcher_factory):
    dictionary = {"key": object()}
    before = metadata._loaded_dictionary_stats()
    first = watch(watcher_factory, dictionary, dictionary)
    assert (
        metadata._loaded_dictionary_stats()["registrations"]
        == before["registrations"] + 1
    )
    second = watch(watcher_factory, dictionary)
    assert first(None) and second(None)
    dictionary["key"] = object()
    third = watch(watcher_factory, dictionary)
    assert not first(None) and not second(None) and third(None)
    dictionary["key"] = object()
    assert not first(None) and not second(None) and not third(None)


def test_epoch_gc_prunes_registrations_for_long_lived_dictionary(watcher_factory):
    dictionary = {"value": object()}
    gc.collect()
    before = metadata._loaded_dictionary_stats()
    for _ in range(1000):
        checker = watch(watcher_factory, dictionary)
    del checker
    gc.collect()
    after = metadata._loaded_dictionary_stats()
    assert after["dictionaries"] == before["dictionaries"]
    assert after["registrations"] == before["registrations"]
    assert after["watcher"] == before["watcher"]


@pytest.mark.parametrize("cycle", [False, True])
def test_dictionary_deallocation_invalidates_without_retaining_tensor(
    watcher_factory, cycle
):
    tensor = torch.ones(2)
    reference = weakref.ref(tensor)
    dictionary = {"value": tensor}
    if cycle:
        dictionary["cycle"] = dictionary
    before = metadata._loaded_dictionary_stats()
    checker = watch(watcher_factory, dictionary)
    del dictionary, tensor
    gc.collect()
    assert reference() is None
    assert not checker(None)
    after = metadata._loaded_dictionary_stats()
    assert after["dictionaries"] == before["dictionaries"]
    assert after["registrations"] == before["registrations"]


def test_deallocated_dictionary_address_cannot_revalidate_old_epoch(watcher_factory):
    dictionary = {}
    address = id(dictionary)
    old = watch(watcher_factory, dictionary)
    del dictionary
    assert not old(None)
    # CPython often reuses the freed dictionary immediately, but allocator
    # behavior is not an API contract; the deterministic deallocation test is
    # separate above.
    replacement = None
    for _ in range(10000):
        candidate = {}
        if id(candidate) == address:
            replacement = candidate
            break
        del candidate
    if replacement is None:
        pytest.skip("allocator did not reuse the former dictionary address")
    fresh = watch(watcher_factory, replacement)
    assert not old(None) and fresh(None)
    del old
    assert fresh(None)
    replacement["changed"] = 1
    assert not fresh(None)


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="CPython watcher API needs 3.12+"
)
def test_exhausted_watcher_slots_fall_back_without_registrations(native_factory):
    code = r"""
import ctypes
from torchcst._backends.cuda import _host_function_metadata as metadata
factory = metadata.get_factory()
assert factory is not None
callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_void_p)
callback = callback_type(lambda *args: 0)
add = ctypes.pythonapi.PyDict_AddWatcher
add.argtypes = [callback_type]
add.restype = ctypes.c_int
clear = ctypes.pythonapi.PyDict_ClearWatcher
clear.argtypes = [ctypes.c_int]
clear.restype = ctypes.c_int
owned = []
try:
    for _ in range(32):
        try:
            owned.append(add(callback))
        except RuntimeError:
            break
    else:
        raise AssertionError('watcher slots unexpectedly unbounded')
    assert metadata.prepare_dictionary_checks(factory, ({},)) is None
    assert metadata.prepare_dictionary_checks(factory, ({},)) is None
    state = metadata._loaded_dictionary_stats()
    assert state['watcher'] == -1
    assert state['dictionaries'] == state['registrations'] == 0
finally:
    for watcher in owned:
        assert clear(watcher) == 0
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        env=os.environ.copy(),
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
