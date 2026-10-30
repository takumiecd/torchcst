"""Batched versions keep replacement, callback and weak-lifetime boundaries."""

import gc
import os
import weakref

import pytest
import torch

from torchcst.operators import _native_metadata_versions as native
from torchcst.operators import metadata


@pytest.fixture(scope="module")
def factory():
    if not torch.cuda.is_available() and os.getenv("CST_TEST_NATIVE_READ_LEASE") != "1":
        pytest.skip("explicit native-host compilation or CUDA required")
    result = native.prepare()
    assert result is not None
    return result


def test_version_counters_and_dead_buffers_are_checked_without_owning_storage(factory):
    values = [torch.ones(3), torch.ones(4)]
    check = factory(tuple(values))
    assert check is not None and check(None)
    values[-1].copy_(values[-1])  # Same value still changes the version.
    assert not check(None)
    check = factory(tuple(values))
    references = [weakref.ref(t) for t in values]
    del values
    gc.collect()
    assert all(r() is None for r in references)
    assert not check(None)


def test_custom_version_descriptor_and_tensor_subclasses_use_the_old_path(
    factory, monkeypatch
):
    class Custom(torch.Tensor):
        pass

    assert factory((torch.ones(3).as_subclass(Custom),)) is None
    assert factory((torch.nn.Parameter(torch.ones(3)),)) is None
    with torch.inference_mode():
        value = torch.ones(3)
    assert factory((value,)) is None
    value = torch.ones(3)
    check = factory((value,))
    monkeypatch.setattr(torch.Tensor, "_version", property(lambda self: 417))
    assert not check(None)
    assert factory((value,)) is None


def test_function_and_dispatch_modes_request_python_fallback(factory):
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    class FunctionMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    class DispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    # Keep this buffer alive: the guard itself deliberately owns no example.
    value = torch.ones(3)
    check = factory((value,))
    with FunctionMode():
        assert not check(None)
    with DispatchMode():
        assert not check(None)
    assert check(None)


@pytest.mark.parametrize("batched", [False, True])
def test_full_guard_detects_inplace_replacement_shape_and_class_changes(
    factory, monkeypatch, batched
):
    import torch._dynamo.guards  # Tests only; runtime never imports tracing here.

    monkeypatch.setattr(native, "_factory", factory if batched else None)
    root = torch.nn.Module()
    root.register_buffer("a", torch.ones(3))
    root.register_buffer("b", torch.ones(4))
    roots = (root,)
    signature = metadata.fixed_state_signature(roots)
    cache = root._fixed_signature_cache
    assert cache.guard is not None and cache.current(roots) is signature
    root.b.copy_(root.b)
    assert cache.current(roots) is None
    signature = metadata.fixed_state_signature(roots)
    assert cache.current(roots) is signature
    root._buffers = dict(root._buffers, b=torch.ones(4))
    assert cache.current(roots) is None
    metadata.fixed_state_signature(roots)
    assert cache.guard is not None and cache.current(roots) is not None
    root.b.resize_(5)
    assert cache.current(roots) is None
    metadata.fixed_state_signature(roots)

    class Custom(torch.Tensor):
        pass

    root.b.__class__ = Custom
    assert cache.current(roots) is None


def test_optional_compiler_failure_keeps_python_version_reads(factory, monkeypatch):
    import torch.utils.cpp_extension

    def unavailable(**kwargs):
        raise RuntimeError("deliberate optional compiler failure")

    monkeypatch.setattr(torch.utils.cpp_extension, "load", unavailable)
    monkeypatch.setattr(native, "_factory", factory)
    native.prepare.cache_clear()
    try:
        assert native.prepare() is None
        assert native._factory is None
    finally:
        native.prepare.cache_clear()


@pytest.mark.parametrize("name", ["dtype", "shape", "layout", "__getattribute__"])
def test_batch_epilogue_keeps_tensor_class_namespace_checks(factory, monkeypatch, name):
    import torch._dynamo.guards

    monkeypatch.setattr(native, "_factory", factory)
    root = torch.nn.Module()
    root.register_buffer("a", torch.ones(3))
    root.register_buffer("b", torch.ones(4))
    roots = (root,)
    metadata.fixed_state_signature(roots)
    cache = root._fixed_signature_cache
    assert cache.guard is not None and cache.current(roots) is not None
    if name == "__getattribute__":
        replacement = lambda self, key: object.__getattribute__(self, key)
    else:
        replacement = property(lambda self: None)
    monkeypatch.setattr(torch.Tensor, name, replacement)
    assert cache.current(roots) is None
