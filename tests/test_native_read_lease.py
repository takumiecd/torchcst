"""Native identity protection preserves the original read-lease/autograd contract."""

import copy
import gc
import os
import weakref

import pytest
import torch
from test_cst_optimizer import site

from torchcst.atoms import Atoms, AtomState
from torchcst.atoms import _native_read_lease as native
from torchcst.atoms import state as atom_state


@pytest.fixture(scope="module")
def factory():
    if (
        not torch.cuda.is_available()
        and os.environ.get("CST_TEST_NATIVE_READ_LEASE") != "1"
    ):
        pytest.skip("explicit native-host compilation or CUDA required")
    result = native.prepare()
    assert result is not None, "native read lease did not qualify"
    return result


@pytest.fixture
def prepared(factory, monkeypatch):
    monkeypatch.setattr(native, "_prepared", factory)
    return factory


def owner():
    return AtomState(Atoms(torch.randn(3, 4, dtype=torch.float64)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex128])
@pytest.mark.parametrize("strided", [False, True])
def test_native_alias_metadata_identity_gradients_and_gradgrad(
    prepared, dtype, strided
):
    state = owner()
    x = torch.randn(3, 8 if strided else 4, dtype=dtype)
    x = x[:, ::2] if strided else x
    x.requires_grad_()
    y = state.protect_tensor(x)
    assert type(y.grad_fn).__name__ == "CppFunction"
    assert y.data_ptr() == x.data_ptr() and y.stride() == x.stride()
    assert y.dtype == x.dtype and y.device == x.device
    loss = y.abs().pow(4).sum()
    grad = torch.autograd.grad(loss, x, create_graph=True)[0]
    second = torch.autograd.grad(grad.abs().square().sum(), x)[0]
    reference = x.detach().requires_grad_()
    rg = torch.autograd.grad(
        reference.abs().pow(4).sum(), reference, create_graph=True
    )[0]
    rs = torch.autograd.grad(rg.abs().square().sum(), reference)[0]
    torch.testing.assert_close(grad, rg, rtol=0, atol=0)
    torch.testing.assert_close(second, rs, rtol=0, atol=0)


def test_retained_freed_graph_and_custom_view_inplace_errors(prepared):
    state = owner()
    y = state.protect_tensor(state.atoms.p)
    assert sum(r() is not None for r in state._readers) == 1
    with pytest.raises(RuntimeError, match="view|inplace"):
        y.add_(1)
    y.sum().backward(retain_graph=True)
    assert sum(r() is not None for r in state._readers) == 1
    with pytest.raises(RuntimeError, match="backward"):
        state.relayout(torch.tensor([2, 0, 1]))
    y.sum().backward()
    assert not any(r() is not None for r in state._readers)
    with pytest.raises(RuntimeError, match="second time|freed"):
        y.sum().backward()
    state.relayout(torch.tensor([2, 0, 1]))


def test_discarded_graph_releases_marker_and_factory_keeps_no_model(prepared):
    state = owner()
    y = state.protect_tensor(state.atoms.p)
    references = [weakref.ref(state), weakref.ref(state.atoms.p), state._readers[0]]
    del y, state
    gc.collect()
    assert all(r() is None for r in references)


@pytest.mark.parametrize("enabled", [False, True])
def test_pruning_is_bounded_but_never_discards_a_live_or_retained_lease(
    prepared, monkeypatch, enabled
):
    if not enabled:
        monkeypatch.setattr(native, "_prepared", None)
    state = owner()
    retained = state.protect_tensor(state.atoms.p)
    marker = state._readers[0]
    for _ in range(200):
        transient = state.protect_tensor(state.atoms.p)
        del transient
        assert len(state._readers) <= 33
        assert marker() is not None
    retained.sum().backward(retain_graph=True)
    with pytest.raises(RuntimeError, match="backward"):
        state.relayout(torch.tensor([2, 0, 1]))
    retained.sum().backward()
    state.relayout(torch.tensor([2, 0, 1]))
    assert state._readers == []


@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (True, False), (False, True)]
)
@pytest.mark.parametrize("offload", [False, True])
def test_module_and_direct_binding_leases_survive_offload_and_retained_vjps(
    prepared, needs_x, needs_p, offload
):
    from torchcst import (
        DefaultRecipe,
        Dispatcher,
        ExecutionPlan,
        LinearBinding,
        LinearInputs,
    )
    from torchcst._backends.catalog import get_registry

    model = site()
    model.atoms.p.requires_grad_(needs_p)
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=needs_x)
    binding = LinearBinding(model.operator)
    dispatcher = Dispatcher(registry=get_registry())
    plan = ExecutionPlan("torch_materialized", "v1", DefaultRecipe())
    packed = []

    def pack(t):
        packed.append(t.shape)
        return t.detach().clone() if offload else t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y1 = model(x)
        y2 = dispatcher.run(binding, LinearInputs(x), plan=plan)
    assert (0,) not in packed  # Internal markers retain their identity hooks.
    selected = tuple(t for t in (x, model.atoms.p) if t.requires_grad)
    first = torch.autograd.grad(y1.square().sum(), selected, retain_graph=True)
    second = torch.autograd.grad(y2.square().sum(), selected)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="backward"):
        model.atom_state.relayout(torch.tensor([2, 0, 1]))
    torch.autograd.grad(y1.square().sum(), selected)
    model.atom_state.relayout(torch.tensor([2, 0, 1]))
    clone = copy.deepcopy(model)
    assert clone.atom_state._readers == []


@pytest.mark.parametrize(
    "mutation",
    [
        "forward",
        "backward",
        "apply",
        "setup_context",
        "defaults",
        "keywords",
        "function_dict",
        "global_missing_none",
        "torch_empty",
        "owner_attribute",
    ],
)
def test_changed_callbacks_keep_the_original_path(prepared, monkeypatch, mutation):
    reader = atom_state._ReadParameter
    state = owner()
    calls = []
    if mutation in ("forward", "backward"):
        original = getattr(reader, mutation)

        def callback(*args):
            calls.append(mutation)
            return original(*args)

        monkeypatch.setattr(reader, mutation, staticmethod(callback))
    elif mutation == "apply":
        original = reader.apply.__func__

        def callback(cls, *args, **kwargs):
            calls.append("apply")
            return original(cls, *args, **kwargs)

        monkeypatch.setattr(reader, "apply", classmethod(callback))
    elif mutation == "setup_context":
        # Defining setup_context changes the Function's signature convention.
        monkeypatch.setattr(reader, "setup_context", staticmethod(lambda *args: None))
    elif mutation == "defaults":
        monkeypatch.setattr(reader.forward, "__defaults__", (None,))
    elif mutation == "keywords":
        monkeypatch.setattr(reader.forward, "__kwdefaults__", {})
    elif mutation == "function_dict":
        monkeypatch.setattr(reader.forward, "native_probe", True, raising=False)
    elif mutation == "global_missing_none":
        assert "super" not in reader.apply.__func__.__globals__
        monkeypatch.setitem(reader.apply.__func__.__globals__, "super", None)
    elif mutation == "torch_empty":
        original = torch.empty

        def callback(*args, **kwargs):
            calls.append("empty")
            return original(*args, **kwargs)

        monkeypatch.setattr(torch, "empty", callback)
    else:
        original = AtomState.__getattribute__

        def callback(self, name):
            if name == "_readers":
                calls.append("readers")
            return original(self, name)

        monkeypatch.setattr(AtomState, "__getattribute__", callback)
    if mutation in ("setup_context", "global_missing_none"):
        with pytest.raises(TypeError):
            state.protect_tensor(state.atoms.p)
        return
    y = state.protect_tensor(state.atoms.p)
    assert type(y.grad_fn).__name__ == "_ReadParameterBackward"
    y.sum().backward()
    if mutation in ("forward", "backward", "apply", "torch_empty", "owner_attribute"):
        assert calls


def test_modified_forward_code_keeps_callback(prepared, monkeypatch):
    fn = atom_state._ReadParameter.forward
    calls = []

    def changed(ctx, parameter, state):
        native_code_calls.append(1)  # noqa: F821 - rebound into the library globals below
        return native_original_forward(ctx, parameter, state)  # noqa: F821

    from types import FunctionType

    original = FunctionType(fn.__code__, fn.__globals__)
    monkeypatch.setitem(fn.__globals__, "native_code_calls", calls)
    monkeypatch.setitem(fn.__globals__, "native_original_forward", original)
    monkeypatch.setattr(fn, "__code__", changed.__code__)
    state = owner()
    y = state.protect_tensor(state.atoms.p)
    assert type(y.grad_fn).__name__ == "_ReadParameterBackward"
    y.sum().backward()
    assert calls == [1]


def test_tensor_subclass_and_owner_subclass_keep_legacy_path(prepared):
    class Custom(torch.Tensor):
        pass

    class CustomOwner(AtomState):
        pass

    state = owner()
    x = torch.randn(3, 4, requires_grad=True).as_subclass(Custom)
    y = state.protect_tensor(x)
    assert type(y.grad_fn).__name__ == "_ReadParameterBackward"
    other = CustomOwner(Atoms(torch.randn(3, 4)))
    assert (
        type(other.protect_tensor(other.atoms.p).grad_fn).__name__
        == "_ReadParameterBackward"
    )


@pytest.mark.parametrize("mode", ["function", "dispatch"])
def test_torch_modes_keep_legacy_path(prepared, mode):
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    class FunctionMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    class DispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    state = owner()
    with FunctionMode() if mode == "function" else DispatchMode():
        y = state.protect_tensor(state.atoms.p)
    assert type(y.grad_fn).__name__ == "_ReadParameterBackward"
    y.sum().backward()


def test_forward_ad_keeps_original_unsupported_jvp_behavior(prepared):
    state = owner()
    with torch.autograd.forward_ad.dual_level():
        x = torch.autograd.forward_ad.make_dual(
            state.atoms.p, torch.ones_like(state.atoms.p)
        )
        with pytest.raises(NotImplementedError, match="jvp"):
            state.protect_tensor(x)


def test_no_grad_and_invalid_owner_keep_existing_checks(prepared):
    state = owner()
    with torch.no_grad():
        assert state.protect_tensor(state.atoms.p) is state.atoms.p
    state._usable = False
    with pytest.raises(RuntimeError, match="invalid"):
        state.protect_tensor(state.atoms.p)


def test_compiler_failure_keeps_optional_python_path(monkeypatch):
    import torch.utils.cpp_extension

    previous = native._prepared
    native.prepare.cache_clear()
    monkeypatch.setattr(native, "_prepared", None)

    def failed(*args, **kwargs):
        raise RuntimeError("test compiler unavailable")

    monkeypatch.setattr(torch.utils.cpp_extension, "load", failed)
    try:
        assert native.prepare() is None
        state = owner()
        y = state.protect_tensor(state.atoms.p)
        assert type(y.grad_fn).__name__ == "_ReadParameterBackward"
        y.sum().backward()
    finally:
        native.prepare.cache_clear()
        native._prepared = previous
