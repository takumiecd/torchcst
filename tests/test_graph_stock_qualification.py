"""Finite clean-bootstrap qualification, independently of CUDA capture."""

import functools
import gc
import os
import subprocess
import sys
import weakref
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import _reduction
from torch.nn import functional as F
from torch.optim import Optimizer

# Bootstrap precedes the models and callbacks made by these tests.
from benchmarks.training import graph_stock_qualification as stock


def cohort():
    model = nn.Sequential(
        nn.Linear(1, 2),
        nn.ReLU(),
        nn.Linear(2, 2, bias=False),
        nn.ReLU(),
        nn.Linear(2, 2),
    )
    return model, torch.optim.AdamW(model.parameters())


def test_canonical_stock_and_real_updates_are_read_only_for_qualification():
    model, optimizer = cohort()
    qualified = stock.StockQualification(model, optimizer, "dense")
    assert qualified.check()
    inputs, labels = torch.ones(4, 1), torch.zeros(4, dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(inputs), labels).backward()
    optimizer.step()
    before = [
        (p.detach().clone(), p.grad.clone(), p._version) for p in model.parameters()
    ]
    clocks = [state["step"].clone() for state in optimizer.state.values()]
    assert qualified.check()
    assert qualified.check()
    for p, (value, grad, version) in zip(model.parameters(), before):
        assert (
            torch.equal(p, value)
            and torch.equal(p.grad, grad)
            and p._version == version
        )
    assert all(
        torch.equal(state["step"], clock)
        for state, clock in zip(optimizer.state.values(), clocks)
    )


def test_actual_radius_cst_module_types_and_profiled_cstadamw():
    from benchmarks.training.cell_mass_mnist import make_radius_cst
    from torchcst.optim import CSTAdamW

    middle = make_radius_cst(
        1024, torch.tensor([[0.3, 0.2, 0.4, 4.0]]), radius_execution="global-response4"
    )
    model = nn.Sequential(
        nn.Linear(784, 1024), nn.ReLU(), middle, nn.ReLU(), nn.Linear(1024, 10)
    )
    optimizer = CSTAdamW(model.parameters(), model=model)
    qualified = stock.StockQualification(model, optimizer, "cst")
    assert qualified.check()
    assert len(qualified._nodes) == 17
    assert optimizer.step.__func__.__wrapped__ is stock._RAW_STEPS[CSTAdamW]


@pytest.mark.parametrize("where", ["root", "relu", "linear", "optimizer"])
def test_cold_instance_override_rejected_without_calling_it(where):
    model, optimizer = cohort()
    calls = []

    def custom(*args, **kwargs):
        calls.append(True)
        raise AssertionError("unqualified callback ran")

    owner = {
        "root": model,
        "relu": model[1],
        "linear": model[0],
        "optimizer": optimizer,
    }[where]
    owner.__dict__["step" if where == "optimizer" else "forward"] = custom
    with pytest.raises(stock.QualificationError):
        stock.StockQualification(model, optimizer, "dense")
    assert not calls


@pytest.mark.parametrize(
    "name", ["named_modules", "named_parameters", "modules", "__iter__"]
)
def test_cold_traversal_override_rejected_before_traversal(monkeypatch, name):
    model, optimizer = cohort()
    calls = []

    def custom(*args, **kwargs):
        calls.append(True)
        raise AssertionError("custom traversal ran")

    monkeypatch.setattr(nn.Sequential, name, custom)
    with pytest.raises(stock.QualificationError):
        stock.StockQualification(model, optimizer, "dense")
    assert not calls


@pytest.mark.parametrize(
    "change",
    ["class", "code", "defaults", "dependency", "native", "compiled", "inplace"],
)
def test_known_live_callable_and_configuration_holes_are_rejected(monkeypatch, change):
    model, optimizer = cohort()
    qualified = stock.StockQualification(model, optimizer, "dense")
    if change == "class":
        monkeypatch.setattr(nn.ReLU, "forward", lambda self, value: value)
    elif change == "code":

        def replacement(self, value):
            return value

        monkeypatch.setattr(nn.Linear.forward, "__code__", replacement.__code__)
    elif change == "defaults":
        monkeypatch.setattr(
            F.cross_entropy,
            "__defaults__",
            F.cross_entropy.__defaults__[:-2] + ("sum", 0.0),
        )
    elif change == "dependency":
        monkeypatch.setattr(_reduction, "get_enum", lambda value: 2)
    elif change == "native":
        monkeypatch.setattr(torch._C._nn, "cross_entropy_loss", lambda *args: None)
    elif change == "compiled":
        model.__dict__["_compiled_call_impl"] = lambda *args: None
    elif change == "inplace":
        model[1].inplace = True
    expected = qualified._check_python()
    actual = qualified.check()
    assert not actual or expected  # Native acceptance must imply the full oracle.
    assert not expected
    assert not actual
    with pytest.raises(stock.QualificationError):
        stock.StockQualification(model, optimizer, "dense")


def test_profiled_stock_step_accepted_but_wrapped_custom_inner_rejected(monkeypatch):
    model, optimizer = cohort()
    assert type(optimizer).step.hooked
    assert stock.StockQualification(model, optimizer, "dense").check()
    raw = stock._RAW_STEPS[torch.optim.AdamW]

    @functools.wraps(raw)
    def custom(*args, **kwargs):
        return None

    custom.hooked = True
    monkeypatch.setattr(torch.optim.AdamW, "step", custom)
    with pytest.raises(stock.QualificationError):
        stock.StockQualification(model, optimizer, "dense")
    wrapped = Optimizer.profile_hook_step(custom)
    wrapped.hooked = True
    monkeypatch.setattr(torch.optim.AdamW, "step", wrapped)
    with pytest.raises(stock.QualificationError):
        stock.StockQualification(model, optimizer, "dense")


def test_exact_bootstrap_disable_cache_is_checked(monkeypatch):
    model, optimizer = cohort()
    qualified = stock.StockQualification(model, optimizer, "dense")
    raw, cached = stock._DYNAMO_CACHES[0]
    assert raw.__dynamo_disable is cached
    monkeypatch.setattr(raw, "__dynamo_disable", lambda *args, **kwargs: None)
    assert not qualified.check()
    with pytest.raises(stock.QualificationError):
        stock.StockQualification(model, optimizer, "dense")


def test_unproven_preexisting_disable_cache_is_not_blessed_or_overwritten():
    source = """
import torch
from torch.optim import Optimizer
raw = Optimizer.zero_grad.__wrapped__
custom = lambda *args, **kwargs: None
raw.__dynamo_disable = custom
from benchmarks.training import graph_stock_qualification as s
assert not s._BOOTSTRAP_SUPPORTED
assert raw.__dynamo_disable is custom
assert not s._DYNAMO_CACHES
"""
    root = Path(__file__).resolve().parents[1]
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=os.pathsep.join((str(root / "src"), str(root))),
    )
    subprocess.run(
        [sys.executable, "-c", source],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_certificate_does_not_retain_model_optimizer_parameters_or_old_dicts():
    model, optimizer = cohort()
    qualified = stock.StockQualification(model, optimizer, "dense")
    module_ref, optimizer_ref = weakref.ref(model), weakref.ref(optimizer)
    parameters = [weakref.ref(p) for p in model.parameters()]
    old_module = model[2]
    old_weight = weakref.ref(old_module.weight)
    old_module_ref = weakref.ref(old_module)
    old_modules = model._modules
    model._modules = dict(old_modules)
    del model._modules["2"]
    optimizer.param_groups[0]["params"] = [
        p for p in optimizer.param_groups[0]["params"] if p is not old_module.weight
    ]
    del old_modules, old_module
    gc.collect()
    assert old_module_ref() is None and old_weight() is None
    assert not qualified.check()
    del model, optimizer
    gc.collect()
    assert module_ref() is None and optimizer_ref() is None
    assert all(ref() is None for ref in parameters)


def test_unknown_cold_closure_is_not_retained_after_rejection():
    model, optimizer = cohort()
    tensor = torch.ones(1)
    ref = weakref.ref(tensor)

    def make_custom(value):
        def custom(*args, **kwargs):
            return value

        return custom

    model.forward = make_custom(tensor)
    with pytest.raises(stock.QualificationError):
        stock.StockQualification(model, optimizer, "dense")
    del model.forward, tensor
    gc.collect()
    assert ref() is None


def test_sticky_rejection_and_no_new_grad_retention():
    model, optimizer = cohort()
    qualified = stock.StockQualification(model, optimizer, "dense")
    p = next(model.parameters())
    p.grad = torch.ones_like(p)
    old_grad = weakref.ref(p.grad)
    assert qualified.check()
    p.grad = None
    gc.collect()
    assert old_grad() is None and qualified.check()
    model[1].inplace = True
    assert not qualified.check()
    model[1].inplace = False
    assert not qualified.check()


@pytest.mark.parametrize("change", ["replace", "new_alias", "none", "rebind"])
def test_raw_edges_rejected_before_new_traversal_even_if_old_nodes_live(change):
    model, optimizer = cohort()
    qualified = stock.StockQualification(model, optimizer, "dense")
    assert qualified.model_ref() is model and qualified.optimizer_ref() is optimizer
    old = model[2]
    calls = []

    def custom(*args, **kwargs):
        calls.append(True)
        raise AssertionError("unqualified nested traversal ran")

    if change == "replace":
        model[2] = nn.Linear(2, 2)
        model[2].named_modules = custom
    elif change == "new_alias":
        old.add_module("alias", model[1])
    elif change == "none":
        old.add_module("none", None)
    else:
        old._modules = dict(old._modules)
    assert old is not None  # Explicitly keep the original owner alive.
    assert not qualified.check()
    assert not calls


def test_cold_custom_registration_dictionary_is_not_iterated():
    model, optimizer = cohort()
    calls = []

    class CustomDict(dict):
        def items(self):
            calls.append(True)
            raise AssertionError("unqualified parameter registration ran")

    model[0]._parameters = CustomDict(model[0]._parameters)
    with pytest.raises(stock.QualificationError):
        stock.StockQualification(model, optimizer, "dense")
    assert not calls


def test_optional_native_unsupported_uses_complete_python_predicate(monkeypatch):
    from torchcst._backends.cuda import _host_function_metadata as metadata

    model, optimizer = cohort()
    monkeypatch.setattr(metadata, "get_factory", lambda: None)
    qualified = stock.StockQualification(model, optimizer, "dense")
    assert qualified.native_backend == "python"
    assert qualified.check() and qualified._check_python()
    model[1].inplace = True
    assert not qualified.check() and not qualified._check_python()


def test_native_runtime_profile_example_is_not_retained():
    model, optimizer = cohort()
    owner = torch.optim.AdamW
    saved = owner.step
    raw = stock._RAW_STEPS[owner]
    fresh = Optimizer.profile_hook_step(raw)
    fresh.hooked = True
    owner.step = fresh
    try:
        qualified = stock.StockQualification(model, optimizer, "dense")
        if qualified.native_backend != "native":
            pytest.skip("optional existing native helpers unavailable")
        reference = weakref.ref(fresh)
        owner.step = saved
        del fresh
        gc.collect()
        assert reference() is None
        assert (
            qualified._check_python()
        )  # Stock shell is valid, but reanchoring is forbidden.
        assert not qualified.check()  # Native certificate conservatively stays invalid.
    finally:
        owner.step = saved


def test_native_closure_drift_rejected_without_reanchoring(monkeypatch):
    model, optimizer = cohort()
    qualified = stock.StockQualification(model, optimizer, "dense")
    if qualified.native_backend != "native":
        pytest.skip("optional existing native helpers unavailable")
    raw = stock._RAW_STEPS[torch.optim.AdamW]
    cell = raw.__closure__[0]
    monkeypatch.setattr(cell, "cell_contents", lambda *args, **kwargs: None)
    assert not qualified._check_python()
    assert not qualified.check()
    assert qualified._native is None
