"""A reusable group contract observes settings and ownership at each boundary."""

import copy
import importlib
import pickle

import pytest
import torch
from test_radius_polar_cuda_update import radius_model

from torchcst import CSTOptimizer


def prepared():
    importlib.import_module("torch._dynamo.guards")
    model = radius_model()
    opt = CSTOptimizer(
        torch.optim.AdamW(
            [{"params": list(model.parameters()), "amplitude_weight_decay": 0.01}],
            lr=0.001,
            weight_decay=0,
        ),
        model=model,
        check_finite=False,
    )
    mapping = opt._validate_groups(opt.param_groups)
    cached = opt.__dict__.get("_group_validation")
    if cached is None:
        pytest.skip("optional native group guard unavailable")
    assert cached.current(opt, opt.param_groups)
    return model, opt, cached, mapping


def test_group_contract_reuses_mapping_but_not_tensor_values():
    model, opt, cached, mapping = prepared()
    model.kernel.scalar("radial_regularization").mul_(1.1)
    with torch.no_grad():
        model.atoms.p.add_(0.01)
    assert cached.current(opt, opt.param_groups)
    assert opt._validate_groups(opt.param_groups) is mapping
    # There are no tensor-valued keys or copies in the cached lookup.
    assert set(mapping) == {id(p) for p in model.parameters()}


@pytest.mark.parametrize(
    "fault",
    ["rate", "decay", "raw_decay", "differentiable", "duplicate", "foreign"],
)
def test_invalid_group_changes_revalidate_before_parameter_or_state_writes(fault):
    model, opt, cached, _ = prepared()
    model.atoms.p.grad = torch.ones_like(model.atoms.p) * 0.01
    before = model.atoms.p.detach().clone()
    group = opt.param_groups[0]
    if fault == "rate":
        group["lr"] = float("nan")
    elif fault == "decay":
        group["amplitude_weight_decay"] = -0.1
    elif fault == "raw_decay":
        group["weight_decay"] = 0.1
    elif fault == "differentiable":
        group["differentiable"] = True
    elif fault == "duplicate":
        group["params"].append(model.atoms.p)
    else:
        group["params"].append(torch.nn.Parameter(torch.zeros(1)))
    assert not cached.current(opt, opt.param_groups)
    with pytest.raises(ValueError):
        opt.step()
    torch.testing.assert_close(model.atoms.p, before, atol=0, rtol=0)
    assert not opt.state.get(model.atoms.p)


@pytest.mark.parametrize(
    "fault", ["parameter", "submodule", "model", "sites", "groups"]
)
def test_registration_or_root_changes_cannot_reuse_group_contract(fault):
    model, opt, cached, _ = prepared()
    if fault == "parameter":
        model.atoms._parameters["p"] = torch.nn.Parameter(
            model.atoms.p.detach().clone()
        )
    elif fault == "submodule":
        model.input_chart = copy.deepcopy(model.input_chart)
    elif fault == "model":
        opt.model = copy.deepcopy(model)
    elif fault == "sites":
        opt._sites = ()
    else:
        opt.param_groups = [dict(group) for group in opt.param_groups]
    assert not cached.current(opt, opt.param_groups)


@pytest.mark.parametrize(
    "callback", ["walker", "code", "chart_method", "trainable", "math"]
)
def test_callback_changes_keep_fresh_validation(monkeypatch, callback):
    model, opt, cached, _ = prepared()
    if callback == "walker":
        original = model.named_parameters
        monkeypatch.setattr(
            model, "named_parameters", lambda *a, **kw: original(*a, **kw)
        )
    elif callback == "code":
        original = torch.nn.Module.named_parameters

        def walk(self, *args, **kwargs):
            return iter(())

        monkeypatch.setattr(original, "__code__", walk.__code__)
    elif callback == "chart_method":
        original = model.cst_charts
        monkeypatch.setattr(model, "cst_charts", lambda: original())
    elif callback == "trainable":
        from torchcst.charts.base import ChartSpec

        def trainable(self):
            return True

        monkeypatch.setattr(ChartSpec.trainable.fget, "__code__", trainable.__code__)
    else:
        import math

        monkeypatch.setattr(math, "isfinite", lambda value: False)
    assert not cached.current(opt, opt.param_groups)
    if callback in ("code", "trainable", "math"):
        with pytest.raises(ValueError):
            opt._validate_groups(opt.param_groups)
    else:
        opt._validate_groups(opt.param_groups)
        assert opt.__dict__.get("_group_validation") is None


def test_chart_spec_change_does_not_skip_trainable_geometry_validation():
    model, opt, cached, _ = prepared()
    from types import SimpleNamespace

    model.input_chart.spec = SimpleNamespace(trainable=True)
    assert not cached.current(opt, opt.param_groups)
    with pytest.raises(ValueError, match="non-Euclidean"):
        opt._validate_groups(opt.param_groups)


def test_pre_hook_changes_revalidate_with_warmed_group_contract():
    model, opt, cached, _ = prepared()
    model.atoms.p.grad = torch.ones_like(model.atoms.p) * 0.01
    before = model.atoms.p.detach().clone()

    def invalid(base, args, kwargs):
        base.param_groups[0]["weight_decay"] = 0.1

    handle = opt.base_optimizer.register_step_pre_hook(invalid)
    try:
        with pytest.raises(ValueError, match="raw weight_decay"):
            opt.step()
    finally:
        handle.remove()
    assert not cached.current(opt, opt.param_groups)
    torch.testing.assert_close(model.atoms.p, before, atol=0, rtol=0)
    assert not opt.state.get(model.atoms.p)


def test_valid_settings_change_rebuilds_contract_and_updates_are_bitwise_equal():
    model, opt, cached, _ = prepared()
    baseline = copy.deepcopy(model)
    reference = CSTOptimizer(
        torch.optim.AdamW(
            [{"params": baseline.parameters(), "amplitude_weight_decay": 0.01}],
            lr=0.001,
            weight_decay=0,
        ),
        model=baseline,
        check_finite=True,
    )
    generator = torch.Generator().manual_seed(192)
    for index in range(5):
        if index == 2:
            for optimizer in (opt, reference):
                optimizer.param_groups[0]["lr"] = 0.002
                optimizer.param_groups[0]["amplitude_weight_decay"] = 0.03
            assert not cached.current(opt, opt.param_groups)
        gradient = torch.randn(model.atoms.p.shape, generator=generator) * 0.01
        model.atoms.p.grad, baseline.atoms.p.grad = gradient.clone(), gradient.clone()
        opt.step()
        reference.step()
        torch.testing.assert_close(model.atoms.p, baseline.atoms.p, atol=0, rtol=0)
        for key, value in reference.state[baseline.atoms.p].items():
            torch.testing.assert_close(
                opt.state[model.atoms.p][key], value, atol=0, rtol=0
            )
    assert opt._group_validation.current(opt, opt.param_groups)
    assert reference.__dict__.get("_group_validation") is None


def test_mutable_group_validation_values_do_not_cache():
    _model, opt, _cached, _ = prepared()
    opt.param_groups[0]["differentiable"] = torch.tensor(False)
    opt._validate_groups(opt.param_groups)
    assert opt.__dict__.get("_group_validation") is None


def test_group_contract_drops_native_references_on_serialization():
    _model, opt, cached, _ = prepared()
    restored = pickle.loads(pickle.dumps(cached))
    assert not restored.current(opt, opt.param_groups)
    assert restored.mapping is None


@pytest.mark.parametrize("mode", ["function", "dispatch"])
def test_tensor_modes_use_fresh_validation(mode):
    _model, opt, cached, _ = prepared()
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    class FunctionMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    class DispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    with FunctionMode() if mode == "function" else DispatchMode():
        assert not cached.current(opt, opt.param_groups)
        opt._validate_groups(opt.param_groups)
        assert opt.__dict__.get("_group_validation") is None


@pytest.mark.parametrize("callback", ["site", "child_walker"])
def test_custom_callbacks_are_not_called_again_for_cache_qualification(
    monkeypatch, callback
):
    model, opt, cached, _ = prepared()
    calls = []
    if callback == "site":
        original = model.cst_charts

        def charts():
            calls.append(1)
            return original()

        monkeypatch.setattr(model, "cst_charts", charts)
    else:
        original = model.input_chart.named_modules

        def walker(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(model.input_chart, "named_modules", walker)
    assert not cached.current(opt, opt.param_groups)
    opt._validate_groups(opt.param_groups)
    assert calls == [1]
    assert opt.__dict__.get("_group_validation") is None


def test_custom_trainable_attribute_does_not_break_fallback(monkeypatch):
    from torchcst.charts.base import ChartState

    _model, opt, cached, _ = prepared()
    monkeypatch.setattr(ChartState, "trainable", False)
    assert not cached.current(opt, opt.param_groups)
    opt._validate_groups(opt.param_groups)
    assert opt.__dict__.get("_group_validation") is None


@pytest.mark.parametrize(
    "shadow", ["input_chart", "atom_state", "atoms", "point", "chart"]
)
def test_instance_shadowing_of_registration_reads_invalidates_contract(shadow):
    model, opt, cached, _ = prepared()
    if shadow == "input_chart":
        model.__dict__["input_chart"] = copy.deepcopy(model.input_chart)
    elif shadow == "atom_state":
        model.__dict__["atom_state"] = copy.deepcopy(model.atom_state)
    elif shadow == "atoms":
        model.atom_state.__dict__["atoms"] = copy.deepcopy(model.atoms)
    elif shadow == "point":
        model.atoms.__dict__["p"] = torch.nn.Parameter(model.atoms.p.detach().clone())
    else:
        model.__dict__["chart"] = copy.deepcopy(model.input_chart)
    assert not cached.current(opt, opt.param_groups)


def test_group_guard_does_not_retain_an_optimizer_after_collection():
    import gc
    import weakref

    model, opt, cached, mapping = prepared()
    guard = cached.guard
    refs = [weakref.ref(model), weakref.ref(opt), weakref.ref(model.atoms.p)]
    del model, opt, cached, mapping
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert guard is not None
