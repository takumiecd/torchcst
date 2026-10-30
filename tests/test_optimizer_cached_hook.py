"""A reusable registration remains last and retains no step invocation data."""

import copy
import gc
import weakref
from collections import OrderedDict

import pytest
import torch
from test_optimizer_amplitude_decay import layer, optimizer

from torchcst.optim import _amplitude_hook


def setup():
    model = layer()
    opt = optimizer(model, 0.03)
    opt.check_finite = False
    return model, opt


def step(model, opt):
    model.atoms.p.grad = torch.ones_like(model.atoms.p) * 0.01
    opt.step()


def test_registration_and_callable_reused_without_invocation_retention():
    model, opt = setup()
    step(model, opt)
    hook = opt._amplitude_pre_hook
    handle = hook.handle
    for _ in range(3):
        step(model, opt)
        assert opt._amplitude_pre_hook is hook
        assert hook.handle is handle
        assert hook.callback is None
        assert list(opt.base_optimizer._optimizer_step_pre_hooks.values()) == [hook]


@pytest.mark.parametrize("replace_table", [False, True])
def test_later_user_hook_runs_before_preparation_and_matches_temporary_path(
    monkeypatch, replace_table
):
    model, opt = setup()
    reference = copy.deepcopy(model)
    other = optimizer(reference, 0.03)
    other.check_finite = False
    install = _amplitude_hook.install

    def choose(owner, callback, *, reuse):
        return install(owner, callback, reuse=reuse and owner is not other)

    monkeypatch.setattr(_amplitude_hook, "install", choose)
    step(model, opt)
    step(reference, other)
    if replace_table:
        opt.base_optimizer._optimizer_step_pre_hooks = OrderedDict(
            opt.base_optimizer._optimizer_step_pre_hooks
        )
    calls = []

    def hook(base, args, kwargs):
        calls.append(base)
        base.param_groups[0]["lr"] = 0.003
        base.param_groups[0]["amplitude_weight_decay"] = 0.07
        base.param_groups[0]["params"][0].grad.mul_(0.3)

    handles = [
        base.register_step_pre_hook(hook)
        for base in (opt.base_optimizer, other.base_optimizer)
    ]
    for _ in range(3):
        step(model, opt)
        step(reference, other)
        torch.testing.assert_close(model.atoms.p, reference.atoms.p, atol=0, rtol=0)
        for key in opt.state[model.atoms.p]:
            torch.testing.assert_close(
                opt.state[model.atoms.p][key],
                other.state[reference.atoms.p][key],
                atol=0,
                rtol=0,
            )
        assert opt._amplitude_pre_hook.callback is None
    assert len(calls) == 6
    for handle in handles:
        handle.remove()
    step(model, opt)
    step(reference, other)
    torch.testing.assert_close(model.atoms.p, reference.atoms.p, atol=0, rtol=0)


@pytest.mark.parametrize("failure", ["group", "pre_hook", "closure"])
def test_failure_detaches_cached_hook(failure):
    model, opt = setup()
    step(model, opt)
    before = model.atoms.p.detach().clone()
    handle = None
    if failure == "group":
        opt.param_groups[0]["lr"] = -1
    elif failure == "pre_hook":

        def fail(*args):
            raise RuntimeError("user pre-hook")

        handle = opt.base_optimizer.register_step_pre_hook(fail)

    def closure():
        raise RuntimeError("user closure")

    with pytest.raises((ValueError, RuntimeError)):
        opt.step(closure if failure == "closure" else None)
    if handle is not None:
        handle.remove()
    assert not opt.base_optimizer._optimizer_step_pre_hooks
    assert "_amplitude_pre_hook" not in opt.__dict__
    torch.testing.assert_close(model.atoms.p, before, atol=0, rtol=0)


def test_closure_and_removed_decay_leave_no_dormant_hook():
    model, opt = setup()
    step(model, opt)
    calls = []

    def closure():
        calls.append(1)
        return torch.tensor(2.0)

    assert opt.step(closure).item() == 2
    assert calls == [1]
    assert not opt.base_optimizer._optimizer_step_pre_hooks
    step(model, opt)
    del opt.param_groups[0]["amplitude_weight_decay"]
    step(model, opt)
    assert not opt.base_optimizer._optimizer_step_pre_hooks


def test_custom_registrar_keeps_temporary_registration(monkeypatch):
    model, opt = setup()
    step(model, opt)
    original = opt.base_optimizer.register_step_pre_hook
    calls = []

    def register(callback):
        calls.append(callback)
        return original(callback)

    monkeypatch.setattr(opt.base_optimizer, "register_step_pre_hook", register)
    for _ in range(2):
        step(model, opt)
        assert not opt.base_optimizer._optimizer_step_pre_hooks
    assert len(calls) == 2


def test_base_optimizer_outlives_wrapper_without_retaining_it():
    model, opt = setup()
    step(model, opt)
    base = opt.base_optimizer
    reference = weakref.ref(opt)
    del opt
    gc.collect()
    assert reference() is None
    assert not base._optimizer_step_pre_hooks


@pytest.mark.parametrize("phase", ["pre", "post"])
def test_global_hooks_keep_order_and_step_values(monkeypatch, phase):
    from torch.optim.optimizer import (
        register_optimizer_step_post_hook,
        register_optimizer_step_pre_hook,
    )

    model, opt = setup()
    reference = copy.deepcopy(model)
    other = optimizer(reference, 0.03)
    other.check_finite = False
    original = _amplitude_hook.install

    def install(owner, callback, *, reuse):
        return original(owner, callback, reuse=reuse and owner is not other)

    monkeypatch.setattr(_amplitude_hook, "install", install)
    step(model, opt)
    step(reference, other)
    calls = []

    def hook(base, args, kwargs):
        if base in (opt.base_optimizer, other.base_optimizer):
            calls.append(base)
            base.param_groups[0]["lr"] *= 0.7
            base.param_groups[0]["amplitude_weight_decay"] = 0.08

    register = (
        register_optimizer_step_pre_hook
        if phase == "pre"
        else register_optimizer_step_post_hook
    )
    handle = register(hook)
    try:
        for _ in range(2):
            step(model, opt)
            step(reference, other)
            torch.testing.assert_close(model.atoms.p, reference.atoms.p, atol=0, rtol=0)
    finally:
        handle.remove()
    assert len(calls) == 4


def test_reentrant_step_rejects_before_adam_and_detaches_registration():
    model, opt = setup()
    step(model, opt)
    before = model.atoms.p.detach().clone()
    handle = opt.base_optimizer.register_step_pre_hook(lambda *args: opt.step())
    try:
        with pytest.raises(RuntimeError, match="reentrant"):
            opt.step()
    finally:
        handle.remove()
    assert not opt.base_optimizer._optimizer_step_pre_hooks
    torch.testing.assert_close(model.atoms.p, before, atol=0, rtol=0)
