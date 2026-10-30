"""Native AdamW preserves CST numerics and one ordinary Optimizer lifecycle."""

import copy

import pytest
import torch
from test_optimizer_amplitude_decay import layer

from torchcst import CSTOptimizer
from torchcst.optim.adamw import CSTAdamW
from torchcst.optim.state import OptimizerStateAdapter


def make(model, *, legacy=False, **options):
    groups = [
        {"params": list(model["cst"].parameters()), "amplitude_weight_decay": 0.3},
        {"params": list(model["ordinary"].parameters()), "weight_decay": 0.17},
    ]
    options = {"lr": 0.02, "weight_decay": 0.0, "foreach": False} | options
    if legacy:
        return CSTOptimizer(torch.optim.AdamW(groups, **options), model=model)
    return CSTAdamW(groups, model=model, **options)


def model():
    return torch.nn.ModuleDict(
        {"cst": layer(), "ordinary": torch.nn.Linear(3, 2).double()}
    )


def gradients(models, seed, *, skip_atoms=False):
    rng = torch.Generator().manual_seed(seed)
    for parameters in zip(*(m.parameters() for m in models)):
        value = (
            torch.randn(parameters[0].shape, generator=rng, dtype=parameters[0].dtype)
            * 0.03
        )
        for parameter in parameters:
            parameter.grad = (
                None
                if skip_atoms and parameter.ndim == 2 and parameter.shape == (7, 4)
                else value.clone()
            )


def equal(left, right, lo, ro):
    for lp, rp in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(lp, rp, atol=0, rtol=0)
        assert lo.state.get(lp, {}).keys() == ro.state.get(rp, {}).keys()
        for key, value in lo.state.get(lp, {}).items():
            torch.testing.assert_close(value, ro.state[rp][key], atol=0, rtol=0)


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize(
    "amsgrad,maximize", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("check_finite", [False, True])
def test_multistep_bitwise_legacy_parameters_and_all_adam_state(
    foreach, amsgrad, maximize, check_finite
):
    actual = model()
    reference = copy.deepcopy(actual)
    native = make(actual, foreach=foreach, amsgrad=amsgrad, maximize=maximize)
    old = make(
        reference, legacy=True, foreach=foreach, amsgrad=amsgrad, maximize=maximize
    )
    native.check_finite = old.check_finite = check_finite
    assert isinstance(native, torch.optim.AdamW)
    assert not hasattr(native, "base_optimizer")
    for index, rate in enumerate([0.02, 0.0, 0.07, 0.03, 0.01]):
        for opt in (native, old):
            opt.param_groups[0]["lr"] = rate
            opt.param_groups[0]["amplitude_weight_decay"] = index * 0.1
        gradients([actual, reference], index, skip_atoms=index == 3)
        native.step()
        old.step()
        equal(actual, reference, native, old)


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_resume_and_explicit_legacy_migration(legacy):
    original = model()
    resumed = copy.deepcopy(original)
    source = make(original, legacy=legacy, amsgrad=True)
    gradients([original], 1)
    source.step()
    checkpoint = copy.deepcopy(source.state_dict())
    with torch.no_grad():
        for destination, parameter in zip(resumed.parameters(), original.parameters()):
            destination.copy_(parameter)
    target = make(resumed, amsgrad=True)
    if legacy:
        with pytest.raises(ValueError, match="checkpoint contract"):
            target.load_state_dict(checkpoint)
        target.load_legacy_state_dict(checkpoint)
    else:
        target.load_state_dict(checkpoint)
    for step in range(3):
        gradients([original, resumed], 10 + step)
        source.step()
        target.step()
        equal(original, resumed, source, target)
    assert target.state_dict()["cst"]["version"] == 3


def test_scheduler_live_prehook_and_single_global_local_hooks_see_final_values():
    actual = model()
    reference = copy.deepcopy(actual)
    opt, expected = make(actual), make(reference, legacy=True)
    schedule = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    calls = []
    from torch.optim.optimizer import (
        register_optimizer_step_post_hook,
        register_optimizer_step_pre_hook,
    )

    def global_pre(optimizer, args, kwargs):
        if optimizer is opt:
            calls.append("global_pre")

    def local_pre(optimizer, args, kwargs):
        calls.append("local_pre")
        optimizer.param_groups[0].update(lr=0.07, amplitude_weight_decay=0.6)

    snapshots = []

    def local_post(optimizer, args, kwargs):
        calls.append("local_post")
        snapshots.append(actual["cst"].atoms.p.detach().clone())

    def global_post(optimizer, args, kwargs):
        if optimizer is opt:
            calls.append("global_post")

    handles = [
        register_optimizer_step_pre_hook(global_pre),
        opt.register_step_pre_hook(local_pre),
        opt.register_step_post_hook(local_post),
        register_optimizer_step_post_hook(global_post),
    ]
    try:
        gradients([actual, reference], 13)
        expected.param_groups[0].update(lr=0.07, amplitude_weight_decay=0.6)
        opt.step()
        expected.step()
        equal(actual, reference, opt, expected)
        torch.testing.assert_close(snapshots[0], actual["cst"].atoms.p, atol=0, rtol=0)
        assert calls == ["global_pre", "local_pre", "local_post", "global_post"]
        schedule.step()
        assert opt.param_groups[0]["lr"] == 0.035
    finally:
        for handle in handles:
            handle.remove()


def test_closure_runs_once_with_grad_enabled_and_returns_loss():
    actual = model()
    opt = make(actual)
    calls = []

    def closure():
        calls.append(torch.is_grad_enabled())
        opt.zero_grad(set_to_none=True)
        loss = sum(p.square().sum() for p in actual.parameters())
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert calls == [True]
    assert loss.requires_grad
    assert all("step" in opt.state[p] for p in actual.parameters())


@pytest.mark.parametrize(
    "field,value",
    [
        ("weight_decay", 0.1),
        ("amplitude_weight_decay", -1),
        ("amplitude_weight_decay", float("nan")),
        ("amplitude_weight_decay", True),
        ("amplitude_weight_decay", 100),
        ("differentiable", True),
    ],
)
def test_invalid_live_groups_fail_before_parameter_or_state_mutation(field, value):
    actual = model()
    opt = make(actual)
    gradients([actual], 2)
    opt.step()
    previous = copy.deepcopy(actual)
    saved = [
        {k: v.clone() for k, v in opt.state[p].items()} for p in actual.parameters()
    ]
    opt.param_groups[0][field] = value
    with pytest.raises((ValueError, TypeError)):
        opt.step()
    for parameter, old, state in zip(actual.parameters(), previous.parameters(), saved):
        torch.testing.assert_close(parameter, old, atol=0, rtol=0)
        for key in state:
            torch.testing.assert_close(
                opt.state[parameter][key], state[key], atol=0, rtol=0
            )


def test_add_group_validation_is_atomic_and_new_owned_parameter_works():
    actual = model()
    opt = make(actual)
    before = len(opt.param_groups)
    with pytest.raises(ValueError, match="owned"):
        opt.add_param_group({"params": [torch.nn.Parameter(torch.ones(2))]})
    assert len(opt.param_groups) == before
    actual.register_parameter(
        "extra", torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    )
    opt.add_param_group({"params": [actual.extra], "weight_decay": 0.2})
    actual.extra.grad = torch.ones_like(actual.extra)
    opt.step()
    assert opt.state[actual.extra]["step"] == 1


def test_custom_state_transport_callback_remains_live_in_unchecked_steps():
    class CountingAdapter(OptimizerStateAdapter):
        def transport(self, site, old, new, state):
            calls.append((old.clone(), new.clone(), set(state)))
            super().transport(site, old, new, state)

    calls = []
    actual = layer()
    opt = CSTAdamW(
        actual.parameters(),
        model=actual,
        state_adapter=CountingAdapter(("exp_avg",)),
        check_finite=False,
        foreach=False,
    )
    for index in range(3):
        actual.atoms.p.grad = torch.full_like(actual.atoms.p, 0.03)
        opt.step()
    assert len(calls) == 3
    assert all("exp_avg" in keys for _, _, keys in calls)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("explicit_unscale", [False, True])
def test_gradscaler_success_and_overflow_skip_entire_coordinate_step(
    fused, explicit_unscale
):
    actual = layer().float().cuda()
    opt = CSTAdamW(
        actual.parameters(),
        model=actual,
        lr=0.02,
        amplitude_weight_decay=0.3,
        fused=fused,
        check_finite=False,
    )
    reference = copy.deepcopy(actual)
    plain = CSTAdamW(
        reference.parameters(),
        model=reference,
        lr=0.02,
        amplitude_weight_decay=0.3,
        fused=fused,
        check_finite=False,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=8.0)
    scaler.scale(actual.atoms.p.square().sum()).backward()
    reference.atoms.p.square().sum().backward()
    plain.step()
    if explicit_unscale:
        scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    equal(actual, reference, opt, plain)
    before = actual.atoms.p.detach().clone()
    state = {key: value.clone() for key, value in opt.state[actual.atoms.p].items()}
    opt.zero_grad(set_to_none=True)
    scaler.scale(actual.atoms.p.sum() * float("inf")).backward()
    if explicit_unscale:
        scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    torch.testing.assert_close(actual.atoms.p, before, atol=0, rtol=0)
    for key, value in state.items():
        torch.testing.assert_close(
            opt.state[actual.atoms.p][key], value, atol=0, rtol=0
        )
    assert state["step"] == 1


def test_checkpoint_hooks_observe_manifest_and_restored_owner_identity():
    actual = model()
    opt = make(actual)
    gradients([actual], 3)
    opt.step()
    observations = []

    def saved(optimizer, checkpoint):
        observations.append(checkpoint["cst"]["version"])

    def loaded(optimizer):
        point = actual["cst"].atoms.p
        observations.append(
            optimizer.state[point] is actual["cst"].atom_state.optimizer_state.fields
        )

    handles = [
        opt.register_state_dict_post_hook(saved),
        opt.register_load_state_dict_post_hook(loaded),
    ]
    try:
        checkpoint = copy.deepcopy(opt.state_dict())
        opt.load_state_dict(checkpoint)
    finally:
        for handle in handles:
            handle.remove()
    assert observations == [3, True]


def test_bad_moment_checkpoint_rejected_without_state_or_parameter_mutation():
    actual = model()
    opt = make(actual)
    gradients([actual], 4)
    opt.step()
    checkpoint = copy.deepcopy(opt.state_dict())
    atom_id = checkpoint["param_groups"][0]["params"][0]
    checkpoint["state"][atom_id]["exp_avg"] = torch.zeros(1)
    before = copy.deepcopy(actual)
    states = [
        {key: value.clone() for key, value in opt.state[p].items()}
        for p in actual.parameters()
    ]
    with pytest.raises(ValueError):
        opt.load_state_dict(checkpoint)
    for parameter, previous, state in zip(
        actual.parameters(), before.parameters(), states
    ):
        torch.testing.assert_close(parameter, previous, atol=0, rtol=0)
        for key, value in state.items():
            torch.testing.assert_close(opt.state[parameter][key], value, atol=0, rtol=0)


def test_add_foreign_owned_atom_group_rejects_without_mutating_groups():
    actual = model()
    foreign = layer()
    owner = CSTAdamW(foreign.parameters(), model=foreign, foreach=False)
    actual["foreign"] = foreign
    opt = make(actual)
    previous = tuple(opt.param_groups)
    with pytest.raises(ValueError, match="already has an optimizer updater"):
        opt.add_param_group({"params": list(foreign.parameters()), "weight_decay": 0.0})
    assert tuple(opt.param_groups) == previous
    assert foreign.atom_state.optimizer_state._updater() is owner


@pytest.mark.parametrize("foreach", [False, True])
def test_no_atom_default_adamw_matches_pytorch(foreach):
    actual = torch.nn.Linear(3, 2, dtype=torch.float64)
    reference = copy.deepcopy(actual)
    native = CSTAdamW(actual.parameters(), model=actual, foreach=foreach)
    plain = torch.optim.AdamW(reference.parameters(), foreach=foreach)
    assert native.param_groups[0]["weight_decay"] == 0.01
    for step in range(4):
        gradients([actual, reference], 20 + step)
        native.step()
        plain.step()
        equal(actual, reference, native, plain)


def test_named_parameter_generator_preserves_names_and_atom_default_decay():
    actual = layer()
    reference = copy.deepcopy(actual)
    opt = CSTAdamW(actual.named_parameters(), model=actual, foreach=False)
    plain = CSTAdamW(reference.parameters(), model=reference, foreach=False)
    assert opt.param_groups[0]["param_names"] == [
        name for name, _ in actual.named_parameters()
    ]
    assert opt.param_groups[0]["weight_decay"] == 0.0
    gradients([actual, reference], 81)
    opt.step()
    plain.step()
    equal(actual, reference, opt, plain)
    assert (
        opt.state_dict()["param_groups"][0]["param_names"]
        == opt.param_groups[0]["param_names"]
    )


def test_user_checkpoint_posthook_replacement_dictionary_is_preserved():
    actual = model()
    opt = make(actual)

    def replace(optimizer, checkpoint):
        return dict(checkpoint, user_metadata={"run": 17})

    handle = opt.register_state_dict_post_hook(replace)
    try:
        result = opt.state_dict()
    finally:
        handle.remove()
    assert result["user_metadata"] == {"run": 17}
    assert result["cst"]["version"] == 3
