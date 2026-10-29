"""Amplitude decay follows an independent Polar law after the task update."""

import copy

import pytest
import torch
from test_radius_polar_cuda_update import radius_model

from torchcst import AtomUpdateInputs, CSTOptimizer


def layer():
    model = radius_model().double()
    with torch.no_grad():
        u = model.atoms.p.new_tensor([0.8, -0.7, 0.2, -0.3, 0.0, 1.0, -1.0])
        t = torch.copysign(
            (1 - u.square()).sqrt(), model.atoms.p.new_tensor([1, -1, 1, -1, 1, -1, 1])
        )
        model.atoms.p[:, :2] = 1.4 * torch.stack((u, t), dim=1)
        model.atoms.p[:, 2:] = model.atoms.p.new_tensor([2.17, 7.31])
    return model


_ABSENT = object()


def optimizer(model, decay=_ABSENT, *, lr=0.02, raw=0.0, kind=torch.optim.AdamW):
    group = {"params": list(model.parameters()), "weight_decay": raw}
    if decay is not _ABSENT:
        group["amplitude_weight_decay"] = decay
    return CSTOptimizer(kind([group], lr=lr, foreach=False), model=model)


def independent_decay(old, task, lr, decay):
    old_radius = old[:, :2].norm(dim=1)
    task_radius = task[:, :2].norm(dim=1)
    old_u = old[:, 0] / old_radius
    task_u = task[:, 0] / task_radius
    target_u = (task_u - lr * decay * old_u).clamp(-1, 1)
    expected = task.clone()
    expected[:, 0] = task_radius * target_u
    expected[:, 1] = torch.copysign(
        task_radius * (1 - target_u.square()).clamp_min(0).sqrt(), task[:, 1]
    )
    return expected


def pair_step(*, decay, lr=0.02, gradient=None):
    actual = layer()
    task = copy.deepcopy(actual)
    oa, ot = optimizer(actual, decay, lr=lr), optimizer(task, lr=lr)
    old = actual.atoms.p.detach().clone()
    gradient = torch.zeros_like(old) if gradient is None else gradient
    actual.atoms.p.grad, task.atoms.p.grad = gradient.clone(), gradient.clone()
    oa.step()
    ot.step()
    return actual, task, oa, ot, old


@pytest.mark.parametrize("decay", [0.0, 0.3, 2.0])
def test_zero_gradient_amplitude_scale_q_and_centers(decay):
    actual, task, _, _, old = pair_step(decay=decay)
    expected = independent_decay(old, task.atoms.p.detach(), 0.02, decay)
    torch.testing.assert_close(actual.atoms.p, expected, atol=3e-14, rtol=3e-14)
    torch.testing.assert_close(
        actual.atoms.p[:, :2].square().sum(1),
        task.atoms.p[:, :2].square().sum(1),
        atol=3e-14,
        rtol=3e-14,
    )
    torch.testing.assert_close(actual.atoms.p[:, 2:], old[:, 2:], atol=0, rtol=0)
    old_u = old[:, 0] / old[:, :2].norm(dim=1)
    new_u = actual.atoms.p[:, 0] / actual.atoms.p[:, :2].norm(dim=1)
    torch.testing.assert_close(
        new_u, old_u * (1 - 0.02 * decay), atol=3e-14, rtol=3e-14
    )


def test_task_update_then_subtracts_old_amplitude_and_preserves_moments():
    gradient = torch.zeros(7, 4, dtype=torch.float64)
    gradient[:, :2] = torch.tensor([0.8, -0.4], dtype=torch.float64)
    actual, task, oa, ot, old = pair_step(decay=0.7, lr=0.1, gradient=gradient)
    expected = independent_decay(old, task.atoms.p.detach(), 0.1, 0.7)
    torch.testing.assert_close(actual.atoms.p, expected, atol=3e-14, rtol=3e-14)
    # Using the post-task amplitude for decay is a different update law.
    wrong = independent_decay(task.atoms.p.detach(), task.atoms.p.detach(), 0.1, 0.7)
    assert float((expected - wrong).abs().max()) > 1e-5
    for key in ("exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(
            oa.state[actual.atoms.p][key], ot.state[task.atoms.p][key], atol=0, rtol=0
        )


def test_signed_zero_second_coordinate_and_clamped_amplitude():
    actual, task, _, _, old = pair_step(decay=0.8)
    expected = independent_decay(old, task.atoms.p.detach(), 0.02, 0.8)
    assert bool((task.atoms.p[-2:, 1] == 0).all())
    assert torch.equal(
        torch.signbit(actual.atoms.p[-2:, 1]), torch.signbit(task.atoms.p[-2:, 1])
    )
    torch.testing.assert_close(
        actual.atoms.p[-2:, :2], expected[-2:, :2], atol=3e-14, rtol=3e-14
    )
    gradient = torch.zeros(7, 4, dtype=torch.float64)
    gradient[0, 0] = 1
    actual, task, _, _, old = pair_step(decay=1 / 6, lr=6.0, gradient=gradient)
    task_u = task.atoms.p[0, 0].detach() / task.atoms.p[0, :2].detach().norm()
    old_u = old[0, 0] / old[0, :2].norm()
    assert float(task_u - old_u) < -1
    expected = independent_decay(old, task.atoms.p.detach(), 6.0, 1 / 6)
    torch.testing.assert_close(actual.atoms.p, expected, atol=3e-14, rtol=3e-14)
    assert (
        float(actual.atoms.p[0, 0].detach() / actual.atoms.p[0, :2].detach().norm())
        == -1
    )


def test_zero_decay_is_bitwise_legacy():
    gradient = torch.randn(
        7, 4, generator=torch.Generator().manual_seed(91), dtype=torch.float64
    )
    actual, task, oa, ot, _ = pair_step(decay=0.0, gradient=gradient)
    torch.testing.assert_close(actual.atoms.p, task.atoms.p, atol=0, rtol=0)
    for key in ("exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(
            oa.state[actual.atoms.p][key], ot.state[task.atoms.p][key], atol=0, rtol=0
        )


@pytest.mark.parametrize("kind", [torch.optim.Adam, torch.optim.AdamW])
def test_absent_option_keeps_legacy_raw_adam_behavior(kind):
    actual = layer()
    task = copy.deepcopy(actual)
    oa = optimizer(actual, lr=0.02, raw=0.1, kind=kind)
    ot = CSTOptimizer(
        kind(task.parameters(), lr=0.02, weight_decay=0.1, foreach=False), model=task
    )
    gradient = torch.ones_like(actual.atoms.p)
    actual.atoms.p.grad, task.atoms.p.grad = gradient.clone(), gradient.clone()
    oa.step()
    ot.step()
    torch.testing.assert_close(actual.atoms.p, task.atoms.p, atol=0, rtol=0)
    for key in ("exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(
            oa.state[actual.atoms.p][key], ot.state[task.atoms.p][key], atol=0, rtol=0
        )


@pytest.mark.parametrize(
    "value", [-0.1, float("nan"), float("inf"), True, None, torch.tensor(0.1), "0.1"]
)
def test_invalid_amplitude_decay_option_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        optimizer(layer(), value)


@pytest.mark.parametrize(
    "kind,lr,decay,raw",
    [
        (torch.optim.Adam, 0.1, 0.2, 0),
        (torch.optim.AdamW, 0.1, 0.2, 0.01),
        (torch.optim.AdamW, 0.6, 2.0, 0),
    ],
)
def test_incompatible_optimizer_raw_decay_or_decay_step_rejected(kind, lr, decay, raw):
    with pytest.raises((ValueError, TypeError)):
        optimizer(layer(), decay, lr=lr, raw=raw, kind=kind)


def test_atom_update_inputs_decay_defaults_to_zero():
    p = torch.zeros(2, 4)
    assert AtomUpdateInputs(p, 0.1).amplitude_weight_decay == 0


def test_checkpoint_persists_option_and_live_option_change_applies():
    model = layer()
    opt = optimizer(model, 0.2)
    model.atoms.p.grad = torch.zeros_like(model.atoms.p)
    opt.step()
    saved = copy.deepcopy(opt.state_dict())
    clone = copy.deepcopy(model)
    restored = optimizer(clone, 0.0)
    restored.load_state_dict(saved)
    assert restored.param_groups[0]["amplitude_weight_decay"] == 0.2
    restored.param_groups[0]["amplitude_weight_decay"] = 0.7
    before = clone.atoms.p.detach().clone()
    baseline = copy.deepcopy(clone)
    plain = optimizer(baseline)
    clone.atoms.p.grad = torch.zeros_like(clone.atoms.p)
    baseline.atoms.p.grad = torch.zeros_like(baseline.atoms.p)
    restored.step()
    plain.step()
    expected = independent_decay(
        before, baseline.atoms.p.detach(), restored.param_groups[0]["lr"], 0.7
    )
    torch.testing.assert_close(clone.atoms.p, expected, atol=3e-14, rtol=3e-14)


def test_mixed_ordinary_group_keeps_standard_adamw_and_state():
    cst = layer()
    ordinary = torch.nn.Linear(3, 2, dtype=torch.float64)
    reference = copy.deepcopy(ordinary)
    combined = torch.nn.ModuleDict({"cst": cst, "ordinary": ordinary})
    base = torch.optim.AdamW(
        [
            {
                "params": list(cst.parameters()),
                "weight_decay": 0.0,
                "amplitude_weight_decay": 0.4,
            },
            {"params": list(ordinary.parameters()), "weight_decay": 0.2},
        ],
        lr=0.03,
        foreach=False,
    )
    opt = CSTOptimizer(base, model=combined)
    plain = torch.optim.AdamW(
        reference.parameters(), lr=0.03, weight_decay=0.2, foreach=False
    )
    for _ in range(3):
        cst.atoms.p.grad = torch.zeros_like(cst.atoms.p)
        for actual, expected in zip(ordinary.parameters(), reference.parameters()):
            actual.grad = torch.ones_like(actual) * 0.3
            expected.grad = actual.grad.clone()
        opt.step()
        plain.step()
        for actual, expected in zip(ordinary.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(
                    opt.state[actual][key], plain.state[expected][key], atol=0, rtol=0
                )


def test_base_prehook_changes_lr_and_lambda_for_actual_step():
    actual = layer()
    task = copy.deepcopy(actual)
    opt = optimizer(actual, 0.2, lr=0.02)
    reference = optimizer(task, lr=0.07)
    before = actual.atoms.p.detach().clone()
    gradient = torch.zeros_like(before)
    gradient[:, :2] = gradient.new_tensor([0.4, -0.2])
    actual.atoms.p.grad, task.atoms.p.grad = gradient.clone(), gradient.clone()

    def change_group(base, args, kwargs):
        base.param_groups[0]["lr"] = 0.07
        base.param_groups[0]["amplitude_weight_decay"] = 0.6

    handle = opt.base_optimizer.register_step_pre_hook(change_group)
    try:
        opt.step()
    finally:
        handle.remove()
    reference.step()
    expected = independent_decay(before, task.atoms.p.detach(), 0.07, 0.6)
    torch.testing.assert_close(actual.atoms.p, expected, atol=3e-14, rtol=3e-14)


@pytest.mark.parametrize("fault", ["raw_decay", "negative_lambda", "excessive_step"])
def test_invalid_base_prehook_option_rejects_before_parameter_or_moments(fault):
    model = layer()
    opt = optimizer(model, 0.2)
    model.atoms.p.grad = torch.ones_like(model.atoms.p) * 0.1
    opt.step()
    before = model.atoms.p.detach().clone()
    state = {key: value.clone() for key, value in opt.state[model.atoms.p].items()}

    def invalid_group(base, args, kwargs):
        group = base.param_groups[0]
        if fault == "raw_decay":
            group["weight_decay"] = 0.01
        elif fault == "negative_lambda":
            group["amplitude_weight_decay"] = -0.1
        else:
            group["lr"] = 1.0
            group["amplitude_weight_decay"] = 2.0

    handle = opt.base_optimizer.register_step_pre_hook(invalid_group)
    try:
        with pytest.raises((ValueError, TypeError)):
            opt.step()
    finally:
        handle.remove()
    torch.testing.assert_close(model.atoms.p, before, atol=0, rtol=0)
    for key, value in state.items():
        torch.testing.assert_close(opt.state[model.atoms.p][key], value, atol=0, rtol=0)


@pytest.mark.parametrize("action", ["add", "remove"])
def test_base_prehook_gradient_membership_is_read_after_hooks(action):
    actual = layer()
    task = copy.deepcopy(actual)
    opt = optimizer(actual, 0.4)
    reference = optimizer(task)
    old = actual.atoms.p.detach().clone()
    gradient = torch.zeros_like(old)
    gradient[:, :2] = gradient.new_tensor([0.2, -0.1])
    actual.atoms.p.grad = gradient.clone() if action == "remove" else None
    task.atoms.p.grad = gradient.clone() if action == "add" else None

    def alter_gradient(base, args, kwargs):
        actual.atoms.p.grad = gradient.clone() if action == "add" else None

    handle = opt.base_optimizer.register_step_pre_hook(alter_gradient)
    try:
        opt.step()
    finally:
        handle.remove()
    reference.step()
    expected = (
        independent_decay(old, task.atoms.p.detach(), 0.02, 0.4)
        if action == "add"
        else old
    )
    torch.testing.assert_close(actual.atoms.p, expected, atol=3e-14, rtol=3e-14)
    if action == "remove":
        assert not opt.state.get(actual.atoms.p)


def test_base_posthook_options_apply_only_to_next_step():
    actual = layer()
    task = copy.deepcopy(actual)
    opt = optimizer(actual, 0.3, lr=0.02)
    reference = optimizer(task, lr=0.02)
    old = actual.atoms.p.detach().clone()
    gradient = torch.zeros_like(old)
    gradient[:, :2] = gradient.new_tensor([0.2, -0.1])
    actual.atoms.p.grad, task.atoms.p.grad = gradient.clone(), gradient.clone()

    def alter_next_group(base, args, kwargs):
        base.param_groups[0]["lr"] = 0.07
        base.param_groups[0]["amplitude_weight_decay"] = 0.8

    handle = opt.base_optimizer.register_step_post_hook(alter_next_group)
    try:
        opt.step()
    finally:
        handle.remove()
    reference.step()
    expected = independent_decay(old, task.atoms.p.detach(), 0.02, 0.3)
    torch.testing.assert_close(actual.atoms.p, expected, atol=3e-14, rtol=3e-14)
    assert opt.param_groups[0]["lr"] == 0.07
    assert opt.param_groups[0]["amplitude_weight_decay"] == 0.8


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


def repeated_layer(count):
    from test_radius_compatibility import model, polar_kernel

    result = model(polar_kernel(radius=True), atom_count=count).float()
    template = layer().atoms.p.detach().float()
    with torch.no_grad():
        result.atoms.p.copy_(template[torch.arange(count) % len(template)])
        # Valid centers immediately adjacent to the periodic seam.
        result.atoms.p[:, 2] = 16.0 - 2**-16
        result.atoms.p[:, 3] = 2**-16
    return result


def gpu_optimizer(model, decay=None):
    from torchcst._backends.catalog import get_registry
    from torchcst._backends.cuda.algorithms.polar_update.plans import (
        RADIUS_FLAT_TORUS_FUSED,
    )
    from torchcst._backends.dispatch import FixedSelector

    group = {"params": list(model.parameters()), "weight_decay": 0.0}
    if decay is not None:
        group["amplitude_weight_decay"] = decay
    return CSTOptimizer(
        torch.optim.AdamW([group], lr=0.02, foreach=False, fused=True),
        model=model,
        check_finite=False,
        update_selector=FixedSelector(RADIUS_FLAT_TORUS_FUSED, registry=get_registry()),
    )


@GPU
@pytest.mark.parametrize("count", [0, 7, 257])
def test_cuda_amplitude_steps_match_cpu_law_with_live_options(count):
    cpu = repeated_layer(count)
    gpu = copy.deepcopy(cpu).cuda()
    reference = optimizer(cpu, 0.3)
    actual = gpu_optimizer(gpu, 0.3)
    rng = torch.Generator().manual_seed(238)
    for index, (rate, decay) in enumerate(
        ((0.02, 0.3), (0.02, 0.0), (0.03, 0.5), (0.01, 0.8), (0.01, 0.2))
    ):
        for opt in (reference, actual):
            opt.param_groups[0]["lr"] = rate
            opt.param_groups[0]["amplitude_weight_decay"] = decay
        previous = cpu.atoms.p.detach().clone()
        previous_gpu = gpu.atoms.p.detach().clone()
        gradient = (
            torch.zeros_like(previous)
            if index == 0
            else torch.randn(count, 4, dtype=torch.float32, generator=rng) * 0.03
        )
        # Construct a task-only public update from the identical CPU state.
        task_model = copy.deepcopy(cpu)
        task_opt = optimizer(task_model)
        task_opt.load_state_dict(copy.deepcopy(reference.state_dict()))
        task_opt.param_groups[0].pop("amplitude_weight_decay")
        task_model.atoms.p.grad = gradient.clone()
        task_opt.step()
        cpu.atoms.p.grad = gradient.clone()
        gpu.atoms.p.grad = gradient.cuda()
        reference.step()
        actual.step()
        expected = independent_decay(previous, task_model.atoms.p.detach(), rate, decay)
        torch.testing.assert_close(cpu.atoms.p, expected, atol=4e-6, rtol=4e-6)
        torch.testing.assert_close(gpu.atoms.p.cpu(), expected, atol=4e-6, rtol=4e-6)
        torch.testing.assert_close(
            gpu.atoms.p[:, :2].square().sum(1).cpu(),
            task_model.atoms.p[:, :2].square().sum(1),
            atol=4e-6,
            rtol=4e-6,
        )
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                actual.state[gpu.atoms.p][key].cpu(),
                reference.state[cpu.atoms.p][key],
                atol=3e-6,
                rtol=3e-6,
            )
        if index == 0:
            torch.testing.assert_close(
                gpu.atoms.p[:, 2:], previous_gpu[:, 2:], atol=0, rtol=0
            )


@GPU
@pytest.mark.parametrize("count", [0, 7, 257])
def test_cuda_zero_amplitude_decay_is_bitwise_legacy(count):
    new = repeated_layer(count).cuda()
    legacy = copy.deepcopy(new)
    new_opt, old_opt = gpu_optimizer(new, 0.0), gpu_optimizer(legacy)
    rng = torch.Generator().manual_seed(261)
    for _ in range(5):
        gradient = (
            torch.randn(count, 4, dtype=torch.float32, generator=rng).cuda() * 0.03
        )
        new.atoms.p.grad, legacy.atoms.p.grad = gradient.clone(), gradient.clone()
        new_opt.step()
        old_opt.step()
        torch.testing.assert_close(new.atoms.p, legacy.atoms.p, atol=0, rtol=0)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                new_opt.state[new.atoms.p][key],
                old_opt.state[legacy.atoms.p][key],
                atol=0,
                rtol=0,
            )


def test_user_closure_runs_once_and_temporary_guard_is_removed():
    model = layer()
    opt = optimizer(model, 0.3)
    calls = []
    loss = torch.tensor(2.0)

    def closure():
        calls.append("closure")
        model.atoms.p.grad = torch.zeros_like(model.atoms.p)
        return loss

    assert opt.step(closure) is loss
    assert calls == ["closure"]
    assert not opt.base_optimizer._optimizer_step_pre_hooks


def test_base_prehook_cannot_replace_replayed_user_closure():
    model = layer()
    opt = optimizer(model, 0.3)
    before = model.atoms.p.detach().clone()
    calls = []

    def closure():
        calls.append("user")
        model.atoms.p.grad = torch.zeros_like(model.atoms.p)
        return torch.tensor(2.0)

    def replace_closure(base, args, kwargs):
        changed = dict(kwargs)
        changed["closure"] = lambda: torch.tensor(-1.0)
        return args, changed

    handle = opt.base_optimizer.register_step_pre_hook(replace_closure)
    try:
        with pytest.raises(ValueError, match="replace.*closure"):
            opt.step(closure)
    finally:
        handle.remove()
    assert calls == ["user"]
    assert not opt.base_optimizer._optimizer_step_pre_hooks
    assert not opt.state.get(model.atoms.p)
    torch.testing.assert_close(model.atoms.p, before, atol=0, rtol=0)


def test_general_algorithm_rejects_nonpolar_decay_context():
    from dataclasses import replace

    from test_cst_optimizer import site

    from torchcst._backends.schema import DefaultRecipe
    from torchcst._backends.torch.algorithms.atom_update.algorithm import (
        AtomUpdateAlgorithm,
    )
    from torchcst.operators import Operator
    from torchcst.operators.atom_update import AtomUpdateBinding

    model = site()
    binding = AtomUpdateBinding(
        Operator(charts=model.cst_charts(), kernel=model.kernel, atoms=model.atoms)
    )
    context = binding.build_context(
        AtomUpdateInputs(model.atoms.p.detach().clone(), 0.1)
    )
    result = AtomUpdateAlgorithm().supports(
        replace(context, amplitude_decay=True), DefaultRecipe()
    )
    assert not result.supported


def test_algorithm_decay_context_must_match_invocation():
    from dataclasses import replace

    from torchcst.operators import Operator
    from torchcst.operators.atom_update import AtomUpdateBinding, amplitude_decay_amount

    model = layer()
    binding = AtomUpdateBinding(
        Operator(charts=model.cst_charts(), kernel=model.kernel, atoms=model.atoms)
    )
    inputs = AtomUpdateInputs(model.atoms.p.detach().clone(), 0.1, 0.2)
    context = binding.build_context(inputs)
    with pytest.raises(ValueError, match="prepared context"):
        amplitude_decay_amount(replace(context, amplitude_decay=False), inputs)
