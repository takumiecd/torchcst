"""Center-driven activity preserves Polar amplitude and actual task derivatives."""

import copy
import math
from dataclasses import replace

import pytest
import torch
from test_radius_compatibility import model

from torchcst import CSTAdamW, FixedSelector, RadiusBounds, kernels
from torchcst._backends.catalog import make_registry
from torchcst._backends.cuda.algorithms.polar_update.algorithm import (
    RadiusFlatTorusPolarUpdateAlgorithm,
)
from torchcst._backends.schema import DefaultRecipe, DeviceInfo, ExecutionPlan
from torchcst._backends.torch.algorithms.polar_update.executor import (
    apply_parameter_update,
)
from torchcst._backends.torch.algorithms.polar_update.radius_eager import (
    RadiusEagerPolarUpdateAlgorithm,
)
from torchcst.operators.atom_update import AtomUpdateBinding, AtomUpdateInputs

PLAN = ExecutionPlan("torch_radius_eager_polar_update", "v1", DefaultRecipe())


def fixture(mode="center_chord", *, gain=0.8, decay=0.12):
    return model(
        kernels.product(
            kernels.cell_mass(),
            coordinates=kernels.polar(
                amplitude_max=3.0,
                w_c=0.7,
                kappa=2.5,
                lower_kappa=1.7,
                upper_decay_power=0.8,
                bounds=RadiusBounds(minimum=0.2, birth=0.7, maximum=2, upper_floor=0.5),
                activity_mode=mode,
                activity_gain=gain,
                radial_regularization=decay,
                dormant_expansion_rate=0,
            ),
        )
    )


def selector():
    registry = make_registry()
    registry.register(RadiusEagerPolarUpdateAlgorithm())
    return FixedSelector(PLAN, registry=registry)


def independent_radius(row, state):
    scalar = lambda name: float(state.scalar(name))
    s, t = row[:2]
    norm2 = s * s + t * t
    a = scalar("amplitude_max") * s / math.sqrt(norm2)
    p = max(0, min(1, (norm2 - 1) / 3))
    x = (a / scalar("w_c")) ** 2
    minimum, birth, maximum, floor = [
        scalar(n + "_input")
        for n in ("radius_min", "radius_birth", "radius_max", "radius_upper_floor")
    ]
    kappa = scalar("kappa")
    lo = minimum + (birth - minimum) / (1 + scalar("lower_kappa") * x)
    hi = max(
        lo,
        floor,
        minimum
        + (maximum - minimum) * kappa / (kappa + x ** scalar("upper_decay_power")),
    )
    return math.exp((1 - p) * math.log(lo) + p * math.log(hi))


@pytest.mark.parametrize("gain", [1.0, 64.0])
@pytest.mark.parametrize("decay", [0.0, 0.12])
def test_scalar_oracle_of_center_activity_and_amplitude(gain, decay):
    layer = fixture(gain=gain, decay=decay)
    old = layer.atoms.p.detach().clone()
    displacement = (
        torch.randn(old.shape, generator=torch.Generator().manual_seed(19)) * 0.2
    )
    result = apply_parameter_update(
        layer.kernel, *layer.cst_charts(), old, displacement, step_size=0.03
    )
    expected = []
    for row, delta in zip(old.tolist(), displacement.tolist()):
        s, t = row[:2]
        norm2 = s * s + t * t
        dot = (s * delta[0] + t * delta[1]) / norm2
        u, v = delta[0] - dot * s, delta[1] - dot * t
        direction_norm = math.hypot(s + u, t + v)
        centers = [(row[j] + delta[j]) % 16 for j in (2, 3)]
        movement = [(new - row[j] + 8) % 16 - 8 for j, new in zip((2, 3), centers)]
        r = independent_radius(row, layer.kernel)
        energy = sum((d / r) ** 2 for d in movement) / 2
        task = max(1, min(4, norm2 + gain * energy))
        accepted = 1 / (
            1
            - (task - 1)
            / task
            * math.exp(-4 * float(layer.kernel.scalar("radial_regularization")) * 0.03)
        )
        expected.append(
            [
                (s + u) / direction_norm * math.sqrt(accepted),
                (t + v) / direction_norm * math.sqrt(accepted),
                *centers,
            ]
        )
    torch.testing.assert_close(
        result, torch.tensor(expected, dtype=old.dtype), atol=3e-14, rtol=3e-14
    )


def test_amplitude_motion_alone_does_not_expand_center_activity():
    layer = fixture(decay=0)
    old = layer.atoms.p.detach().clone()
    delta = torch.zeros_like(old)
    delta[:, 0] = 0.2
    delta[:, 1] = -0.1
    new = apply_parameter_update(
        layer.kernel, *layer.cst_charts(), old, delta, step_size=0.01
    )
    torch.testing.assert_close(
        new[:, :2].square().sum(-1), old[:, :2].square().sum(-1), atol=2e-15, rtol=2e-15
    )
    assert not torch.allclose(new[:, :2], old[:, :2])
    torch.testing.assert_close(new[:, 2:], old[:, 2:], atol=0, rtol=0)


def test_shortest_accepted_center_motion_not_full_period_proposal():
    layer = fixture(decay=0)
    old = layer.atoms.p.detach().clone()
    delta = torch.zeros_like(old)
    delta[:, 2] = 16
    delta[:, 3] = -32
    new = apply_parameter_update(
        layer.kernel, *layer.cst_charts(), old, delta, step_size=0.01
    )
    torch.testing.assert_close(new, old, atol=3e-15, rtol=3e-15)
    old[:, 2] = 15.99
    delta.zero_()
    delta[:, 2] = 0.02
    wrapped = apply_parameter_update(
        layer.kernel, *layer.cst_charts(), old, delta, step_size=0.01
    )
    old[:, 2] = 4.99
    regular = apply_parameter_update(
        layer.kernel, *layer.cst_charts(), old, delta, step_size=0.01
    )
    torch.testing.assert_close(wrapped[:, :2], regular[:, :2], atol=2e-14, rtol=2e-14)


def test_same_operator_all_vjps_and_radial_task_gradient_zero():
    old, new = fixture("finite_chord"), fixture()
    x = torch.randn(
        3, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(9)
    )
    values = []
    for layer in (old, new):
        live = x.clone().requires_grad_()
        y = layer(live)
        gx, gp = torch.autograd.grad(y.square().sum(), (live, layer.atoms.p))
        values.append((y, gx, gp))
        assert (
            float((layer.atoms.p[:, :2].detach() * gp[:, :2]).sum(-1).abs().max())
            < 1e-12
        )
    for a, b in zip(*values):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_registered_only_explicitly_supports_eager_and_native_rejects():
    layer = fixture()
    context = AtomUpdateBinding(layer.operator).build_context(
        AtomUpdateInputs(layer.atoms.p.detach().clone(), 0.01)
    )
    algorithm = RadiusEagerPolarUpdateAlgorithm()
    assert algorithm.supports(context, PLAN.recipe).supported
    assert not algorithm.supports(
        replace(context, execution_mode="cuda_graph"), PLAN.recipe
    ).supported
    cuda = replace(context, device=DeviceInfo("cuda", 0))
    assert algorithm.supports(cuda, PLAN.recipe).supported
    assert (
        not RadiusFlatTorusPolarUpdateAlgorithm()
        .supports(cuda, DefaultRecipe())
        .supported
    )
    registry = make_registry()
    with pytest.raises((KeyError, ValueError)):
        registry.validate_plan(PLAN)
    registry.register(algorithm)
    assert registry.loads_plan(registry.dumps_plan(PLAN)) == PLAN


@pytest.mark.parametrize("mode", ["finite_chord", "time_energy", "center_chord"])
def test_exact_cpu_optimizer_resume_and_contract_rejection(mode):
    layer = fixture(mode)
    opt = CSTAdamW(
        layer.parameters(),
        model=layer,
        lr=0.01,
        weight_decay=0,
        update_selector=selector(),
    )
    for _ in range(2):
        layer.atoms.p.grad = torch.full_like(layer.atoms.p, 0.1)
        opt.step()
    other = fixture(mode)
    other.load_state_dict(copy.deepcopy(layer.state_dict()))
    resumed = CSTAdamW(
        other.parameters(),
        model=other,
        lr=0.01,
        weight_decay=0,
        update_selector=selector(),
    )
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    for m, o in ((layer, opt), (other, resumed)):
        m.atoms.p.grad = torch.full_like(m.atoms.p, -0.07)
        o.step()
    torch.testing.assert_close(layer.atoms.p, other.atoms.p, atol=0, rtol=0)
    for key in opt.state[layer.atoms.p]:
        torch.testing.assert_close(
            opt.state[layer.atoms.p][key],
            resumed.state[other.atoms.p][key],
            atol=0,
            rtol=0,
        )
    if mode == "center_chord":
        with pytest.raises((ValueError, RuntimeError)):
            fixture("finite_chord").load_state_dict(layer.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["finite_chord", "time_energy", "center_chord"])
def test_cuda_optimizer_parameters_and_moments_match_cpu(mode):
    cpu = fixture(mode).float()
    gpu = copy.deepcopy(cpu).cuda()
    opts = [
        CSTAdamW(
            m.parameters(), model=m, lr=0.01, weight_decay=0, update_selector=selector()
        )
        for m in (cpu, gpu)
    ]
    generator = torch.Generator().manual_seed(41)
    for _ in range(5):
        grad = torch.randn(cpu.atoms.p.shape, generator=generator) * 0.01
        for m, o in zip((cpu, gpu), opts):
            m.atoms.p.grad = grad.to(m.atoms.p)
            o.step()
        torch.testing.assert_close(cpu.atoms.p, gpu.atoms.p.cpu(), atol=4e-6, rtol=4e-6)
        for key in opts[0].state[cpu.atoms.p]:
            torch.testing.assert_close(
                opts[0].state[cpu.atoms.p][key],
                opts[1].state[gpu.atoms.p][key].cpu(),
                atol=3e-7,
                rtol=3e-6,
            )
