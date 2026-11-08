"""Envelope task derivatives against independent quadrature and finite differences."""

import copy
import io
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_radius_cell_mass import oracle_mass

from torchcst import (
    ChebyshevSpec,
    CSTAdamW,
    CSTLinear,
    FixedSelector,
    chart_presets,
    init,
    kernels,
    make_algorithm_registry,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_multirow import (
    LM_PHYSICAL_LEVELS,
    RadiusLMPhysicalLevelsAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.radius_envelope import (
    RADIUS_ENVELOPE_LM,
    RadiusEnvelopeLMAlgorithm,
)
from torchcst._backends.schema import DeviceInfo
from torchcst._backends.torch.parameterizations.radius_envelope import (
    from_state,
    physical_parameters,
)
from torchcst.kernels.radius_cell_mass import radius_cell_mass_axes
from torchcst.operators.execution import LinearInputs


def make(ni=16, no=32, atoms=2, dtype=torch.float64, **kwargs):
    return CSTLinear(
        chart_presets.regular_grid(no, spacing=1.0, geometry="flat_torus"),
        chart_presets.regular_grid(ni, spacing=1.0, geometry="flat_torus"),
        atoms=atoms,
        dtype=dtype,
        backend="factored",
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8)),
            coordinates=kernels.radius_envelope(radius_max=4.7, **kwargs),
        ),
    )


def values(p, **kw):
    settings = {"radius_min": 0.5, "radius_max": 4.7, "tau": 0.03, "amplitude_max": 1.0}
    settings.update(kw)
    return physical_parameters(p, **settings)


@pytest.mark.parametrize("tau", [0.01, 0.03, 0.1])
@pytest.mark.parametrize("q", [0.0, 0.25, 1.0])
def test_envelope_endpoints_and_complete_analytic_radius_jacobian(tau, q):
    p = torch.tensor(
        [[0.0, q, 1.0, 2.0], [0.7 * tau, q, 2.0, 3.0], [-2 * tau, q, 3.0, 4.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    r = values(p, tau=tau)[:, 1]
    g = torch.autograd.grad(r.sum(), p)[0]
    a = p[:, 0].detach()
    l = 0.5 + 4.2 * torch.exp(-((a / tau) ** 2))
    expected = torch.exp((1 - q) * torch.log(l) + q * math.log(4.7))
    la = -2 * a / (tau * tau) * 4.2 * torch.exp(-((a / tau) ** 2))
    torch.testing.assert_close(r, expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(
        g[:, 0], r.detach() * (1 - q) * la / l, atol=2e-12, rtol=2e-12
    )
    torch.testing.assert_close(
        g[:, 1], r.detach() * torch.log(4.7 / l), atol=2e-12, rtol=2e-12
    )
    assert g[0, 0] == 0 and g[0, 1] == 0
    assert torch.equal(g[:, 2:], torch.zeros_like(g[:, 2:]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finite_extreme_tau_and_decimal_encoded_radius_cap(dtype):
    p = torch.tensor(
        [[0.0, 0.0, 1.0, 2.0], [0.3, 1.0, 2.0, 3.0], [-0.3, 0.4, 3.0, 4.0]],
        dtype=dtype,
        requires_grad=True,
    )
    r = values(p, tau=1e-300, radius_max=2.96)[:, 1]
    g = torch.autograd.grad(r.sum(), p)[0]
    assert torch.isfinite(r).all() and torch.isfinite(g).all()
    assert (r.double() <= 2.96).all() and (r.double() >= 0.5).all()


def test_public_y_dx_and_every_atom_gradient_against_independent_oracle():
    p = torch.tensor(
        [[0.04, 0.25, 2.3, 3.7], [-0.07, 0.4, 5.13, 9.21]], dtype=torch.float64
    )
    model = make(atoms=p)
    rng = np.random.default_rng(904)
    x = rng.normal(size=(3, 16))
    dy = rng.normal(size=(3, 32))
    axes = radius_cell_mass_axes(
        model.execution_declaration().charts, model.execution_declaration().kernel
    )

    def weight(parameters):
        result = np.zeros((32, 16))
        for a, q, co, ci in parameters:
            lo = 0.5 + 4.2 * math.exp(-((a / 0.03) ** 2))
            r = lo ** (1 - q) * 4.7**q
            result += a * np.outer(
                oracle_mass(axes[0], r, co), oracle_mass(axes[1], r, ci)
            )
        return result

    expected = weight(p.numpy())
    xt = torch.tensor(x, requires_grad=True)
    y = model(xt)
    g = torch.autograd.grad(y, (xt, model.atoms.p), torch.tensor(dy), retain_graph=True)
    again = torch.autograd.grad(y, (xt, model.atoms.p), torch.tensor(dy))
    torch.testing.assert_close(y, torch.tensor(x @ expected.T), atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        g[0], torch.tensor(dy @ expected), atol=2e-12, rtol=2e-12
    )
    fd = np.zeros(p.shape)
    eps = 1e-6
    for a in range(2):
        for k in range(4):
            plus = p.numpy().copy()
            minus = plus.copy()
            plus[a, k] += eps
            minus[a, k] -= eps
            fd[a, k] = np.sum(((x @ weight(plus).T) - (x @ weight(minus).T)) * dy) / (
                2 * eps
            )
    torch.testing.assert_close(g[1], torch.tensor(fd), atol=2e-8, rtol=2e-5)
    for left, right in zip(g, again):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    materialized = copy.deepcopy(model)
    materialized.backend = "materialized"
    torch.testing.assert_close(
        materialized(xt.detach()), y.detach(), atol=1e-14, rtol=1e-12
    )


def optimizer(model):
    return CSTAdamW(
        model.parameters(),
        model=model,
        lr=0.002,
        weight_decay=0,
        field_lrs={"radius_ratio": 0.001},
    )


def test_q_learns_and_serialized_adam_resume_preserves_next_update():
    torch.manual_seed(92)
    m = make(atoms=7, q_init=0.25)
    o = optimizer(m)
    x = torch.randn(4, 16, dtype=torch.float64)
    target = torch.randn(4, 32, dtype=torch.float64)
    initial = m.atoms.p[:, 1].detach().clone()
    for _ in range(3):
        o.zero_grad()
        (m(x) - target).square().mean().backward()
        o.step()
    assert not torch.equal(m.atoms.p[:, 1], initial)
    assert ((m.atoms.p[:, 1] >= 0) & (m.atoms.p[:, 1] <= 1)).all()
    other = make(atoms=7, q_init=0.25)
    other.load_state_dict(copy.deepcopy(m.state_dict()))
    oo = optimizer(other)
    oo.load_state_dict(copy.deepcopy(o.state_dict()))
    for model, opt in [(m, o), (other, oo)]:
        opt.zero_grad()
        (model(x) - target).square().mean().backward()
        opt.step()
    torch.testing.assert_close(m.atoms.p, other.atoms.p, atol=0, rtol=0)
    for left, right in zip(o.state.values(), oo.state.values()):
        for key in left:
            if isinstance(left[key], torch.Tensor):
                torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
            else:
                assert left[key] == right[key]


def test_projection_and_fixed_scalar_precision():
    from torchcst._backends.torch.updates.radius_envelope import apply_parameter_update

    m = make(atoms=2, tau=0.0312345678901234)
    before = m.kernel.spec
    m.float().double()
    assert m.kernel.declaration().parameterization == before.parameterization
    p = m.atoms.p.detach()
    d = torch.zeros_like(p)
    d[:, 0] = torch.tensor([-20.0, 20.0])
    d[:, 1] = torch.tensor([-20.0, 20.0])
    result = apply_parameter_update(m.kernel, *m.cst_charts(), p, d, step_size=0.1)
    assert result[:, 0].tolist() == [-1.0, 1.0] and result[:, 1].tolist() == [0.0, 1.0]


@pytest.mark.parametrize(
    "bad",
    [
        {"tau": 0},
        {"tau": float("nan")},
        {"tau": True},
        {"q_init": -1},
        {"q_init": float("inf")},
        {"amplitude_std": -1},
    ],
)
def test_invalid_declarations_fail_before_execution(bad):
    with pytest.raises((TypeError, ValueError)):
        kernels.radius_envelope(**bad)


def test_moment_initializer_preserves_centers_and_requested_q():
    torch.manual_seed(318)
    m = make(atoms=64)
    centers = m.atoms.p[:, 2:].detach().clone()
    report = init.xavier_normal_(m, q=0.25)
    assert report.relative_error <= 0.005
    assert torch.equal(m.atoms.p[:, 2:], centers)
    assert torch.equal(m.atoms.p[:, 1], torch.full((64,), 0.25, dtype=torch.float64))


@pytest.mark.parametrize("ni,no", [(512, 2048), (2048, 512)])
def test_native_admission_is_separate_and_plan_roundtrips(ni, no):
    m = make(ni, no, atoms=1, dtype=torch.float32)
    c = m.build_context(LinearInputs(torch.zeros(1, ni)))
    c = replace(
        c,
        atom_count=8192,
        input_shape=(256, ni),
        input_strides=(ni, 1),
        device=DeviceInfo("cuda", 0, "metadata-only"),
    )
    assert RadiusEnvelopeLMAlgorithm().supports(c, RADIUS_ENVELOPE_LM.recipe).supported
    assert (
        not RadiusLMPhysicalLevelsAlgorithm()
        .supports(c, LM_PHYSICAL_LEVELS.recipe)
        .supported
    )
    for fields in [
        {"atom_count": 8191},
        {"dtype": torch.float64},
        {"execution_mode": "cuda_graph"},
        {"input_shape": (257, ni)},
    ]:
        assert (
            not RadiusEnvelopeLMAlgorithm()
            .supports(replace(c, **fields), RADIUS_ENVELOPE_LM.recipe)
            .supported
        )
    reg = make_algorithm_registry()
    reg.register(RadiusEnvelopeLMAlgorithm())
    assert reg.loads_plan(reg.dumps_plan(RADIUS_ENVELOPE_LM)) == RADIUS_ENVELOPE_LM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("ni,no", [(512, 2048), (2048, 512)])
@pytest.mark.parametrize("rows", [7, 256])
def test_cuda_complete_envelope_vjp_and_retained_backward(ni, no, rows):
    torch.manual_seed(434)
    base = make(ni, no, atoms=8192, dtype=torch.float32, q_init=0.25)
    with torch.no_grad():
        base.atoms.p[:, 0].uniform_(-0.3, 0.3)
        base.atoms.p[::11, 0] = 0
        base.atoms.p[::3, 1] = 0
        base.atoms.p[1::3, 1] = 1
    x = torch.randn(rows, ni * 2)
    dy = torch.randn(rows, no)
    # Keep FP32 physical-row encoding. A double latent model can turn .5
    # into .5+epsilon and choose a different ceil/window branch. Accumulate
    # the physical reference in FP64, preserving the full map's VJP.
    physical = from_state(base.kernel, base.atoms.p)
    operator = base.execution_declaration()
    reference = CSTLinear(
        *operator.charts,
        atoms=1,
        dtype=torch.float64,
        backend="factored",
        kernel=kernels.product(
            *operator.kernel.profiles,
            coordinates=kernels.physical(radius_init=(0.5, 4.7), radius_floor=0.5),
        ),
    )
    vi, uo = reference.operator.factors(physical.double())
    cx = x.double()[:, ::2].requires_grad_()
    cy = (cx @ vi) @ uo.T
    cg = torch.autograd.grad(cy, (cx, base.atoms.p), dy.double())
    gpu = copy.deepcopy(base).cuda()
    torch.testing.assert_close(
        from_state(gpu.kernel, gpu.atoms.p).cpu(), physical.detach(), atol=0, rtol=0
    )
    reg = make_algorithm_registry()
    reg.register(RadiusEnvelopeLMAlgorithm())
    gpu.selector = FixedSelector(RADIUS_ENVELOPE_LM, registry=reg)
    gx = x.cuda()[:, ::2].requires_grad_()
    gy = gpu(gx)
    gg = torch.autograd.grad(gy, (gx, gpu.atoms.p), dy.cuda(), retain_graph=True)
    again = torch.autograd.grad(gy, (gx, gpu.atoms.p), dy.cuda())
    for actual, expected in zip((gy, *gg), (cy, *cg)):
        torch.testing.assert_close(
            actual.cpu().double(), expected.double(), atol=1e-4, rtol=4e-3
        )
    for a, b in zip(gg, again):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.isfinite(gg[1]).all() and torch.count_nonzero(gg[1][:, 1]) > 0
    opt = optimizer(gpu)
    opt.zero_grad()
    gpu(gx.detach()).square().mean().backward()
    opt.step()
    assert ((gpu.atoms.p[:, 1] >= 0) & (gpu.atoms.p[:, 1] <= 1)).all()
    payload = io.BytesIO()
    torch.save({"model": gpu.state_dict(), "optimizer": opt.state_dict()}, payload)
    payload.seek(0)
    saved = torch.load(payload, weights_only=False)
    restored = copy.deepcopy(gpu)
    restored.load_state_dict(saved["model"])
    restored_opt = optimizer(restored)
    restored_opt.load_state_dict(saved["optimizer"])
    for model, current_opt in ((gpu, opt), (restored, restored_opt)):
        current_opt.zero_grad()
        model(gx.detach()).square().mean().backward()
        current_opt.step()
    torch.testing.assert_close(gpu.atoms.p, restored.atoms.p, atol=0, rtol=0)
    for key, value in opt.state[gpu.atoms.p].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(
                value, restored_opt.state[restored.atoms.p][key], atol=0, rtol=0
            )


def test_encoded_radius_branch_is_not_invariant_under_double_conversion():
    p = torch.tensor([[0.15, 0.0, 2.3, 3.7]], dtype=torch.float32)
    assert values(p)[0, 1].item() == 0.5
    assert values(p.double())[0, 1].item() > 0.5
