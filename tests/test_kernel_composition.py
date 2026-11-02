"""Composition contracts checked against independent dense scalar formulas."""

import copy
import io
from dataclasses import asdict

import pytest
import torch

from torchcst import (
    BandwidthBounds,
    CoordinateSpec,
    CSTAdamW,
    CSTLinear,
    CSTOptimizer,
    GaussianSpec,
    RadiusBounds,
    TriweightSpec,
    geometry_presets,
    kernels,
)
from torchcst import (
    chart_presets as charts,
)
from torchcst._backends.torch.profiles import execution as profiles
from torchcst.kernels.state import ProfileState


def rule(kind):
    return getattr(kernels, kind)(
        amplitude_max=2.0,
        w_c=0.5,
        input_bounds=BandwidthBounds(
            minimum=0.5, birth=1.0, maximum=2.0, upper_floor=0.5
        ),
        output_bounds=BandwidthBounds(
            minimum=0.25, birth=0.5, maximum=1.0, upper_floor=0.25
        ),
        alpha_init=0.25,
    )


def sigma(amplitude, alpha, minimum, birth, maximum):
    # Independent width law. Task differentiation holds this derived state fixed.
    x = (amplitude / 0.5).square()
    lo = minimum + (birth - minimum) / (1 + 3 * x)
    hi = torch.maximum(minimum + (maximum - minimum) * 3 / (3 + x), lo)
    return ((1 - alpha) * lo.log() + alpha * hi.log()).exp().detach()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
@pytest.mark.parametrize("kind", ["fixed", "polar", "direct"])
@pytest.mark.parametrize("shape", [GaussianSpec, TriweightSpec])
@pytest.mark.parametrize("normalized", [False, True])
def test_radial_pair_values_input_and_all_atom_gradients(
    kind, shape, normalized, device
):
    factor = (kernels.l2 if normalized else kernels.point)(
        shape(), **({"sigma": 0.5} if kind == "fixed" else {})
    )
    spec = kernels.radial(factor, coordinates=None if kind == "fixed" else rule(kind))
    out = charts.regular_grid((2, 2), spacing=0.25)
    inp = charts.regular_grid(3, spacing=0.5)
    centers = torch.tensor([[0.1, 0.2, 0.3], [0.3, -0.1, 0.8]], dtype=torch.float64)
    prefix = torch.tensor([[0.3, 1.2], [-0.5, 1.1]], dtype=torch.float64)
    p = centers if kind == "fixed" else torch.cat((prefix, centers), 1)
    model = CSTLinear(charts=(out, inp), atoms=p, kernel=spec, dtype=torch.float64)
    model = model.to(device)
    x = torch.randn(2, 3, device=device, dtype=torch.float64, requires_grad=True)
    ox = x.detach().clone().requires_grad_()
    op = model.atoms.p.detach().clone().requires_grad_()
    if kind == "fixed":
        amp = op.new_ones(2)
        ci = op
        si = so = 0.5
    else:
        q = op[:, :2].square().sum(-1) if kind == "polar" else op[:, 1].clamp(1, 4)
        amp = 2 * op[:, 0] / q.sqrt() if kind == "polar" else op[:, 0].clamp(-2, 2)
        alpha = ((q - 1) / 3).clamp(0, 1)
        si = sigma(amp, alpha, 0.5, 1.0, 2.0)
        so = sigma(amp, alpha, 0.25, 0.5, 1.0)
        ci = op[:, 2:]
    output_sites = op.new_tensor([[0, 0], [0, 0.25], [0.25, 0], [0.25, 0.25]])
    input_sites = op.new_tensor([0, 0.5, 1.0])
    qo = (output_sites[:, None] - ci[None, :, :2]).square().sum(-1) / so**2
    qi = (input_sites[:, None] - ci[None, :, 2]).square() / si**2
    q = qo[:, None] + qi[None]
    raw = (-0.5 * q).exp() if shape is GaussianSpec else (1 - q).clamp_min(0).pow(3)
    if normalized:
        raw = raw / raw.square().sum((0, 1)).sqrt().clamp_min(1e-6)
    weight = (raw * amp).sum(-1)
    expected = ox @ weight.T
    actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    grad = torch.autograd.grad(actual.square().sum(), (x, model.atoms.p))
    oracle = torch.autograd.grad(expected.square().sum(), (ox, op))
    for a, b in zip(grad, oracle):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)
    assert model.declaration().kernel.composition == "radial"
    assert model.charts[0].ncs == 4


@pytest.mark.parametrize("kind", ["polar", "direct"])
def test_gaussian_radial_equals_chart_product_including_optimizer_state(kind):
    torch.manual_seed(3)
    output, inp = (
        charts.regular_grid(3, spacing=0.25),
        charts.regular_grid(5, spacing=0.5),
    )
    coordinate = rule(kind)
    assert isinstance(coordinate, CoordinateSpec)
    assert "profiles" not in asdict(coordinate)
    factor = kernels.l2(GaussianSpec())
    a = CSTLinear(
        charts=(output, inp),
        atoms=3,
        kernel=kernels.product(factor, coordinates=coordinate),
        dtype=torch.float64,
    )
    b = CSTLinear(
        charts=(output, inp),
        atoms=a.atoms.p.detach(),
        kernel=kernels.radial(factor, coordinates=coordinate),
        dtype=torch.float64,
    )
    opts = [
        CSTOptimizer(torch.optim.Adam(m.parameters(), lr=0.01), model=m) for m in (a, b)
    ]
    x = torch.randn(2, 5, dtype=torch.float64)
    for _ in range(3):
        for m, opt in zip((a, b), opts):
            opt.zero_grad()
            m(x).square().sum().backward()
            opt.step()
        torch.testing.assert_close(a.atoms.p, b.atoms.p, rtol=1e-10, atol=1e-11)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                opts[0].state[a.atoms.p][key],
                opts[1].state[b.atoms.p][key],
                rtol=1e-10,
                atol=1e-11,
            )


@pytest.mark.parametrize("kind", ["polar", "direct", "fixed"])
def test_radial_geometry_update_and_weights_only_checkpoint(kind):
    out = charts.points(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        geometry=geometry_presets.sphere(2),
    )
    inp = charts.regular_grid(5, spacing=0.5, geometry="flat_torus")
    factor = kernels.l2(GaussianSpec(), **({"sigma": 0.5} if kind == "fixed" else {}))
    model = CSTLinear(
        charts=(out, inp),
        atoms=3,
        kernel=kernels.radial(
            factor, coordinates=None if kind == "fixed" else rule(kind)
        ),
        dtype=torch.float64,
    )
    opt = CSTOptimizer(torch.optim.Adam(model.parameters(), lr=0.01), model=model)
    x = torch.randn(2, 5, dtype=torch.float64)
    model(x).square().sum().backward()
    opt.step()
    start = 0 if kind == "fixed" else 2
    torch.testing.assert_close(
        model.atoms.p[:, start : start + 3].norm(dim=-1),
        torch.ones(3, dtype=torch.float64),
    )
    assert ((model.atoms.p[:, -1] >= 0) & (model.atoms.p[:, -1] < 2.5)).all()
    stream = io.BytesIO()
    torch.save((model.state_dict(), opt.state_dict()), stream)
    stream.seek(0)
    state, optimizer_state = torch.load(stream, weights_only=True)
    restored = copy.deepcopy(model)
    restored.load_state_dict(state)
    other = CSTOptimizer(
        torch.optim.Adam(restored.parameters(), lr=0.01), model=restored
    )
    other.load_state_dict(optimizer_state)
    for m, o in [(model, opt), (restored, other)]:
        o.zero_grad()
        m(x).square().sum().backward()
        o.step()
    torch.testing.assert_close(model.atoms.p, restored.atoms.p)


def test_composition_rejects_undefined_or_ambiguous_widths():
    bounds = RadiusBounds(minimum=0.5, birth=1.0, maximum=2.0, upper_floor=0.5)
    cell = kernels.cell_mass()
    coord = kernels.polar(amplitude_max=1.0, w_c=0.3, bounds=bounds)
    assert kernels.product(cell, coordinates=coord).revision == 7
    with pytest.raises(ValueError, match="radius cell masses"):
        kernels.radial(cell, coordinates=coord)
    with pytest.raises(ValueError, match="once"):
        kernels.product(
            kernels.l2(GaussianSpec(), sigma=1.0), coordinates=rule("polar")
        )
    with pytest.raises(TypeError, match="CoordinateSpec"):
        kernels.product(kernels.l2(GaussianSpec()), coordinates=object())


def test_custom_compact_floor_forward_and_analytic_tangent():
    profile = ProfileState(kernels.l2(TriweightSpec(), sigma=1.0, floor=0.2)).double()
    chart = charts.points(((0.0,), (0.2,), (2.0,)))
    from torchcst import compile_chart

    chart = compile_chart(chart, dtype=torch.float64)
    p = torch.tensor([[0.95], [5.0]], dtype=torch.float64, requires_grad=True)
    values, tangent = profiles.tangent(profile, chart, p)
    raw = (
        (1 - (p.T - torch.tensor([[0.0], [0.2], [2.0]], dtype=torch.float64)).square())
        .clamp_min(0)
        .pow(3)
    )
    expected = raw / raw.norm(dim=0).clamp_min(0.2)
    torch.testing.assert_close(values, expected)
    gradient = torch.autograd.grad(expected.sum(), p)[0]
    torch.testing.assert_close(tangent.sum(0), gradient)
    assert torch.isfinite(tangent).all()


def test_empty_radial_atom_table_is_a_differentiable_zero_operator():
    chart = charts.regular_grid(3)
    m = CSTLinear(
        charts=(chart, chart),
        atoms=torch.empty(0, 2),
        kernel=kernels.radial(kernels.l2(GaussianSpec(), sigma=1.0)),
    )
    x = torch.randn(2, 3, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert torch.equal(y, torch.zeros_like(y))
    assert m.atoms.p.grad.shape == (0, 2)


@pytest.mark.parametrize("native", [False, True])
def test_pre_unification_center_order_checkpoints_are_rejected(native):
    # Old center columns had identical shapes, so shape checks are insufficient.
    chart = charts.regular_grid(3)
    model = CSTLinear(
        charts=(chart, chart),
        atoms=2,
        kernel=kernels.product(kernels.l2(GaussianSpec()), coordinates=rule("polar")),
    )
    opt = (
        CSTAdamW(model.parameters(), model=model)
        if native
        else CSTOptimizer(torch.optim.Adam(model.parameters()), model=model)
    )
    saved = copy.deepcopy(model.state_dict())
    del saved["kernel._extra_state"]["center_order"]
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        model.load_state_dict(saved)
    saved_opt = copy.deepcopy(opt.state_dict())
    del saved_opt["cst"]["sites"][0][1]["center_order"]
    with pytest.raises(ValueError, match="checkpoint contract"):
        opt.load_state_dict(saved_opt)
