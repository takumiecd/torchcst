"""Independent integral/KKT oracle for the I-driven mathematical reference."""

import math
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest
import torch
from numpy.polynomial.legendre import leggauss

from torchcst import FourierPeakSpec, IntegerFourierPeriodSpec, NormalizationSpec
from torchcst._backends.torch.kernels import effective_fourier as impl
from torchcst._backends.torch.profiles import fourier_peak as shape

DT = torch.float64


def oracle_coefficients(q, modes):
    """Numerical transform + full saddle-point solve, no analytic transform."""
    nodes, weights = leggauss(192)
    nodes, weights = torch.tensor(nodes), torch.tensor(weights)
    k = torch.arange(modes + 1, dtype=DT)
    integral = torch.cos(math.pi * q * k[:, None] * nodes) @ (
        weights * (1 - nodes.square()) ** 4
    )
    c = q / 2 * integral * torch.where(k == 0, 1.0, 2.0)
    rows = 2 if modes == 4 else 3
    a = torch.stack(
        [(-1.0) ** k * (k / modes) ** (2 * j) for j in range(rows)]
        + [torch.ones_like(k)]
    )
    metric = torch.diag(torch.where(k == 0, 2.0, 1.0).double())
    system = torch.cat(
        (
            torch.cat((metric, a.T), 1),
            torch.cat((a, torch.zeros(len(a), len(a), dtype=DT)), 1),
        ),
        0,
    )
    target = torch.zeros(len(a), dtype=DT)
    target[-1] = 1
    return torch.linalg.solve(system, torch.cat((metric @ c, target)))[: modes + 1]


def oracle(sites, mu, width, profile, period, support):
    # Routing is computed independently with Python scalar arithmetic. Width
    # remains live in the coefficients and in the transition polynomial.
    step, tile = period.minimum_half_period, period.tile_width
    k = math.ceil((float(width.detach()) + tile / 2) / step)
    low = k * step
    if period.transition_fraction:
        z = (
            (width - (low - tile / 2 - period.transition_fraction * step))
            / (period.transition_fraction * step)
        ).clamp(0, 1)
        w = 6 * z**5 - 15 * z**4 + 10 * z**3
    else:
        w = width * 0
    r = sites - mu
    result = torch.zeros_like(r)
    for length, weight in ((low, 1 - w), (low + step, w)):
        b = oracle_coefficients(width / length, profile.modes)
        index = torch.arange(profile.modes + 1, dtype=DT)
        value = torch.cos(r[:, None] * index * math.pi / length) @ b
        if support == "centered":
            mask = r.abs() < length
        else:
            left = math.floor(float((mu - width).detach()) / tile) * tile
            right = math.ceil(float((mu + width).detach()) / tile) * tile
            mask = (sites >= left) & (sites < right)
        result = result + value * weight * mask
    return result


def test_declarations_are_independent_and_validated():
    from torchcst.kernels import FourierPeakSpec as a
    from torchcst.kernels.profiles import FourierPeakSpec as b

    assert a is b is FourierPeakSpec
    profile = FourierPeakSpec(modes=4)
    assert {f.name for f in fields(profile)} == {"id", "revision", "modes"}
    with pytest.raises(FrozenInstanceError):
        profile.modes = 8
    for modes in (True, 6, 4.0):
        with pytest.raises(ValueError):
            FourierPeakSpec(modes=modes)
    for name in ("tile_width", "minimum_half_period"):
        for bad in (0, -1, float("inf"), float("nan"), True):
            options = {"tile_width": 0.8, "minimum_half_period": 0.2, name: bad}
            with pytest.raises(ValueError):
                IntegerFourierPeriodSpec(**options)
    for bad in (-0.1, 1, float("nan"), True):
        with pytest.raises(ValueError):
            IntegerFourierPeriodSpec(
                tile_width=0.8, minimum_half_period=0.2, transition_fraction=bad
            )
    for q in (0.0, 1.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="q=I/L"):
            shape.coefficients(profile, torch.tensor(q))


@pytest.mark.parametrize("modes", [4, 8])
def test_transform_kkt_derivatives_periodicity_and_endpoints(modes):
    profile = FourierPeakSpec(modes=modes)
    k = torch.arange(modes + 1, dtype=DT)
    for number in (
        0.0001,
        0.05,
        0.2,
        4 / (math.pi * modes) - 1e-7,
        4 / (math.pi * modes) + 1e-7,
        0.5,
        0.9,
        0.9999,
    ):
        q = torch.tensor(number, dtype=DT, requires_grad=True)
        actual, dq = shape.coefficients(profile, q)
        expected = oracle_coefficients(q, modes)
        torch.testing.assert_close(actual, expected, atol=3e-13, rtol=2e-10)
        cot = torch.sin(k + 0.3)
        derivative = torch.autograd.grad(expected @ cot, q, retain_graph=True)[0]
        torch.testing.assert_close(dq @ cot, derivative, atol=2e-11, rtol=2e-9)
        derivative2 = torch.autograd.grad(actual @ cot, q)[0]
        torch.testing.assert_close(dq @ cot, derivative2, atol=2e-11, rtol=2e-9)
        torch.testing.assert_close(
            actual.sum(), torch.tensor(1.0, dtype=DT), atol=2e-14, rtol=0
        )
        for j in range(2 if modes == 4 else 3):
            assert (
                abs(
                    float(
                        (actual * (-1.0) ** k * (k / modes) ** (2 * j)).sum().detach()
                    )
                )
                < 3e-14
            )
        phase = torch.linspace(-math.pi, math.pi, 33, dtype=DT)
        torch.testing.assert_close(
            shape.values(profile, phase, q),
            shape.values(profile, phase + 2 * math.pi, q),
            atol=3e-14,
            rtol=0,
        )


@pytest.mark.parametrize("transition", [0.0, 0.25])
def test_period_safety_for_independent_scales_and_live_widths(transition):
    for tile in (0.13, 0.8, 3.2):
        for step in (0.07, 0.25, 0.7):
            period = IntegerFourierPeriodSpec(
                tile_width=tile,
                minimum_half_period=step,
                transition_fraction=transition,
            )
            width = torch.linspace(0.001, 5.0, 151, dtype=DT)
            lengths, weights, dw, k = impl.period_components(period, width)
            assert bool(
                (
                    (2 * lengths - 2 * width[:, None] - tile >= -1e-14) | (weights == 0)
                ).all()
            )
            assert bool(((k - 1) * step < width + tile / 2).all())
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(width))
            assert bool((weights >= 0).all())
            assert bool((dw.sum(-1) == 0).all())
            centers = torch.linspace(-tile, tile, len(width), dtype=DT)
            left, right = impl.tile_bounds(period, centers, width)
            assert bool((left >= centers - 2 * lengths[:, 0] + width - 1e-14).all())
            assert bool((right <= centers + 2 * lengths[:, 0] - width + 1e-14).all())


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("support", ["centered", "whole_tiles"])
@pytest.mark.parametrize("normalize", [False, True])
def test_values_normalization_and_both_gradients_against_integral(
    modes, support, normalize
):
    profile = FourierPeakSpec(modes=modes)
    period = IntegerFourierPeriodSpec(
        tile_width=0.8, minimum_half_period=0.25, transition_fraction=0.25
    )
    norm = (
        NormalizationSpec(kind="discrete_l2", domain="chart_sites", floor=1e-6)
        if normalize
        else NormalizationSpec()
    )
    x = torch.arange(-37, 44, dtype=DT) * 0.1
    for width0, mu0 in (
        (0.08, 0.013),
        (0.332, -0.017),
        (0.585, 0.13),
        (1.34, 4.07),
        (0.24, 20.0),
    ):
        mu = torch.tensor(mu0, dtype=DT, requires_grad=True)
        width = torch.tensor(width0, dtype=DT, requires_grad=True)
        value, dm, di, raw_norm = impl.values_and_derivatives(
            profile, period, x, mu, width, support=support, normalization=norm
        )
        expected = oracle(x, mu, width, profile, period, support)
        torch.testing.assert_close(raw_norm, expected.norm(), atol=3e-12, rtol=2e-10)
        if normalize:
            expected = expected / expected.norm().clamp_min(1e-6)
            if raw_norm > 1e-6:
                torch.testing.assert_close(value.norm(), torch.tensor(1.0, dtype=DT))
        cot = torch.sin(x * 0.72 + 0.13)
        actual_grad = torch.autograd.grad(value @ cot, (mu, width), retain_graph=True)
        expected_grad = torch.autograd.grad(expected @ cot, (mu, width))
        torch.testing.assert_close(value, expected, atol=4e-12, rtol=2e-9)
        for a, b, c in zip(actual_grad, expected_grad, (dm @ cot, di @ cot)):
            torch.testing.assert_close(a, b, atol=5e-10, rtol=2e-8)
            torch.testing.assert_close(a, c, atol=5e-10, rtol=2e-8)


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("normalize", [False, True])
def test_safe_period_transition_is_c2_but_hard_selection_has_shape_jump(
    modes, normalize
):
    spec = FourierPeakSpec(modes=modes)
    period = IntegerFourierPeriodSpec(
        tile_width=0.8, minimum_half_period=0.25, transition_fraction=0.25
    )
    # Restrict x away from whole-tile changes to isolate period continuity.
    x = torch.tensor([0.031, 0.17, 0.29, 0.47], dtype=DT)
    norm = (
        NormalizationSpec(kind="discrete_l2", domain="chart_sites", floor=1e-6)
        if normalize
        else None
    )
    for switch in (0.35, 0.6, 0.85):
        for boundary in (switch - 0.0625, switch):
            samples = []
            for delta in (-1e-9, 0, 1e-9):
                width = torch.tensor(boundary + delta, dtype=DT, requires_grad=True)
                y = impl.values(
                    spec,
                    period,
                    x,
                    torch.tensor(0.013),
                    width,
                    support="centered",
                    normalization=norm,
                ).sum()
                dy = torch.autograd.grad(y, width, create_graph=True)[0]
                ddy = torch.autograd.grad(dy, width)[0]
                samples.append(torch.stack((y, dy * boundary, ddy * boundary**2)))
            torch.testing.assert_close(samples[0], samples[2], atol=2e-5, rtol=2e-5)
    hard = IntegerFourierPeriodSpec(tile_width=0.8, minimum_half_period=0.25)
    a = impl.values(
        spec,
        hard,
        x,
        torch.tensor(0.013),
        torch.tensor(0.35 - 1e-9, dtype=DT),
        support="centered",
    )
    b = impl.values(
        spec,
        hard,
        x,
        torch.tensor(0.013),
        torch.tensor(0.35 + 1e-9, dtype=DT),
        support="centered",
    )
    assert (a - b).abs().max() > 1e-4


@pytest.mark.parametrize("modes", [4, 8])
def test_whole_tile_cut_is_a_different_function_with_small_not_zero_jumps(modes):
    spec = FourierPeakSpec(modes=modes)
    period = IntegerFourierPeriodSpec(
        tile_width=0.8, minimum_half_period=0.25, transition_fraction=0.25
    )
    x = torch.arange(-16, 17, dtype=DT) * 0.1
    width = torch.tensor(0.332, dtype=DT)
    mu = torch.tensor(0.8 - 0.332, dtype=DT)
    a = impl.values(spec, period, x, mu - 1e-9, width, support="whole_tiles")
    b = impl.values(spec, period, x, mu + 1e-9, width, support="whole_tiles")
    assert (a - b).abs().max() > 1e-7
    # Branch gradients do not include finite jumps caused by adding full tiles.
    for center in (mu - 1e-9, mu + 1e-9):
        value = impl.values(spec, period, x, center, width, support="centered")
        assert bool((value[(x - center).abs() > 1.0] == 0).all())


@pytest.mark.parametrize("modes", [4, 8])
def test_live_multiatom_product_y_dx_amp_center_width_gradients(modes):
    spec = FourierPeakSpec(modes=modes)
    period = IntegerFourierPeriodSpec(
        tile_width=0.8, minimum_half_period=0.25, transition_fraction=0.25
    )
    xin, xout = torch.arange(13, dtype=DT) * 0.1, torch.arange(17, dtype=DT) * 0.1
    generator = torch.Generator().manual_seed(41)
    for step in range(3):
        # Per-axis I and centers remain independent per atom and per step.
        widths = (
            torch.tensor([[0.332, 0.4], [0.585, 0.29], [0.1, 0.13]], dtype=DT)
            + 0.007 * step
        )
        centers = (
            torch.tensor([[0.41, 0.63], [0.21, 1.59], [8.0, 8.0]], dtype=DT)
            + 0.011 * step
        )
        amp = torch.tensor([0.7, -0.2, 0.3], dtype=DT)
        inp = torch.randn(2, 13, generator=generator, dtype=DT)
        cot = torch.randn(2, 17, generator=generator, dtype=DT)
        all_results = []
        for use_oracle in (False, True):
            w, m, a, v = [
                p.detach().clone().requires_grad_() for p in (widths, centers, amp, inp)
            ]
            if use_oracle:
                fi = torch.stack(
                    [
                        oracle(xin, m[j, 0], w[j, 0], spec, period, "whole_tiles")
                        for j in range(3)
                    ],
                    -1,
                )
                fo = torch.stack(
                    [
                        oracle(xout, m[j, 1], w[j, 1], spec, period, "whole_tiles")
                        for j in range(3)
                    ],
                    -1,
                )
            else:
                fi = impl.values(
                    spec, period, xin, m[:, 0], w[:, 0], support="whole_tiles"
                )
                fo = impl.values(
                    spec, period, xout, m[:, 1], w[:, 1], support="whole_tiles"
                )
            atom = fo[:, None, :] * fi[None, :, :]
            # Apply the floor only once to the complete atom's sampled norm.
            atom = atom / torch.linalg.vector_norm(atom, dim=(0, 1)).clamp_min(1e-6)
            output = v @ (atom * a).sum(-1).T
            grads = torch.autograd.grad((output * cot).sum(), (v, a, m, w))
            all_results.append((output, *grads))
        for actual, expected in zip(*all_results):
            torch.testing.assert_close(actual, expected, atol=8e-10, rtol=2e-8)


def test_empty_sample_domain_and_floor():
    spec = FourierPeakSpec()
    period = IntegerFourierPeriodSpec(tile_width=0.8, minimum_half_period=0.25)
    norm = NormalizationSpec(kind="discrete_l2", domain="chart_sites", floor=1e-6)
    result = impl.values_and_derivatives(
        spec,
        period,
        torch.empty(0, dtype=DT),
        torch.tensor([0.0, 10.0]),
        torch.tensor([0.1, 0.3]),
        support="centered",
        normalization=norm,
    )
    assert result[0].shape == (0, 2)
    assert bool((result[3] == 0).all())
    with pytest.raises(ValueError, match="I must"):
        impl.period_components(period, torch.tensor(0.0))


def test_inference_warmed_projection_does_not_poison_autograd():
    shape._projection.cache_clear()
    with torch.inference_mode():
        shape.coefficients(FourierPeakSpec(), torch.tensor(0.5, dtype=DT))
    q = torch.tensor(0.5, dtype=DT, requires_grad=True)
    b, _ = shape.coefficients(FourierPeakSpec(), q)
    assert np.isfinite(float(torch.autograd.grad(b[1], q)[0]))
