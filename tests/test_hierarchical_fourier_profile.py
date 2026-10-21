"""Independent hierarchy oracle, all-live derivatives and boundary gates."""

import math
from dataclasses import replace

import pytest
import torch
from test_effective_fourier_profile import oracle_coefficients

from torchcst import FourierPeakSpec, HierarchicalFourierPeriodSpec, NormalizationSpec
from torchcst._backends.torch.kernels import hierarchical_fourier as impl

DT = torch.float64
NORM = NormalizationSpec(kind="discrete_l2", domain="chart_sites", floor=1e-8)


def make_spec(**kw):
    return HierarchicalFourierPeriodSpec(
        minimum_tile_width=0.1,
        minimum_half_period=0.025,
        relative_width_floor=0.54,
        **kw,
    )


def oracle(sites, mu, width, modes, spec, support):
    thresholds, tiles, steps = [], [], []
    for j in range(spec.level_count):
        tile = spec.minimum_tile_width * 2**j
        step = spec.minimum_half_period * (2**j if spec.k_stride == "dyadic" else 1)
        tiles.append(tile)
        steps.append(step)
        thresholds.append(
            (tile / 2 + (1 + spec.period_transition_fraction) * step)
            / (1 / spec.relative_width_floor - 1)
        )
    live = float(width.detach())
    coarse = max(j for j, threshold in enumerate(thresholds) if live >= threshold)
    fine = max(coarse - 1, 0)
    if coarse and spec.hierarchy_transition_fraction:
        interval = spec.hierarchy_transition_fraction * (
            thresholds[coarse] - thresholds[fine]
        )
        z = ((width - thresholds[coarse]) / interval).clamp(0, 1)
        h = 6 * z**5 - 15 * z**4 + 10 * z**3
    else:
        h = width * 0 + 1
    result = sites * 0
    for level, outer in ((fine, 1 - h), (coarse, h)):
        tile, step = tiles[level], steps[level]
        low = math.ceil((live + tile / 2) / step) * step
        if spec.period_transition_fraction:
            z = (
                (width - low + tile / 2 + spec.period_transition_fraction * step)
                / (spec.period_transition_fraction * step)
            ).clamp(0, 1)
            w = 6 * z**5 - 15 * z**4 + 10 * z**3
        else:
            w = width * 0
        for L, inner in ((low, 1 - w), (low + step, w)):
            b = oracle_coefficients(width / L, modes)
            k = torch.arange(modes + 1, dtype=DT)
            y = torch.cos((sites - mu)[:, None] * math.pi * k / L) @ b
            if support == "centered":
                mask = abs(sites - mu) < L
            else:
                left = math.floor(float((mu - width).detach()) / tile) * tile
                right = math.ceil(float((mu + width).detach()) / tile) * tile
                mask = (sites >= left) & (sites < right)
            result = result + outer * inner * y * mask
    return result


def test_validation_and_exact_global_k_convention():
    for key, value in (
        ("minimum_tile_width", 0),
        ("minimum_half_period", -1),
        ("relative_width_floor", 1),
        ("level_count", True),
        ("k_stride", "other"),
        ("hierarchy_transition_fraction", 1),
        ("period_transition_fraction", float("nan")),
    ):
        with pytest.raises(ValueError):
            replace(make_spec(), **{key: value})
    with pytest.raises(ValueError, match="below"):
        impl.components(make_spec(), torch.tensor(0.001))


@pytest.mark.parametrize("stride", ["fixed", "dyadic"])
@pytest.mark.parametrize("gamma", [0.0, 0.3, 0.9])
def test_full_range_safety_weights_and_integer_multiples(stride, gamma):
    spec = make_spec(k_stride=stride, hierarchy_transition_fraction=gamma)
    width = torch.linspace(0.1, 10, 10001, dtype=DT)
    L, tiles, w, dw, k, level = impl.components(spec, width)
    active = w > 1e-13
    assert bool(((2 * L >= 2 * width[:, None] + tiles - 1e-12) | ~active).all())
    assert bool(((width[:, None] / L >= 0.54 - 1e-13) | ~active).all())
    torch.testing.assert_close(L, k * 0.025)
    torch.testing.assert_close(k, k.round())
    if stride == "dyadic":
        assert bool((k.remainder(2.0**level) == 0).all())
    torch.testing.assert_close(w.sum(-1), torch.ones_like(width))
    torch.testing.assert_close(dw.sum(-1), torch.zeros_like(width), atol=2e-12, rtol=0)
    assert bool((w >= -1e-14).all())


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("support", ["centered", "whole_tiles"])
@pytest.mark.parametrize("stride", ["fixed", "dyadic"])
def test_integral_oracle_normalized_values_and_live_gradients(modes, support, stride):
    spec = make_spec(k_stride=stride)
    shape = FourierPeakSpec(modes=modes)
    sites = torch.arange(-400, 401, dtype=DT) * 0.1
    for i in (0.1, 0.19, 0.2, 0.332, 0.8, 1.67, 3.2, 6.11, 10.0):
        width = torch.tensor(i, dtype=DT, requires_grad=True)
        mu = torch.tensor(0.137, dtype=DT, requires_grad=True)
        y, dm, di, norm = impl.values_and_derivatives(
            shape, spec, sites, mu, width, support=support, normalization=NORM
        )
        expected = oracle(sites, mu, width, modes, spec, support)
        expected = expected / expected.norm().clamp_min(1e-8)
        torch.testing.assert_close(y, expected, atol=1e-11, rtol=3e-9)
        cot = torch.sin(sites * 0.43 + 0.17)
        actual = torch.autograd.grad(y @ cot, (mu, width), retain_graph=True)
        target = torch.autograd.grad(expected @ cot, (mu, width))
        for a, b, c in zip(actual, target, (dm @ cot, di @ cot)):
            torch.testing.assert_close(a, b, atol=3e-9, rtol=2e-8)
            torch.testing.assert_close(a, c, atol=3e-9, rtol=2e-8)
        assert norm > 0


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("stride", ["fixed", "dyadic"])
def test_level_and_period_transitions_are_c2_after_sampled_l2(modes, stride):
    spec = make_spec(k_stride=stride)
    tile, _, step, threshold = impl.level_metadata(spec)
    knots = []
    for j in range(1, 7):
        knots.extend(
            (
                float(threshold[j]),
                float(threshold[j] + 0.3 * (threshold[j] - threshold[j - 1])),
            )
        )
        low = float(threshold[j] + 0.5 * threshold[j])
        n = math.ceil((low + float(tile[j]) / 2) / float(step[j]))
        boundary = n * float(step[j]) - float(tile[j]) / 2
        knots.extend((boundary - 0.25 * float(step[j]), boundary))
    x = torch.arange(-400, 401, dtype=DT) * 0.1
    cot = torch.sin(x * 0.31 + 0.27)
    for boundary in knots:
        if not 0.1 < boundary < 10:
            continue
        result = []
        for delta in (-1e-9, 1e-9):
            # Resolve the narrowest period transition, even at I=10.
            I = torch.tensor(
                boundary + delta * spec.minimum_half_period,
                dtype=DT,
                requires_grad=True,
            )
            y = (
                impl.values(
                    FourierPeakSpec(modes=modes),
                    spec,
                    x,
                    torch.tensor(0.137),
                    I,
                    support="centered",
                    normalization=NORM,
                )
                @ cot
            )
            first = torch.autograd.grad(y, I, create_graph=True)[0]
            second = torch.autograd.grad(first, I)[0]
            result.append(torch.stack((y, first * boundary, second * boundary**2)))
        torch.testing.assert_close(result[0], result[1], atol=5e-5, rtol=2e-5)


@pytest.mark.parametrize("modes", [4, 8])
def test_product_y_dx_all_atom_gradients_over_dynamic_steps(modes):
    spec = make_spec()
    shape = FourierPeakSpec(modes=modes)
    x = torch.arange(31, dtype=DT) * 0.1
    z = torch.arange(37, dtype=DT) * 0.1
    generator = torch.Generator().manual_seed(302)
    for step in range(3):
        centers = (
            torch.tensor([[0.137, 0.389], [1.19, 1.631], [1.73, 2.137]], dtype=DT)
            + step * 0.013
        )
        widths = (
            torch.tensor([[0.1, 10.0], [0.193, 0.801], [3.2, 6.11]], dtype=DT)
            + step * 0.003
        )
        amp = torch.tensor([0.3, -0.2, 0.7], dtype=DT)
        inp = torch.randn(2, 31, generator=generator, dtype=DT)
        cot = torch.randn(2, 37, generator=generator, dtype=DT)
        results = []
        for independent in (False, True):
            m, w, a, v = [
                t.clone().requires_grad_() for t in (centers, widths, amp, inp)
            ]
            if independent:
                fi = torch.stack(
                    [
                        oracle(x, m[j, 0], w[j, 0], modes, spec, "whole_tiles")
                        for j in range(3)
                    ],
                    -1,
                )
                fo = torch.stack(
                    [
                        oracle(z, m[j, 1], w[j, 1], modes, spec, "whole_tiles")
                        for j in range(3)
                    ],
                    -1,
                )
            else:
                fi = impl.values(
                    shape, spec, x, m[:, 0], w[:, 0], support="whole_tiles"
                )
                fo = impl.values(
                    shape, spec, z, m[:, 1], w[:, 1], support="whole_tiles"
                )
            product = fo[:, None, :] * fi[None, :, :]
            product = product / torch.linalg.vector_norm(product, dim=(0, 1)).clamp_min(
                1e-8
            )
            y = v @ (product * a).sum(-1).T
            results.append((y, *torch.autograd.grad((y * cot).sum(), (v, a, m, w))))
        for a, b in zip(*results):
            torch.testing.assert_close(a, b, atol=5e-9, rtol=2e-8)
