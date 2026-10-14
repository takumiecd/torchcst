"""Native periodic image sum and adjacent log-width interpolation."""

import torch


def interpolation(spec, sigma):
    sigma = sigma.double()
    knots = sigma.new_tensor(spec.width_knots)
    bounded = sigma.clamp(knots[0], knots[-1])
    j = (torch.searchsorted(knots, bounded.contiguous(), right=True) - 1).clamp(
        0, len(knots) - 2
    )
    t = ((bounded / knots[j]).log() / (knots[j + 1] / knots[j]).log()).clamp(0, 1)
    return j, t, knots


def fixed_values(displacement, b, period):
    raw = displacement.double()
    d = (raw - torch.round(raw / period).detach() * period).abs()
    q = torch.exp(-period / b)
    one = -torch.expm1(-period / b)

    def half(z):
        return torch.exp(-z / b) * ((1 + z / b) / one + (period / b) * q / one.square())

    peak = (1 + q) / one + 2 * (period / b) * q / one.square()
    return (half(d) + half(period - d)) / peak


def values(spec, displacement, sigma, period):
    j, t, knots = interpolation(spec, sigma)
    return (1 - t) * fixed_values(
        displacement, knots[j].square(), period
    ) + t * fixed_values(displacement, knots[j + 1].square(), period)
