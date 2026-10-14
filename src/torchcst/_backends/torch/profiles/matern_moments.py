"""Sampled adjacent-width Gram values from immutable exponential moments.

Only static preparation sums over sites. Live centre evaluation uses three
moments for each of four periodic-image orientations and adjacent width pair.
All constants and arithmetic use FP64; no continuum L2 identity is substituted
for the discrete chart norm.
"""

import math

import torch

ORIENTATIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def static_tables(spec, size, period):
    """Return CPU FP64 constants[J,7] and moments[2J-1,4,3].

    Constants columns are lambda, A, B, peak, exp(-lambda*h), q, 1-q.
    Pair rows alternate self and next-width cross products, ending with the
    last self product. Moment columns are sums of v**l exp(-abs(R)*v), l=0..2.
    """
    if type(size) is not int or size < 1:
        raise ValueError("size must be a positive integer")
    if not math.isfinite(period) or period <= 0:
        raise ValueError("period must be positive and finite")
    lam = torch.tensor(spec.width_knots, dtype=torch.float64).square().reciprocal()
    q = torch.exp(-period * lam)
    one = -torch.expm1(-period * lam)
    a = one.reciprocal() + period * lam * q / one.square()
    b = lam / one
    peak = (1 + q) / one + 2 * period * lam * q / one.square()
    step = torch.exp(-lam * (period / size))
    constants = torch.stack((lam, a, b, peak, step, q, one), dim=-1)
    v = torch.arange(size, dtype=torch.float64) * (period / size)
    moments = []
    for row in range(2 * len(lam) - 1):
        j, k = row // 2, (row + 1) // 2
        oriented = []
        for aj, ak in ORIENTATIONS:
            exponent = (aj * lam[j] + ak * lam[k]).abs()
            weights = torch.exp(-exponent * v)
            oriented.append(
                torch.stack(tuple((weights * v**l).sum() for l in range(3)))
            )
        moments.append(torch.stack(oriented))
    return constants, torch.stack(moments)


def evaluate_gram(spec, delta, period, size, static=None):
    """Return (Gram, dGram/ddelta), each shape delta.shape + (J-1,3).

    Last columns are g00, g01, g11 for each adjacent-width interval. Delta is
    the distance from the centre to the first forward grid site, in [0,h].
    Both endpoints represent the same sampled circle. Static tables may be
    passed to reuse preparation; they must match spec, size and period.
    """
    delta = torch.as_tensor(delta, dtype=torch.float64)
    constants, moments = static_tables(spec, size, period) if static is None else static
    constants = constants.to(device=delta.device, dtype=torch.float64)
    moments = moments.to(device=delta.device, dtype=torch.float64)
    count = len(spec.width_knots)
    if constants.shape != (count, 7) or moments.shape != (2 * count - 1, 4, 3):
        raise ValueError("static table shape does not match profile")
    pair_values, pair_derivatives = [], []
    for row in range(2 * count - 1):
        j, k = row // 2, (row + 1) // 2
        lj, aj0, bj, pj = constants[j, :4].unbind()
        lk, ak0, bk, pk = constants[k, :4].unbind()
        total = torch.zeros_like(delta)
        derivative = torch.zeros_like(delta)
        for orientation, (sj, sk) in enumerate(ORIENTATIONS):
            rate = sj * lj + sk * lk
            reverse = rate < 0
            u0 = torch.where(reverse, rate.new_tensor((size - 1) * period / size), 0)
            direction = torch.where(reverse, -1, 1)
            rj = delta + u0 if sj == 1 else period - delta - u0
            rk = delta + u0 if sk == 1 else period - delta - u0
            xj, xk = aj0 + bj * rj, ak0 + bk * rk
            yj, yk = direction * sj * bj, direction * sk * bk
            s0, s1, s2 = moments[row, orientation].unbind()
            polynomial = xj * xk * s0 + (xj * yk + xk * yj) * s1 + yj * yk * s2
            polynomial_derivative = (sj * bj * xk + sk * bk * xj) * s0 + (
                sj * bj * yk + sk * bk * yj
            ) * s1
            weight = torch.exp(-(lj * rj + lk * rk)) / (pj * pk)
            total = total + weight * polynomial
            derivative = derivative + weight * (
                polynomial_derivative - rate * polynomial
            )
        pair_values.append(total)
        pair_derivatives.append(derivative)
    pairs = torch.stack(pair_values, dim=-1)
    derivatives = torch.stack(pair_derivatives, dim=-1)
    indices = torch.arange(count - 1, device=delta.device)[:, None] * 2
    indices = indices + torch.arange(3, device=delta.device)
    return pairs[..., indices], derivatives[..., indices]
