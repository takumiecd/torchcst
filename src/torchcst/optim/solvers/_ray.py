"""Exact minimization of a scalar quartic on [0,1], using cubic roots."""

import torch


def unit_quartic_minimum(coefficients):
    c = coefficients.double()
    scale = c.abs().amax().clamp_min(1e-300)
    c = c / scale
    d, cc, b, a = c[0], 2 * c[1], 3 * c[2], 4 * c[3]
    cubic = a.abs() > 1e-14
    quadratic = (~cubic) & (b.abs() > 1e-14)
    linear = (~cubic) & (~quadratic) & (cc.abs() > 1e-14)
    safe_a = torch.where(cubic, a, 1)
    bn, cn, dn = b / safe_a, cc / safe_a, d / safe_a
    p = cn - bn.square() / 3
    q = 2 * bn.pow(3) / 27 - bn * cn / 3 + dn
    delta = (q / 2).square() + (p / 3).pow(3)
    root_delta = delta.clamp_min(0).sqrt()

    def cbrt(x):
        return torch.sign(x) * x.abs().pow(1 / 3)

    real = cbrt(-q / 2 + root_delta) + cbrt(-q / 2 - root_delta) - bn / 3
    radius = (-p / 3).clamp_min(1e-300).sqrt()
    angle = torch.acos((-q / (2 * radius.pow(3))).clamp(-1, 1)) / 3
    pi = 3.141592653589793
    roots = (
        2 * radius * torch.cos(angle + c.new_tensor([0.0, 2 * pi / 3, 4 * pi / 3]))
        - bn / 3
    )
    roots = torch.where(delta >= 0, real.expand_as(roots), roots)
    qb = torch.where(quadratic, b, 1)
    qdisc = cc.square() - 4 * b * d
    # Stable quadratic formula and product relation for the second root.
    numerator = -0.5 * (cc + torch.copysign(qdisc.clamp_min(0).sqrt(), cc))
    qr1 = numerator / qb
    qr2 = d / torch.where(numerator != 0, numerator, 1)
    candidates = torch.cat(
        (
            c.new_tensor([0.0, 1.0]),
            roots,
            torch.stack((qr1, qr2, -d / torch.where(linear, cc, 1))),
        )
    )
    valid = torch.cat(
        (
            torch.ones(2, device=c.device, dtype=torch.bool),
            cubic.expand(3),
            (quadratic & (qdisc >= 0)).expand(2),
            linear.reshape(1),
        )
    )
    # Two scalar Newton corrections repair cancellation in Cardano's formula.
    # They evaluate only four scalar coefficients, not the full CST objective.
    for _ in range(2):
        derivative = ((a * candidates + b) * candidates + cc) * candidates + d
        second = (3 * a * candidates + 2 * b) * candidates + cc
        correction = derivative / torch.where(second.abs() > 1e-14, second, 1)
        interior = torch.arange(8, device=c.device) >= 2
        candidates = torch.where(
            interior & (second.abs() > 1e-14), candidates - correction, candidates
        )
    valid = valid & torch.isfinite(candidates) & (candidates >= 0) & (candidates <= 1)
    values = ((c[3] * candidates + c[2]) * candidates + c[1]) * candidates.square() + c[
        0
    ] * candidates
    values = torch.where(valid, values, torch.inf)
    return candidates.gather(0, values.argmin().reshape(1)).squeeze(0)


def ray_coefficients(a, b, j, h, metric, inverse_lr, x, v):
    base = torch.einsum("kmp,kp->m", j, x) + 0.5 * torch.einsum(
        "kmpq,kp,kq->m", h, x, x
    )
    first = torch.einsum("kmp,kp->m", j, v) + torch.einsum("kmpq,kp,kq->m", h, x, v)
    second = 0.5 * torch.einsum("kmpq,kp,kq->m", h, v, v)
    c1 = (
        (a * v).sum()
        + torch.einsum("kp,kpq,kq->", x, b, v)
        + inverse_lr * (metric * base * first).sum()
    )
    c2 = (
        0.5 * torch.einsum("kp,kpq,kq->", v, b, v)
        + 0.5 * inverse_lr * (metric * (first.square() + 2 * base * second)).sum()
    )
    c3 = inverse_lr * (metric * first * second).sum()
    c4 = 0.5 * inverse_lr * (metric * second.square()).sum()
    return torch.stack((c1, c2, c3, c4))
