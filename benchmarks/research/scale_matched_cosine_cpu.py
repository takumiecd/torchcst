"""CPU mathematical prototype: positive cosine kernels with exact scale seams.

M=2p bounds the Fourier order in the coarser basis. The kernel has TWO compact
Fourier pieces, not a single finite Fourier polynomial throughout [-L,L].
No public runtime KernelSpec/dispatcher or GPU implementation is installed.
"""

from dataclasses import dataclass
from functools import cache
from math import comb

import numpy as np
from numpy.polynomial import Chebyshev
from numpy.polynomial.legendre import leggauss

from benchmarks.research.compact_poly_fourier_cpu import geometry
from benchmarks.research.polynomial_fourier_cpu import normalize, smooth


@cache
def quadrature(order=96):
    x, w = leggauss(order)
    return (x + 1) / 2, w / 2


def moments(c, p, order=96):
    """L=1, central lobe only: continuous variance and dc derivative."""
    u, w = quadrature(order)
    radius = np.arccos(c) / np.pi
    x = radius * u
    cs = np.cos(np.pi * x)
    v = (cs - c) / (1 - c)
    q = v**p
    qc = p * v ** (p - 1) * (cs - 1) / (1 - c) ** 2
    a = 2 * radius * (w @ q)
    b = 2 * radius * (w @ (x * x * q))
    ac = 2 * radius * (w @ qc)
    bc = 2 * radius * (w @ (x * x * qc))
    return float(a), float(b), float((bc * a - b * ac) / a**2)


def variance(c, p, order=96):
    a, b, dc = moments(c, p, order)
    return b / a, dc


def solve_c(target, p):
    lo, hi = -0.5, 0.95
    for _ in range(52):
        mid = (lo + hi) / 2
        if variance(mid, p)[0] > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def weight_variance(w):
    return (1 + 7 * w) / (4 * (1 + w)), 1.5 / (1 + w) ** 2


@dataclass(frozen=True)
class Calibration:
    p: int
    c0: float
    rho: float
    d0: float
    residual: Chebyshev
    residual_derivative: Chebyshev

    def evaluate(self, t):
        t = np.asarray(t)
        base = self.c0 + self.d0 * (2 * t**3 - 3 * t * t + t)
        db = self.d0 * (6 * t * t - 6 * t + 1)
        factor = t * t * (1 - t) ** 2
        df = 2 * t * (1 - t) * (1 - 2 * t)
        value = self.residual(t)
        return (
            base + factor * value,
            db + df * value + factor * self.residual_derivative(t),
        )


@cache
def calibration(p, degree=32):
    c0 = 0.5
    v0, dv = variance(c0, p)
    rho = np.sqrt(v0)
    d0 = 2 * np.log(2) * v0 / dv
    t = np.linspace(0.002, 0.998, 193)
    weight = smooth(t)[0]
    target = v0 * 2 ** (2 * t) / (4 * weight_variance(weight)[0])
    exact = np.array([solve_c(float(v), p) for v in target])
    base = c0 + d0 * (2 * t**3 - 3 * t * t + t)
    residual = (exact - base) / (t * t * (1 - t) ** 2)
    poly = Chebyshev.fit(t, residual, degree, domain=[0, 1])
    return Calibration(p, c0, float(rho), float(d0), poly, poly.deriv())


@dataclass(frozen=True)
class Config:
    modes: int = 8
    spacing: float = 0.1
    tile_points: int = 32
    calibration_degree: int = 32

    @property
    def tile(self):
        return self.spacing * self.tile_points


def parameters(sigma, cfg, forced_L=None):
    p = cfg.modes // 2
    cal = calibration(p, cfg.calibration_degree)
    if forced_L is None:
        level = int(np.ceil(np.log2(sigma / (cal.rho * cfg.tile))))
        L = cfg.tile * 2.0**level
    else:
        L = forced_L
    t = float(np.clip(np.log2(2 * sigma / (cal.rho * L)), 0, 1))
    w, dw = smooth(t)
    dw = float(dw / (sigma * np.log(2)))
    c, ct = cal.evaluate(t)
    return {
        "p": p,
        "L": L,
        "t": t,
        "w": float(w),
        "ws": dw,
        "c": float(c),
        "cs": float(ct / (sigma * np.log(2))),
        "rho": cal.rho,
    }


def cosine_coeff(c, p):
    """Exact Laurent expansion of ((cos(theta)-c)/(1-c))**p."""

    def power(n):
        a = np.zeros(2 * p + 1)
        for j in range(n + 1):
            a[p + n - 2 * j] = comb(n, j) / 2**n
        return a

    v = np.zeros(2 * p + 1)
    vc = np.zeros_like(v)
    for n in range(p + 1):
        basis = comb(p, n) * power(n)
        v += basis * (-c) ** (p - n)
        if n < p:
            vc -= basis * (p - n) * (-c) ** (p - n - 1)
    return v / (1 - c) ** p, vc / (1 - c) ** p + p * v / (1 - c) ** (p + 1)


def components(mu, sigma, cfg, forced_L=None):
    state = parameters(sigma, cfg, forced_L)
    p, L, w, ws, c, cs = (state[k] for k in ("p", "L", "w", "ws", "c", "cs"))
    coeff, dc = cosine_coeff(c, p)
    k = np.arange(-p, p + 1)
    result = []
    for scale, weight, derivative in [(L / 2, 1 - w, -ws), (L, w, ws)]:
        if weight == 0 and derivative == 0:
            continue
        omega = np.pi * k / scale
        phase = np.exp(-1j * omega * mu)
        value = weight * coeff * phase
        result.append(
            {
                "L": scale,
                "radius": scale * np.arccos(c) / np.pi,
                "omega": omega,
                "coefficient": value,
                "center": -1j * omega * value,
                "width": (derivative * coeff + weight * dc * cs) * phase,
            }
        )
    return state, result


def raw(x, mu, sigma, cfg, forced_L=None):
    state = parameters(sigma, cfg, forced_L)
    p, L, w, ws, c, cs = (state[k] for k in ("p", "L", "w", "ws", "c", "cs"))
    out = [np.zeros_like(x, dtype=float) for _ in range(3)]
    # Reference uses separate cos/sin evaluations. One-cos double-angle fusion
    # is a future implementation choice, not a measured speed claim.
    for scale, weight, derivative in [(L / 2, 1 - w, -ws), (L, w, ws)]:
        if weight == 0 and derivative == 0:
            continue
        radius = scale * np.arccos(c) / np.pi
        mask = abs(x - mu) <= radius
        theta = np.pi * (x[mask] - mu) / scale
        co = np.cos(theta)
        z = np.maximum((co - c) / (1 - c), 0)
        q = z**p
        qm = p * z ** (p - 1) * np.sin(theta) * np.pi / (scale * (1 - c))
        qc = p * z ** (p - 1) * (co - 1) / (1 - c) ** 2
        out[0][mask] += weight * q
        out[1][mask] += weight * qm
        out[2][mask] += derivative * q + weight * qc * cs
    return tuple(out)


def fourier_raw(x, mu, sigma, cfg, forced_L=None):
    _, terms = components(mu, sigma, cfg, forced_L)
    out = [np.zeros_like(x, dtype=float) for _ in range(3)]
    for term in terms:
        mask = abs(x - mu) <= term["radius"]
        basis = np.exp(1j * np.outer(x[mask], term["omega"]))
        for v, key in zip(out, ("coefficient", "center", "width")):
            v[mask] += np.real(basis @ term[key])
    return tuple(out)


def profile(x, mu, sigma, cfg, forced_L=None):
    return normalize(*raw(x, mu, sigma, cfg, forced_L))


class PrefixBank:
    def __init__(self, x, values, cfg):
        self.x, self.values, self.cfg = x, values, cfg
        self.cache = {}

    def get(self, left, right, k, L):
        level = round(np.log2(L / self.cfg.tile))
        if not np.isclose(L / self.cfg.tile, 2.0**level):
            raise ValueError("non-dyadic frequency")
        n = int(k)
        if n == 0:
            key = (0, 0)
        else:
            while n % 2 == 0:
                n, level = n // 2, level - 1
            key = (n, level)
        if key not in self.cache:
            values = self.values * np.exp(1j * np.pi * k * self.x / L)[None, :]
            self.cache[key] = np.pad(np.cumsum(values, axis=1), ((0, 0), (1, 0)))
        lo = np.searchsorted(self.x, left, side="left")
        hi = np.searchsorted(self.x, right, side="right")
        return self.cache[key][:, hi] - self.cache[key][:, lo]

    @property
    def payload(self):
        return sum(v.nbytes for v in self.cache.values())


def cross(a, b, ka, kb, x, mu):
    base = max(a["L"], b["L"])
    sa, sb = round(base / a["L"]), round(base / b["L"])
    ca = np.zeros((len(a[ka]) - 1) * sa + 1, complex)
    ca[::sa] = a[ka]
    cb = np.zeros((len(b[kb]) - 1) * sb + 1, complex)
    cb[::sb] = b[kb]
    product = np.convolve(ca, cb)
    m = (len(product) - 1) // 2
    radius = min(a["radius"], b["radius"])
    z = [geometry(x, mu - radius, mu + radius, k, base) for k in range(-m, m + 1)]
    return float(np.real(product @ z))


def norm(mu, sigma, cfg, x):
    _, terms = components(mu, sigma, cfg)
    result = np.zeros(3)
    for a in terms:
        for b in terms:
            for j, key in enumerate(("coefficient", "center", "width")):
                result[j] += cross(a, b, "coefficient", key, x, mu)
    return tuple(result)


def projection(mu, sigma, cfg, bank):
    p = cfg.modes // 2
    _, terms = components(mu, sigma, cfg)
    h = [np.zeros(len(bank.values)) for _ in range(3)]
    for term in terms:
        moment = np.array(
            [
                bank.get(mu - term["radius"], mu + term["radius"], k, term["L"])
                for k in range(-p, p + 1)
            ]
        )
        for v, key in zip(h, ("coefficient", "center", "width")):
            v += np.real(term[key] @ moment)
    e, em, es = norm(mu, sigma, cfg, bank.x)
    n = np.sqrt(max(e, 1e-24))
    if e > 1e-24:
        return h[0] / n, h[1] / n - h[0] * em / n**3, h[2] / n - h[0] * es / n**3
    return tuple(v / n for v in h)
