"""Corrected CPU contract: support [-L,L], Fourier period 2L, dyadic L.

Kernel prototype only. Whole-tile-only queries and exact compact support are
different algorithms: the exact shared reference below stores site prefixes.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np

from benchmarks.research.polynomial_fourier_cpu import (
    fourier_coeff,
    native,
    normalize,
    smooth,
)


@dataclass(frozen=True)
class Config:
    modes: int = 8
    spacing: float = 0.1
    tile_points: int = 32
    minimum_L_tiles: int = 1
    native_ratio: float = 0.65  # R/L0
    level_ratio: float = 0.85  # R/L_child at dyadic transition
    overlap: float = 0.08

    @property
    def tile(self):
        return self.spacing * self.tile_points

    @property
    def L0(self):
        return self.tile * self.minimum_L_tiles


@cache
def endpoint_projection(modes):
    """Least coefficient change, with F(1)=F''(1)=0 and F(0)=1.

    Coordinate is r/L. Odd derivatives at both ends vanish for cosine series.
    Zero extension is therefore C3. The matrix is sigma-independent.
    """
    k = np.arange(modes + 1, dtype=float)
    sign = (-1.0) ** k
    matrix = np.stack([sign, sign * k * k / modes**2, np.ones_like(k)])
    right = matrix.T @ np.linalg.inv(matrix @ matrix.T)
    return np.eye(modes + 1) - right @ matrix, right @ np.array([0.0, 0.0, 1.0])


def coefficients(sigma, L, modes):
    c, ds = fourier_coeff(sigma, 2 * L, modes)
    a = np.r_[c[modes].real, 2 * c[modes + 1 :].real]
    da = np.r_[ds[modes].real, 2 * ds[modes + 1 :].real]
    projection, offset = endpoint_projection(modes)
    a, da = projection @ a + offset, projection @ da
    return (
        np.r_[a[:0:-1] / 2, a[0], a[1:] / 2].astype(complex),
        np.r_[da[:0:-1] / 2, da[0], da[1:] / 2].astype(complex),
    )


def gate(radius, boundary, cfg, hard=False):
    if hard:
        return float(radius >= boundary), 0.0
    low = boundary * (1 - cfg.overlap)
    width = 2 * cfg.overlap * boundary
    value, derivative = smooth((radius - low) / width)
    return float(value), float(derivative * np.sqrt(11) / width)


def components(mu, sigma, cfg, hard=False):
    radius = np.sqrt(11) * sigma
    alpha, ds_alpha = gate(radius, cfg.native_ratio * cfg.L0, cfg, hard)
    wp, dwp = 1 - alpha, -ds_alpha
    if not alpha and not ds_alpha:
        return wp, dwp, []
    nearest = max(0, round(np.log2(radius / (cfg.level_ratio * cfg.L0))))
    boundary = cfg.level_ratio * cfg.L0 * 2**nearest
    if boundary * (1 - cfg.overlap) <= radius <= boundary * (1 + cfg.overlap):
        v, dv = gate(radius, boundary, cfg, hard)
        levels = [
            (cfg.L0 * 2**nearest, 1 - v, -dv),
            (cfg.L0 * 2 ** (nearest + 1), v, dv),
        ]
    else:
        j = max(0, int(np.ceil(np.log2(radius / (cfg.level_ratio * cfg.L0)))))
        levels = [(cfg.L0 * 2**j, 1.0, 0.0)]
    terms = []
    k = np.arange(-cfg.modes, cfg.modes + 1)
    for L, b, db in levels:
        weight = alpha * b
        dw = ds_alpha * b + alpha * db
        if weight == 0 and dw == 0:
            continue
        c, ds = coefficients(sigma, L, cfg.modes)
        omega = np.pi * k / L
        phase = np.exp(-1j * omega * mu)
        value = weight * c * phase
        terms.append(
            {
                "L": L,
                "period": 2 * L,
                "omega": omega,
                "coefficient": value,
                "center": -1j * omega * value,
                "width": (weight * ds + dw * c) * phase,
            }
        )
    return wp, dwp, terms


def raw(x, mu, sigma, cfg, hard=False):
    wp, dwp, terms = components(mu, sigma, cfg, hard)
    p, pm, ps = native(x, mu, sigma)
    q, qm, qs = wp * p, wp * pm, wp * ps + dwp * p
    # Native is already zero at R<L0 throughout its active interval.
    for term in terms:
        mask = abs(x - mu) <= term["L"]
        e = np.exp(1j * np.outer(x[mask], term["omega"]))
        for v, key in [(q, "coefficient"), (qm, "center"), (qs, "width")]:
            v[mask] += np.real(e @ term[key])
    return q, qm, qs


def profile(x, mu, sigma, cfg, hard=False):
    return normalize(*raw(np.asarray(x), mu, sigma, cfg, hard))


class PrefixBank:
    """Exact range moments: two gathers per frequency, including partial tiles.

    A prefix at every site is explicitly retained; this is not the payload of
    a whole-tile-only bank. A new X requires rebuilding every live frequency.
    """

    def __init__(self, x, values, cfg):
        self.x, self.values, self.cfg = x, values, cfg
        self.cache = {}
        self.products = self.hits = 0

    def key(self, k, L):
        d = round(L / self.cfg.L0)
        while k and k % 2 == 0 and d > 1:
            k, d = k // 2, d // 2
        return (0, 1) if k == 0 else (k, d)

    def get(self, left, right, k, L):
        key = self.key(k, L)
        if key not in self.cache:
            w = np.exp(1j * np.pi * k * self.x / L)
            prefix = np.cumsum(self.values * w[None, :], axis=1)
            self.cache[key] = np.pad(prefix, ((0, 0), (1, 0)))
            self.products += self.values.size
        else:
            self.hits += 1
        lo = np.searchsorted(self.x, left, side="left")
        hi = np.searchsorted(self.x, right, side="right")
        v = self.cache[key]
        return v[:, hi] - v[:, lo]

    @property
    def payload(self):
        return sum(v.nbytes for v in self.cache.values())


def geometry(x, left, right, k, L):
    lo = np.searchsorted(x, left, side="left")
    hi = np.searchsorted(x, right, side="right")
    n = int(hi - lo)
    if not n:
        return 0j
    spacing = (x[-1] - x[0]) / (len(x) - 1)
    theta = np.pi * k * spacing / L
    return (
        n
        * np.sinc(n * theta / (2 * np.pi))
        / np.sinc(theta / (2 * np.pi))
        * np.exp(1j * np.pi * k / L * (x[lo] + (n - 1) * spacing / 2))
    )


def cross(a, b, ka, kb, x, mu):
    base = max(a["L"], b["L"])
    sa, sb = round(base / a["L"]), round(base / b["L"])
    ca = np.zeros((len(a[ka]) - 1) * sa + 1, complex)
    ca[::sa] = a[ka]
    cb = np.zeros((len(b[kb]) - 1) * sb + 1, complex)
    cb[::sb] = b[kb]
    c = np.convolve(ca, cb)
    m = (len(c) - 1) // 2
    radius = min(a["L"], b["L"])
    z = [geometry(x, mu - radius, mu + radius, k, base) for k in range(-m, m + 1)]
    return float(np.real(np.dot(c, z)))


def norm(mu, sigma, cfg, x):
    wp, dwp, terms = components(mu, sigma, cfg)
    ids = (
        np.flatnonzero(abs(x - mu) < np.sqrt(11) * sigma)
        if wp or dwp
        else np.array([], int)
    )
    p, pm, ps = native(x[ids], mu, sigma)
    p, pm, ps = wp * p, wp * pm, wp * ps + dwp * p
    e, em, es = p @ p, p @ pm, p @ ps
    for a in terms:
        for b in terms:
            e += cross(a, b, "coefficient", "coefficient", x, mu)
            em += cross(a, b, "coefficient", "center", x, mu)
            es += cross(a, b, "coefficient", "width", x, mu)
        mask = abs(x[ids] - mu) <= a["L"]
        basis = np.exp(1j * np.outer(x[ids][mask], a["omega"]))
        f, fm, fs = (np.real(basis @ a[k]) for k in ("coefficient", "center", "width"))
        e += 2 * p[mask] @ f
        em += pm[mask] @ f + p[mask] @ fm
        es += ps[mask] @ f + p[mask] @ fs
    return float(e), float(em), float(es)


def projection(mu, sigma, cfg, bank):
    wp, dwp, terms = components(mu, sigma, cfg)
    ids = (
        np.flatnonzero(abs(bank.x - mu) < np.sqrt(11) * sigma)
        if wp or dwp
        else np.array([], int)
    )
    p, pm, ps = native(bank.x[ids], mu, sigma)
    h = wp * (bank.values[:, ids] @ p)
    hm = wp * (bank.values[:, ids] @ pm)
    hs = wp * (bank.values[:, ids] @ ps) + dwp * (bank.values[:, ids] @ p)
    for term in terms:
        moment = np.array(
            [
                bank.get(mu - term["L"], mu + term["L"], k, term["L"])
                for k in range(-cfg.modes, cfg.modes + 1)
            ]
        )
        for v, key in [(h, "coefficient"), (hm, "center"), (hs, "width")]:
            v += np.real(term[key] @ moment)
    e, em, es = norm(mu, sigma, cfg, bank.x)
    n = np.sqrt(max(e, 1e-24))
    if e > 1e-24:
        hm, hs = hm / n - h * em / n**3, hs / n - h * es / n**3
    else:
        hm, hs = hm / n, hs / n
    return h / n, hm, hs
