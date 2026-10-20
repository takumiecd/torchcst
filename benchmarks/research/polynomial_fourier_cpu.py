"""CPU mathematical screen, not a runtime KernelSpec or GPU benchmark.

P=(1-u*u)^4_+, radius=sqrt(11)*sigma: continuous L1 variance is sigma**2.
Fourier coefficients integrate this mother shape over one period. A finite
Fourier polynomial, restricted to whole tiles, defines the broad candidate.
Quintic blends are part of the new candidate, not an additional spatial window.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np
from numpy.polynomial.legendre import leggauss


@dataclass(frozen=True)
class Config:
    modes: int = 8  # positive harmonics; real feature count = 1+2M
    spacing: float = 0.1
    tile_points: int = 32
    period_tiles: int = 8
    rho: float = 0.34
    overlap: float = 0.08
    native_start: float = 0.16  # radius / minimum period
    native_end: float = 0.22
    tile_blend: bool = False

    @property
    def tile(self):
        return self.spacing * self.tile_points

    @property
    def period0(self):
        return self.tile * self.period_tiles


def smooth(t):
    z = np.clip(t, 0, 1)
    return z**3 * (10 + z * (-15 + 6 * z)), 30 * z**2 * (1 - z) ** 2


def native(x, mu, sigma):
    r = np.sqrt(11.0) * sigma
    u = (np.asarray(x) - mu) / r
    v = np.maximum(1 - u * u, 0)
    return v**4, 8 * u * v**3 / r, 8 * u * u * v**3 / sigma


@cache
def quadrature(order=96):
    u, w = leggauss(order)
    return u, w * (1 - u * u) ** 4


def fourier_coeff(sigma, period, modes, order=None):
    """Return complex-series coefficients and d/dsigma, before center phase."""
    k = np.arange(-modes, modes + 1)
    radius = np.sqrt(11.0) * sigma
    z = 2 * np.pi * k * radius / period
    if order is None:
        # Integral P(u) exp(izu) du = 768*j4(z)/z**4.
        # Its derivative is -768*j5(z)/z**4. Even symmetry avoids negative
        # arguments in spherical_jn; z=0 uses the exact limiting integral.
        t = abs(z)
        nonzero = t != 0
        integral = np.full_like(t, 256 / 315)
        zintegral_prime = np.zeros_like(t)
        # Stable small-z series; large-z closed trigonometric j4/j5 formula.
        small = t < 4
        ts = t[small]
        term = np.full_like(ts, 256 / 315)
        v, dv = term.copy(), np.zeros_like(ts)
        for n in range(20):
            term *= -(ts * ts) / ((2 * n + 2) * (2 * n + 11))
            v += term
            dv += (2 * n + 2) * term
        integral[small], zintegral_prime[small] = v, dv
        large = nonzero & ~small
        z = t[large]
        z2 = z * z
        integral[large] = (
            768
            * (
                (105 - 45 * z2 + z2 * z2) * np.sin(z)
                - (105 * z - 10 * z * z2) * np.cos(z)
            )
            / z**9
        )
        zintegral_prime[large] = (
            -768
            * (
                (945 - 420 * z2 + 15 * z2 * z2) * np.sin(z)
                - (945 * z - 105 * z * z2 + z * z2 * z2) * np.cos(z)
            )
            / z**9
        )
        return (
            (radius / period * integral).astype(complex),
            (np.sqrt(11.0) / period * (integral + zintegral_prime)).astype(complex),
        )
    u, w = quadrature(order)
    phase = z[:, None] * u
    c = radius / period * (np.cos(phase) @ w)
    ds = np.sqrt(11.0) / period * ((np.cos(phase) - phase * np.sin(phase)) @ w)
    return c.astype(complex), ds.astype(complex)


def routes(sigma, cfg, hard=False):
    """(native weight, derivative, [(L, Fourier weight, derivative), ...])."""
    r = np.sqrt(11.0) * sigma
    l0 = cfg.period0
    if hard:
        level = max(0, int(np.ceil(np.log2(r / (cfg.rho * l0)))))
        return (
            (1.0, 0.0, [])
            if r < cfg.native_end * l0
            else (0.0, 0.0, [(l0 * 2**level, 1.0, 0.0)])
        )
    a, da = smooth((r / l0 - cfg.native_start) / (cfg.native_end - cfg.native_start))
    da *= np.sqrt(11.0) / (l0 * (cfg.native_end - cfg.native_start))
    if a == 0 and da == 0:
        return 1.0, 0.0, []
    nearest = max(0, int(np.round(np.log2(r / (cfg.rho * l0)))))
    boundary = cfg.rho * l0 * 2**nearest
    low, high = boundary * (1 - cfg.overlap), boundary * (1 + cfg.overlap)
    if low <= r <= high:
        v, dv = smooth((r - low) / (high - low))
        dv *= np.sqrt(11.0) / (high - low)
        levels = [(l0 * 2**nearest, 1 - v, -dv), (l0 * 2 ** (nearest + 1), v, dv)]
    else:
        j = max(0, int(np.ceil(np.log2(r / (cfg.rho * l0)))))
        levels = [(l0 * 2**j, 1.0, 0.0)]
    return (
        1 - a,
        -da,
        [
            (l, a * b, da * b + a * db)
            for l, b, db in levels
            if a * b != 0 or da * b + a * db != 0
        ],
    )


def components(mu, sigma, cfg, hard=False):
    wp, dwp, levels = routes(sigma, cfg, hard)
    result = []
    for period, weight, dweight in levels:
        k = np.arange(-cfg.modes, cfg.modes + 1)
        omega = 2 * np.pi * k / period
        c, ds = fourier_coeff(sigma, period, cfg.modes)
        phase = np.exp(-1j * omega * mu)
        coefficient = weight * c * phase
        center = -1j * omega * coefficient
        width = (weight * ds + dweight * c) * phase
        count = round(period / cfg.tile) - 1  # odd, all inside one period
        first = int(np.floor(mu / cfg.tile)) - count // 2
        if cfg.tile_blend:
            blend, db = smooth(mu / cfg.tile - np.floor(mu / cfg.tile))
            db /= cfg.tile
            blocks = [(first, 1 - blend, -db), (first + 1, blend, db)]
        else:
            blocks = [(first, 1.0, 0.0)]
        for start, b, db in blocks:
            if b == 0 and db == 0:
                continue
            result.append(
                {
                    "period": period,
                    "omega": omega,
                    "coefficient": b * coefficient,
                    "center": b * center + db * coefficient,
                    "width": b * width,
                    "first": start,
                    "count": count,
                }
            )
    return wp, dwp, result


def raw(x, mu, sigma, cfg, hard=False):
    p, pm, ps = native(x, mu, sigma)
    wp, dwp, terms = components(mu, sigma, cfg, hard)
    q, qm, qs = wp * p, wp * pm, wp * ps + dwp * p
    for term in terms:
        use = (x >= term["first"] * cfg.tile) & (
            x < (term["first"] + term["count"]) * cfg.tile
        )
        basis = np.exp(1j * np.outer(x[use], term["omega"]))
        for dest, key in ((q, "coefficient"), (qm, "center"), (qs, "width")):
            dest[use] += (basis @ term[key]).real
    return q, qm, qs


def normalize(q, qm, qs, floor=1e-12):
    norm = max(float(np.linalg.norm(q)), floor)
    active = np.linalg.norm(q) > floor
    out = [q / norm]
    for dq in (qm, qs):
        out.append(dq / norm - q * np.dot(q, dq) / norm**3 if active else dq / norm)
    return tuple(out)


def profile(x, mu, sigma, cfg, hard=False):
    return normalize(*raw(np.asarray(x), mu, sigma, cfg, hard))


class TileBank:
    """Input-dependent per-tile moments; rational-frequency aliases are shared.

    This is a lazy CPU cost instrument. No per-site prefix is stored. A new X
    requires a fresh bank. Geometry-only moments have a separate lifetime.
    """

    def __init__(self, x, values, cfg):
        self.x, self.values, self.cfg = x, values, cfg
        self.cache = {}
        self.build_terms = 0
        self.hits = self.misses = 0
        self.tiles = np.floor((x + 1e-10 * cfg.spacing) / cfg.tile).astype(int)
        self.members = {
            int(t): np.flatnonzero(self.tiles == t) for t in np.unique(self.tiles)
        }

    def frequency_key(self, harmonic, period):
        denominator = round(period / self.cfg.period0)
        n, d = int(harmonic), denominator
        while n and n % 2 == 0 and d > 1:
            n, d = n // 2, d // 2
        return (0, 1) if n == 0 else (n, d)

    def moment(self, first, count, harmonic, period):
        key = self.frequency_key(harmonic, period)
        out = np.zeros(self.values.shape[0], complex)
        for tile in range(first, first + count):
            if tile not in self.members:
                continue
            cache_key = (tile, key)
            if cache_key not in self.cache:
                ids = self.members[tile]
                basis = np.exp(2j * np.pi * harmonic * self.x[ids] / period)
                self.cache[cache_key] = self.values[:, ids] @ basis
                self.build_terms += len(ids) * self.values.shape[0]
                self.misses += 1
            else:
                self.hits += 1
            out += self.cache[cache_key]
        return out

    @property
    def payload(self):
        return sum(v.nbytes for v in self.cache.values())


def geometry(first, count, period, harmonic, bank):
    """Exact sampled trig moment, using cached tile sums (not orthogonality)."""
    return bank.moment(first, count, harmonic, period)[0]


def fourier_cross(a, b, ka, kb, bank):
    first = max(a["first"], b["first"])
    count = min(a["first"] + a["count"], b["first"] + b["count"]) - first
    if count <= 0:
        return 0.0
    base = max(a["period"], b["period"])
    sa, sb = round(base / a["period"]), round(base / b["period"])
    va = np.zeros((len(a[ka]) - 1) * sa + 1, complex)
    vb = np.zeros((len(b[kb]) - 1) * sb + 1, complex)
    va[::sa], vb[::sb] = a[ka], b[kb]
    product = np.convolve(va, vb)
    low = -(len(product) - 1) // 2
    moments = [
        geometry(first, count, base, k, bank) for k in range(low, low + len(product))
    ]
    return float(np.dot(product, moments).real)


def sampled_norm(mu, sigma, cfg, bank, hard=False):
    """E, <q,q_mu>, <q,q_sigma>; no native correction at Fourier boundaries.

    Native/Fourier overlap needs an explicit mixed term over native support.
    That cost is included, not hidden as a cheap Fourier-only normalization.
    """
    wp, dwp, terms = components(mu, sigma, cfg, hard)
    radius = np.sqrt(11.0) * sigma
    ids = np.flatnonzero(abs(bank.x - mu) < radius) if wp or dwp else np.array([], int)
    p, pm, ps = native(bank.x[ids], mu, sigma)
    pv, mv, sv = wp * p, wp * pm, wp * ps + dwp * p
    e, em, es = np.dot(pv, pv), np.dot(pv, mv), np.dot(pv, sv)
    mixed_site_modes = 0
    for a in terms:
        for b in terms:
            e += fourier_cross(a, b, "coefficient", "coefficient", bank)
            em += fourier_cross(a, b, "coefficient", "center", bank)
            es += fourier_cross(a, b, "coefficient", "width", bank)
        mask = (bank.x[ids] >= a["first"] * cfg.tile) & (
            bank.x[ids] < (a["first"] + a["count"]) * cfg.tile
        )
        basis = np.exp(1j * np.outer(bank.x[ids][mask], a["omega"]))
        f, fm, fs = (
            np.real(basis @ a[key]) for key in ("coefficient", "center", "width")
        )
        e += 2 * np.dot(pv[mask], f)
        em += np.dot(mv[mask], f) + np.dot(pv[mask], fm)
        es += np.dot(sv[mask], f) + np.dot(pv[mask], fs)
        mixed_site_modes += mask.sum() * len(a["omega"])
    return float(e), float(em), float(es), int(mixed_site_modes)


def shared_projection(mu, sigma, cfg, bank, norm_bank):
    wp, dwp, terms = components(mu, sigma, cfg)
    radius = np.sqrt(11.0) * sigma
    ids = np.flatnonzero(abs(bank.x - mu) < radius) if wp or dwp else np.array([], int)
    p, pm, ps = native(bank.x[ids], mu, sigma)
    h = wp * (bank.values[:, ids] @ p)
    hm = wp * (bank.values[:, ids] @ pm)
    hs = wp * (bank.values[:, ids] @ ps) + dwp * (bank.values[:, ids] @ p)
    for term in terms:
        moments = np.array(
            [
                bank.moment(term["first"], term["count"], k, term["period"])
                for k in range(-cfg.modes, cfg.modes + 1)
            ]
        )
        for dest, key in ((h, "coefficient"), (hm, "center"), (hs, "width")):
            dest += np.real(term[key] @ moments)
    e, em, es, mixed = sampled_norm(mu, sigma, cfg, norm_bank)
    n = np.sqrt(max(e, 1e-24))
    if e > 1e-24:
        hm, hs = hm / n - h * em / n**3, hs / n - h * es / n**3
    else:
        hm, hs = hm / n, hs / n
    return h / n, hm, hs, mixed
