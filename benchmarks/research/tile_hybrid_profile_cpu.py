"""CPU-only polynomial/Fourier profile with exact and whole-tile executors.

The whole-tile executor drops partial tiles and is explicitly approximate. It
has small but nonzero center jumps; it is not the differentiable exact kernel.
No public runtime profile, KernelSpec, dispatcher, or GPU implementation.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np

from benchmarks.research.compact_poly_fourier_cpu import Config as BaseConfig
from benchmarks.research.compact_poly_fourier_cpu import PrefixBank, gate, geometry
from benchmarks.research.polynomial_fourier_cpu import fourier_coeff, native, normalize


@dataclass(frozen=True)
class Config(BaseConfig):
    minimum_L_tiles: int = 8
    endpoint_zeros: int = 3


@cache
def projection_matrix(modes, zeros):
    """L2(r) optimal constrained cosine projection, not coefficient-Euclidean."""
    if not 1 <= zeros < modes:
        raise ValueError("endpoint_zeros must be between 1 and M-1")
    k = np.arange(modes + 1, dtype=float)
    a = np.stack(
        [(-1.0) ** k * (k / modes) ** (2 * j) for j in range(zeros)] + [np.ones_like(k)]
    )
    inv_metric = np.diag(np.r_[0.5, np.ones(modes)])
    right = np.linalg.solve(a @ inv_metric @ a.T, a @ inv_metric).T
    return np.eye(modes + 1) - right @ a, right[:, -1]


def coefficients(sigma, L, cfg):
    c, cs = fourier_coeff(sigma, 2 * L, cfg.modes)
    a = np.r_[c[cfg.modes].real, 2 * c[cfg.modes + 1 :].real]
    ds = np.r_[cs[cfg.modes].real, 2 * cs[cfg.modes + 1 :].real]
    matrix, offset = projection_matrix(cfg.modes, cfg.endpoint_zeros)
    a, ds = matrix @ a + offset, matrix @ ds
    return tuple(np.r_[v[:0:-1] / 2, v[0], v[1:] / 2].astype(complex) for v in (a, ds))


def components(mu, sigma, cfg):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = np.sqrt(11) * sigma
    alpha, alpha_s = gate(radius, cfg.native_ratio * cfg.L0, cfg)
    wp, wps = 1 - alpha, -alpha_s
    if not alpha and not alpha_s:
        return wp, wps, []
    nearest = max(0, round(np.log2(radius / (cfg.level_ratio * cfg.L0))))
    boundary = cfg.level_ratio * cfg.L0 * 2**nearest
    if boundary * (1 - cfg.overlap) <= radius <= boundary * (1 + cfg.overlap):
        v, vs = gate(radius, boundary, cfg)
        levels = [
            (cfg.L0 * 2**nearest, 1 - v, -vs),
            (cfg.L0 * 2 ** (nearest + 1), v, vs),
        ]
    else:
        level = max(0, int(np.ceil(np.log2(radius / (cfg.level_ratio * cfg.L0)))))
        levels = [(cfg.L0 * 2**level, 1.0, 0.0)]
    result = []
    for L, w, ws in levels:
        w, ws = alpha * w, alpha_s * w + alpha * ws
        if not w and not ws:
            continue
        c, cs = coefficients(sigma, L, cfg)
        omega = np.pi * np.arange(-cfg.modes, cfg.modes + 1) / L
        phase = np.exp(-1j * omega * mu)
        a = w * c * phase
        result.append(
            {
                "L": L,
                "omega": omega,
                "coefficient": a,
                "center": -1j * omega * a,
                "width": (ws * c + w * cs) * phase,
                "left": mu - L,
                "right": mu + L,
            }
        )
    return wp, wps, result


def bounded_components(mu, sigma, cfg, origin, whole_tiles=False):
    wp, wps, terms = components(mu, sigma, cfg)
    if whole_tiles:
        for t in terms:
            t["left"] = origin + np.ceil((t["left"] - origin) / cfg.tile) * cfg.tile
            t["right"] = origin + np.floor((t["right"] - origin) / cfg.tile) * cfg.tile
    return wp, wps, terms


def raw(x, mu, sigma, cfg, whole_tiles=False):
    wp, wps, terms = bounded_components(mu, sigma, cfg, x[0], whole_tiles)
    p, pm, ps = native(x, mu, sigma)
    out = [wp * p, wp * pm, wp * ps + wps * p]
    for t in terms:
        use = (x >= t["left"]) & (x < t["right"])
        e = np.exp(1j * np.outer(x[use], t["omega"]))
        for dest, key in zip(out, ("coefficient", "center", "width")):
            dest[use] += np.real(e @ t[key])
    return tuple(out)


def profile(x, mu, sigma, cfg, whole_tiles=False):
    return normalize(*raw(x, mu, sigma, cfg, whole_tiles))


class TileBank(PrefixBank):
    """Only a prefix over whole-tile moments, never a prefix at each site."""

    def get(self, left, right, k, L):
        key = self.key(k, L)
        points = self.cfg.tile_points
        if len(self.x) % points:
            raise ValueError("N must be divisible by tile_points")
        if key not in self.cache:
            v = self.values * np.exp(1j * np.pi * k * self.x / L)[None, :]
            v = v.reshape(len(v), -1, points).sum(axis=2)
            self.cache[key] = np.pad(np.cumsum(v, axis=1), ((0, 0), (1, 0)))
            self.products += self.values.size
        else:
            self.hits += 1
        lo = max(0, round((left - self.x[0]) / self.cfg.tile))
        hi = min(len(self.x) // points, round((right - self.x[0]) / self.cfg.tile))
        lo, hi = min(lo, self.cache[key].shape[1] - 1), max(hi, 0)
        return (
            self.cache[key][:, hi] - self.cache[key][:, lo]
            if hi > lo
            else np.zeros(len(self.values), complex)
        )


def cross(a, b, ka, kb, x):
    base = max(a["L"], b["L"])
    sa, sb = round(base / a["L"]), round(base / b["L"])
    ca = np.zeros((len(a[ka]) - 1) * sa + 1, complex)
    cb = np.zeros((len(b[kb]) - 1) * sb + 1, complex)
    ca[::sa], cb[::sb] = a[ka], b[kb]
    c = np.convolve(ca, cb)
    m = (len(c) - 1) // 2
    left, right = max(a["left"], b["left"]), min(a["right"], b["right"])
    if right <= left:
        return 0.0
    # Half-open query; Fourier endpoints vanish in the exact evaluator.
    right = np.nextafter(right, -np.inf)
    z = [geometry(x, left, right, k, base) for k in range(-m, m + 1)]
    return float(np.real(c @ z))


def norm(mu, sigma, cfg, x, whole_tiles=False):
    wp, wps, terms = bounded_components(mu, sigma, cfg, x[0], whole_tiles)
    ids = (
        np.flatnonzero(abs(x - mu) < np.sqrt(11) * sigma)
        if wp or wps
        else np.array([], int)
    )
    p, pm, ps = native(x[ids], mu, sigma)
    p, pm, ps = wp * p, wp * pm, wp * ps + wps * p
    result = np.array([p @ p, p @ pm, p @ ps])
    for a in terms:
        for b in terms:
            for j, key in enumerate(("coefficient", "center", "width")):
                result[j] += cross(a, b, "coefficient", key, x)
        use = (x[ids] >= a["left"]) & (x[ids] < a["right"])
        basis = np.exp(1j * np.outer(x[ids][use], a["omega"]))
        f, fm, fs = [
            np.real(basis @ a[key]) for key in ("coefficient", "center", "width")
        ]
        result += [2 * p[use] @ f, pm[use] @ f + p[use] @ fm, ps[use] @ f + p[use] @ fs]
    return tuple(result)


def projection(mu, sigma, cfg, bank, whole_tiles=False):
    wp, wps, terms = bounded_components(mu, sigma, cfg, bank.x[0], whole_tiles)
    ids = (
        np.flatnonzero(abs(bank.x - mu) < np.sqrt(11) * sigma)
        if wp or wps
        else np.array([], int)
    )
    p, pm, ps = native(bank.x[ids], mu, sigma)
    h = wp * (bank.values[:, ids] @ p)
    hm = wp * (bank.values[:, ids] @ pm)
    hs = wp * (bank.values[:, ids] @ ps) + wps * (bank.values[:, ids] @ p)
    for t in terms:
        v = np.array(
            [
                bank.get(
                    t["left"],
                    np.nextafter(t["right"], -np.inf)
                    if not whole_tiles
                    else t["right"],
                    k,
                    t["L"],
                )
                for k in range(-cfg.modes, cfg.modes + 1)
            ]
        )
        for dest, key in zip((h, hm, hs), ("coefficient", "center", "width")):
            dest += np.real(t[key] @ v)
    e, em, es = norm(mu, sigma, cfg, bank.x, whole_tiles)
    n = np.sqrt(max(e, 1e-24))
    return (
        (h / n, hm / n - h * em / n**3, hs / n - h * es / n**3)
        if e > 1e-24
        else (h / n, hm / n, hs / n)
    )


def stream_projection(mu, sigma, cfg, x, values):
    """Exact coarse-tile moments plus streamed, shared partial-tile prefixes.

    Group metadata is rebuilt for live centers/widths. At most ONE boundary
    tile/frequency prefix is retained; it is shared by all queries in that group.
    This computes H and mu/sigma derivatives, not a complete training step.
    """
    bank = TileBank(x, values, cfg)
    points = cfg.tile_points
    if len(x) % points:
        raise ValueError("N must be divisible by tile_points")
    h = np.zeros((3, len(mu), len(values)))
    energies = []
    groups = {}
    edge_queries = 0
    full_queries = 0
    native_sites = 0
    coefficients_built = 0
    for j, (center, width) in enumerate(zip(mu, sigma)):
        wp, wps, terms = components(center, width, cfg)
        energies.append(norm(center, width, cfg, x))
        ids = (
            np.flatnonzero(abs(x - center) < np.sqrt(11) * width)
            if wp or wps
            else np.array([], int)
        )
        native_sites += len(ids)
        p, pm, ps = native(x[ids], center, width)
        h[:, j] += np.array(
            [
                values[:, ids] @ (wp * p),
                values[:, ids] @ (wp * pm),
                values[:, ids] @ (wp * ps + wps * p),
            ]
        )
        coefficients_built += len(terms)
        for t in terms:
            start = np.searchsorted(x, t["left"], side="left")
            end = np.searchsorted(x, t["right"], side="left")
            if end <= start:
                continue
            first, last = (start + points - 1) // points, end // points
            if first <= last:
                edges = [
                    (start, min(first * points, end)),
                    (max(last * points, start), end),
                ]
            else:
                edges = [(start, end)]
            for ki, k in enumerate(range(-cfg.modes, cfg.modes + 1)):
                a = np.array([t[key][ki] for key in ("coefficient", "center", "width")])
                if first < last:
                    v = bank.get(
                        x[0] + first * cfg.tile, x[0] + last * cfg.tile, k, t["L"]
                    )
                    h[:, j] += np.real(a[:, None] * v)
                    full_queries += 1
                for left, right in edges:
                    if right <= left:
                        continue
                    tile = left // points
                    key = (*bank.key(k, t["L"]), tile)
                    if key not in groups:
                        groups[key] = [t["omega"][ki], []]
                    groups[key][1].append(
                        (j, left - tile * points, right - tile * points, a)
                    )
                    edge_queries += 1
    scratch_peak = 0
    for (*_, tile), (omega, queries) in groups.items():
        start = tile * points
        v = (
            values[:, start : start + points]
            * np.exp(1j * omega * x[start : start + points])[None, :]
        )
        prefix = np.pad(np.cumsum(v, axis=1), ((0, 0), (1, 0)))
        # Includes cumsum's temporary during pad; not Python/RSS peak memory.
        scratch_peak = max(scratch_peak, 2 * v.nbytes + prefix.nbytes + points * 16)
        for j, lo, hi, a in queries:
            h[:, j] += np.real(a[:, None] * (prefix[:, hi] - prefix[:, lo]))
    for j, (e, em, es) in enumerate(energies):
        n = np.sqrt(max(e, 1e-24))
        base = h[0, j].copy()
        h[:, j] /= n
        if e > 1e-24:
            h[1, j] -= base * em / n**3
            h[2, j] -= base * es / n**3
    unique_coefficients = {
        id(a): a for _, queries in groups.values() for _, _, _, a in queries
    }
    return h, {
        "tile_prefix_payload": bank.payload,
        "edge_scratch_arrays_peak": scratch_peak,
        "edge_queries": edge_queries,
        "edge_groups": len(groups),
        "edge_build_eliminated_fraction": 1 - len(groups) / max(edge_queries, 1),
        "edge_input_sites": len(groups) * points,
        "native_atom_sites": native_sites,
        "full_tile_queries": full_queries,
        "frequency_count": len(bank.cache),
        "fourier_components": coefficients_built,
        "output_bytes": h.nbytes,
        "task_coefficient_payload": sum(a.nbytes for a in unique_coefficients.values()),
    }
