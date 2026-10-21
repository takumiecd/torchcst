"""Exact 1D/2D Fourier-only CPU prototype, not a runtime/GPU implementation.

Input-dependent data flow is bottom-up. Query metadata is compiled before X is
read. Parent frequency requirements are carried down as metadata, and leaf
moments are merged upward. All preparation, including extra odd frequencies,
is counted. Memory/read counters are logical models, not GPU transactions.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from benchmarks.research.polynomial_fourier_cpu import smooth
from benchmarks.research.tile_hybrid_profile_cpu import Config as CoeffConfig
from benchmarks.research.tile_hybrid_profile_cpu import coefficients


@dataclass(frozen=True)
class Config:
    modes: int = 8
    dim: int = 1
    side: int = 256
    leaf: int = 8
    spacing: float = 0.1
    min_L_points: int = 4
    overlap: float = 0.12
    chunk: int = 32

    @property
    def denominator(self):
        return 4 * self.side

    @property
    def axis(self):
        return np.arange(self.side) * self.spacing


def norm_cross(a, b, key, cfg):
    """Sampled geometry including exact discrete frequency aliases."""
    points = max(a["points"], b["points"])
    sa, sb = points // a["points"], points // b["points"]
    ca = np.zeros((len(a["coefficient"]) - 1) * sa + 1, complex)
    cb = np.zeros((len(b[key]) - 1) * sb + 1, complex)
    ca[::sa], cb[::sb] = a["coefficient"], b[key]
    c = np.convolve(ca, cb)
    m = (len(c) - 1) // 2
    lo = np.searchsorted(cfg.axis, max(a["left"], b["left"]), side="left")
    hi = np.searchsorted(cfg.axis, min(a["right"], b["right"]), side="left")
    if hi <= lo:
        return 0.0
    # Integer reduction avoids sinc(large integer)/sinc(integer) cancellation.
    k = np.arange(-m, m + 1)
    residue = (k + points) % (2 * points) - points
    theta = np.pi * residue / points
    n = hi - lo
    z = (
        n
        * np.sinc(n * theta / (2 * np.pi))
        / np.sinc(theta / (2 * np.pi))
        * np.exp(1j * theta * (lo + (n - 1) / 2))
    )
    return float(np.real(c @ z))


def axis_terms(mu, sigma, cfg):
    radius = np.sqrt(11) * sigma
    base = cfg.min_L_points * cfg.spacing
    nearest = max(0, round(np.log2(radius / (0.85 * base))))
    boundary = 0.85 * base * 2**nearest
    if boundary * (1 - cfg.overlap) <= radius <= boundary * (1 + cfg.overlap):
        z = (radius - boundary * (1 - cfg.overlap)) / (2 * cfg.overlap * boundary)
        w, dw = smooth(z)
        dw *= np.sqrt(11) / (2 * cfg.overlap * boundary)
        levels = [(nearest, 1 - w, -dw), (nearest + 1, w, dw)]
    else:
        levels = [(max(0, int(np.ceil(np.log2(radius / (0.85 * base))))), 1.0, 0.0)]
    terms = []
    ccfg = CoeffConfig(modes=cfg.modes, endpoint_zeros=2 if cfg.modes == 4 else 3)
    for level, w, ws in levels:
        if w == 0 and ws == 0:
            continue
        points = cfg.min_L_points * 2**level
        if points > 2 * cfg.side:
            raise ValueError("L exceeds declared dyadic frequency lattice")
        L = points * cfg.spacing
        c, cs = coefficients(sigma, L, ccfg)
        omega = np.pi * np.arange(-cfg.modes, cfg.modes + 1) / L
        phase = np.exp(-1j * omega * mu)
        value = w * c * phase
        terms.append(
            {
                "L": L,
                "points": points,
                "omega": omega,
                "coefficient": value,
                "center": -1j * omega * value,
                "width": (ws * c + w * cs) * phase,
                "left": mu - L,
                "right": mu + L,
            }
        )
    norms = np.zeros(3)
    for a in terms:
        for b in terms:
            for j, key in enumerate(("coefficient", "center", "width")):
                norms[j] += norm_cross(a, b, key, cfg)
    e, em, es = norms
    n = np.sqrt(max(e, 1e-24))
    for t in terms:
        c = t["coefficient"].copy()
        t["coefficient"] /= n
        t["center"] = t["center"] / n - c * em / n**3 if e > 1e-24 else t["center"] / n
        t["width"] = t["width"] / n - c * es / n**3 if e > 1e-24 else t["width"] / n
        t["lo"] = max(0, int(np.searchsorted(cfg.axis, t["left"], side="left")))
        t["hi"] = min(cfg.side, int(np.searchsorted(cfg.axis, t["right"], side="left")))
    return terms


class Geometry:
    """Fixed relative-leaf Fourier tables, shared by all input-dependent runs."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.table = {}
        self.mapping = {}
        self.axial_table = {}
        self.coords = np.indices((cfg.leaf,) * cfg.dim).reshape(cfg.dim, -1).T

    def map(self, points):
        if points in self.mapping:
            return self.mapping[points]
        m, D = self.cfg.modes, self.cfg.denominator
        raw = np.array(list(product(range(-m, m + 1), repeat=self.cfg.dim)))
        nums = raw * np.array([D // (2 * p) for p in points])
        nums %= D
        canonical = []
        conjugate = []
        for row in nums:
            key = tuple(int(v) for v in row)
            neg = tuple((-v) % D for v in key)
            canonical.append(min(key, neg))
            conjugate.append(neg < key)
        keys = sorted(set(canonical))
        lookup = {k: i for i, k in enumerate(keys)}
        inverse = np.array([lookup[k] for k in canonical])
        value = (keys, inverse, np.array(conjugate))
        self.mapping[points] = value
        return value

    def basis(self, keys, start):
        D = self.cfg.denominator
        missing = [key for key in keys if key not in self.table]
        if missing:
            q = np.array(missing, dtype=float)
            q[q > D / 2] -= D
            values = np.exp(2j * np.pi / D * (self.coords @ q.T))
            for j, key in enumerate(missing):
                self.table[key] = values[:, j].copy()
        local = np.stack([self.table[key] for key in keys], axis=1)
        q = np.array(keys, dtype=float)
        q[q > D / 2] -= D
        phase = np.exp(2j * np.pi / D * (q @ np.asarray(start)))
        return local * phase[None, :]

    @property
    def payload(self):
        return (
            sum(v.nbytes for v in self.table.values())
            + sum(inv.nbytes + conj.nbytes for _, inv, conj in self.mapping.values())
            + sum(v.nbytes for v in self.axial_table.values())
        )

    def axial_basis(self, keys, start):
        D = self.cfg.denominator
        for key in keys:
            if key not in self.axial_table:
                q = key if key <= D / 2 else key - D
                self.axial_table[key] = np.exp(
                    2j * np.pi * q / D * np.arange(self.cfg.leaf)
                )
        q = np.array(keys, dtype=float)
        q[q > D / 2] -= D
        return (
            np.stack([self.axial_table[k] for k in keys], axis=1)
            * np.exp(2j * np.pi * q * start / D)[None, :]
        )


def children(node, cfg):
    start, size = node
    half = size // 2
    return [
        (tuple(s + b * half for s, b in zip(start, bits)), half)
        for bits in product((0, 1), repeat=cfg.dim)
    ]


def covered_nodes(lo, hi, cfg, node=None):
    if node is None:
        node = ((0,) * cfg.dim, cfg.side)
    start, size = node
    if any(h <= s or l >= s + size for l, h, s in zip(lo, hi, start)):
        return []
    if all(l <= s and h >= s + size for l, h, s in zip(lo, hi, start)):
        return [(node, None)]
    if size == cfg.leaf:
        return [
            (
                node,
                (
                    tuple(max(l - s, 0) for l, s in zip(lo, start)),
                    tuple(min(h - s, size) for h, s in zip(hi, start)),
                ),
            )
        ]
    return [
        v for child in children(node, cfg) for v in covered_nodes(lo, hi, cfg, child)
    ]


def make_profiles(mu, sigma, cfg):
    return [
        [axis_terms(mu[j, d], sigma[j, d], cfg) for d in range(cfg.dim)]
        for j in range(len(mu))
    ]


def make_plan(profiles, cfg, geometry):
    tasks = {}
    active = set()
    for atom, axes in enumerate(profiles):
        for components in product(*axes):
            if any(t["hi"] <= t["lo"] for t in components):
                continue
            arrays = []
            if cfg.dim == 1:
                arrays = [components[0][k] for k in ("coefficient", "center", "width")]
            else:
                a, b = components
                arrays = [
                    np.outer(a["coefficient"], b["coefficient"]),
                    np.outer(a["center"], b["coefficient"]),
                    np.outer(a["width"], b["coefficient"]),
                    np.outer(a["coefficient"], b["center"]),
                    np.outer(a["coefficient"], b["width"]),
                ]
            arrays = np.array(arrays).reshape(1 + 2 * cfg.dim, -1)
            keys, inverse, conjugate = geometry.map(
                tuple(t["points"] for t in components)
            )
            arrays[:, conjugate] = arrays[:, conjugate].conj()
            folded = np.zeros((1 + 2 * cfg.dim, len(keys)), complex)
            for j in range(len(folded)):
                np.add.at(folded[j], inverse, arrays[j])
            for node, partial in covered_nodes(
                tuple(t["lo"] for t in components),
                tuple(t["hi"] for t in components),
                cfg,
            ):
                tasks.setdefault(node, []).append((atom, keys, folded, partial))
                start, size = node
                while True:
                    active.add((start, size))
                    if size == cfg.side:
                        break
                    size *= 2
                    start = tuple((s // size) * size for s in start)
    return tasks, active


def counters():
    return {
        k: 0
        for k in (
            "input_values",
            "weighted_products",
            "weighted_scratch_values",
            "complex_products",
            "intermediate_values",
            "basis_values",
            "prefix_values",
            "merge_adds",
            "moment_reads",
            "moment_writes",
            "coefficient_products",
            "coefficient_values",
            "query_output_updates",
            "scratch_arrays_peak",
            "carry_arrays_peak",
            "cached_bank_payload",
            "leaf_visits",
        )
    }


def apply_full(tasks, keys, moments, h, stats):
    lookup = {k: i for i, k in enumerate(keys)}
    for atom, qkeys, coef, partial in tasks:
        if partial is not None:
            continue
        index = [lookup[k] for k in qkeys]
        h[:, atom] += np.real(coef @ moments[:, index].T)
        stats["moment_reads"] += moments.shape[0] * len(index)
        stats["coefficient_products"] += coef.shape[0] * moments.shape[0] * len(index)
        stats["coefficient_values"] += coef.size
        stats["query_output_updates"] += h.shape[0] * h.shape[2]


def leaf_data(node, keys, tasks, X, cfg, geometry, h, stats, values=None):
    start, size = node
    slices = (slice(None),) + tuple(slice(s, s + size) for s in start)
    if values is None:
        values = X[slices].copy().reshape(len(X), -1)
        stats["input_values"] += values.size
        stats["leaf_visits"] += 1
    moments = np.zeros((len(X), len(keys)), complex)
    lookup = {k: i for i, k in enumerate(keys)}
    operations = {}
    for atom, qkeys, coef, partial in tasks:
        if partial is None:
            continue
        indices = np.array([lookup[k] for k in qkeys])
        for chunk in np.unique(indices // cfg.chunk):
            use = np.flatnonzero(indices // cfg.chunk == chunk)
            operations.setdefault(int(chunk), []).append(
                (atom, indices[use] - chunk * cfg.chunk, coef[:, use], partial)
            )
    for first in range(0, len(keys), cfg.chunk):
        kk = keys[first : first + cfg.chunk]
        basis = geometry.basis(kk, start)
        weighted = values[:, :, None] * basis[None, :, :]
        moments[:, first : first + len(kk)] = weighted.sum(axis=1)
        stats["weighted_products"] += weighted.size
        stats["weighted_scratch_values"] += weighted.size
        stats["basis_values"] += basis.size
        stats["moment_writes"] += len(X) * len(kk)
        scratch = values.nbytes + basis.nbytes + weighted.nbytes
        if first // cfg.chunk in operations:
            ops = operations[first // cfg.chunk]
            partial_indices = np.unique(np.concatenate([op[1] for op in ops]))
            prefix = weighted[:, :, partial_indices].reshape(
                (len(X),) + (size,) * cfg.dim + (len(partial_indices),)
            )
            for axis in range(cfg.dim):
                prefix = np.cumsum(prefix, axis=axis + 1)
            prefix = np.pad(prefix, ((0, 0),) + ((1, 0),) * cfg.dim + ((0, 0),))
            stats["prefix_values"] += (
                len(X) * size**cfg.dim * len(partial_indices) * cfg.dim
            )
            scratch += 2 * prefix.nbytes
            for atom, indices, coef, (lo, hi) in ops:
                v = np.zeros((len(X), len(partial_indices)), complex)
                for bits in product((0, 1), repeat=cfg.dim):
                    corner = tuple(
                        hi[d] if bit else lo[d] for d, bit in enumerate(bits)
                    )
                    v += (-1) ** (cfg.dim - sum(bits)) * prefix[
                        (slice(None),) + corner + (slice(None),)
                    ]
                h[:, atom] += np.real(
                    coef @ v[:, np.searchsorted(partial_indices, indices)].T
                )
                stats["moment_reads"] += (2**cfg.dim) * len(X) * len(indices)
                stats["coefficient_products"] += coef.shape[0] * len(X) * len(indices)
                stats["coefficient_values"] += coef.size
                stats["query_output_updates"] += h.shape[0] * h.shape[2]
        stats["scratch_arrays_peak"] = max(stats["scratch_arrays_peak"], scratch)
    return moments


def separable_leaf_data(node, keys, tasks, X, cfg, geometry, h, stats):
    start, size = node
    values = (
        X[(slice(None),) + tuple(slice(s, s + size) for s in start)]
        .copy()
        .reshape(len(X), -1)
    )
    stats["input_values"] += values.size
    stats["leaf_visits"] += 1
    moments = np.zeros((len(X), len(keys)), complex)
    lookup = {k: i for i, k in enumerate(keys)}
    partial_tasks = [q for q in tasks if q[3] is not None]
    partial_keys = sorted(query_keys(partial_tasks))
    if partial_keys:
        partial = leaf_data(
            node, partial_keys, partial_tasks, X, cfg, geometry, h, stats, values=values
        )
        moments[:, [lookup[k] for k in partial_keys]] = partial
    partial_set = set(partial_keys)
    full = [k for k in keys if k not in partial_set]
    if not full:
        return moments
    if cfg.dim == 1:
        for first in range(0, len(full), cfg.chunk):
            kk = full[first : first + cfg.chunk]
            basis = geometry.basis(kk, start)
            moments[:, [lookup[k] for k in kk]] = values @ basis
            stats["weighted_products"] += len(X) * size * len(kk)
            stats["basis_values"] += basis.size
            stats["moment_writes"] += len(X) * len(kk)
            stats["scratch_arrays_peak"] = max(
                stats["scratch_arrays_peak"], values.nbytes + basis.nbytes
            )
        return moments
    xs = sorted({k[0] for k in full})
    xlookup = {k: i for i, k in enumerate(xs)}
    Ex = geometry.axial_basis(xs, start[0])
    G = np.einsum("bxy,xk->bky", values.reshape(len(X), size, size), Ex, optimize=True)
    stats["weighted_products"] += len(X) * size * size * len(xs)
    stats["basis_values"] += Ex.size
    stats["intermediate_values"] += G.size
    for first in range(0, len(full), cfg.chunk):
        kk = full[first : first + cfg.chunk]
        ys = sorted({k[1] for k in kk})
        ylookup = {k: i for i, k in enumerate(ys)}
        Ey = geometry.axial_basis(ys, start[1])
        selected = G[:, [xlookup[k[0]] for k in kk], :]
        weights = Ey[:, [ylookup[k[1]] for k in kk]].T
        moments[:, [lookup[k] for k in kk]] = np.einsum("bky,ky->bk", selected, weights)
        stats["complex_products"] += len(X) * size * len(kk)
        stats["intermediate_values"] += 3 * selected.size
        stats["basis_values"] += Ey.size + weights.size
        stats["moment_writes"] += len(X) * len(kk)
        stats["scratch_arrays_peak"] = max(
            stats["scratch_arrays_peak"],
            values.nbytes
            + Ex.nbytes
            + G.nbytes
            + selected.nbytes
            + Ey.nbytes
            + weights.nbytes,
        )
    return moments


def query_keys(tasks):
    return {k for _, keys, _, _ in tasks for k in keys}


def aggregate(X, profiles, cfg, geometry, method):
    tasks, active = make_plan(profiles, cfg, geometry)
    h = np.zeros((1 + 2 * cfg.dim, len(profiles), len(X)))
    stats = counters()
    if method.startswith("bottom_up"):

        def visit(node, parent_keys, retained):
            own = tasks.get(node, [])
            keys = sorted(set(parent_keys) | query_keys(own))
            if not keys and node not in active:
                return None, keys
            if node[1] == cfg.leaf:
                function = (
                    separable_leaf_data
                    if method == "bottom_up_separable"
                    else leaf_data
                )
                moments = function(node, keys, own, X, cfg, geometry, h, stats)
                stats["carry_arrays_peak"] = max(
                    stats["carry_arrays_peak"], retained + moments.nbytes
                )
            else:
                moments = np.zeros((len(X), len(keys)), complex)
                for child in children(node, cfg):
                    child_data, child_keys = visit(
                        child, keys, retained + moments.nbytes
                    )
                    if child_data is not None and keys:
                        lookup = {k: i for i, k in enumerate(child_keys)}
                        index = [lookup[k] for k in keys]
                        moments += child_data[:, index]
                        stats["merge_adds"] += moments.size
                        stats["moment_reads"] += 2 * moments.size
                        stats["moment_writes"] += moments.size
                    del child_data
            apply_full(own, keys, moments, h, stats)
            stats["cached_bank_payload"] += len(X) * len(query_keys(own)) * 16
            return moments, keys

        visit(((0,) * cfg.dim, cfg.side), [], 0)
    else:
        for node, own in tasks.items():
            keys = sorted(query_keys(own))
            if node[1] == cfg.leaf:
                moments = leaf_data(node, keys, own, X, cfg, geometry, h, stats)
            else:
                moments = np.zeros((len(X), len(keys)), complex)
                starts = [range(s, s + node[1], cfg.leaf) for s in node[0]]
                for start in product(*starts):
                    v = leaf_data(
                        (start, cfg.leaf), keys, [], X, cfg, geometry, h, stats
                    )
                    moments += v
                    stats["merge_adds"] += moments.size
                    stats["moment_reads"] += 2 * moments.size
                    stats["moment_writes"] += moments.size
            stats["carry_arrays_peak"] = max(stats["carry_arrays_peak"], moments.nbytes)
            stats["cached_bank_payload"] += moments.nbytes
            apply_full(own, keys, moments, h, stats)
    stats["query_nodes"] = len(tasks)
    stats["geometry_payload"] = geometry.payload
    # Counts the main arrays/operations, not cache misses or GPU traffic.
    stats["traffic_model_bytes"] = int(
        8 * stats["input_values"]
        + 32 * stats["weighted_scratch_values"]
        + 16 * stats["intermediate_values"]
        + 16 * stats["basis_values"]
        + 32 * stats["prefix_values"]
        + 16 * (stats["moment_reads"] + stats["moment_writes"])
        + 16 * stats["coefficient_values"]
        + 16 * stats["query_output_updates"]
    )
    return h, stats


def direct(X, profiles, cfg):
    h = np.zeros((1 + 2 * cfg.dim, len(profiles), len(X)))
    reads = 0
    for j, axes in enumerate(profiles):
        bounds = [(min(t["lo"] for t in ts), max(t["hi"] for t in ts)) for ts in axes]
        vectors = []
        for ts, (lo, hi) in zip(axes, bounds):
            values = np.zeros((3, hi - lo))
            for t in ts:
                use = (np.arange(lo, hi) >= t["lo"]) & (np.arange(lo, hi) < t["hi"])
                basis = np.exp(1j * np.outer(cfg.axis[lo:hi][use], t["omega"]))
                for dest, key in zip(values, ("coefficient", "center", "width")):
                    dest[use] += np.real(basis @ t[key])
            vectors.append(values)
        values = X[(slice(None),) + tuple(slice(lo, hi) for lo, hi in bounds)]
        reads += values.size
        if cfg.dim == 1:
            h[:, j] = vectors[0] @ values.T
        else:
            a, b = vectors
            for dest, (u, v) in enumerate(
                ((a[0], b[0]), (a[1], b[0]), (a[2], b[0]), (a[0], b[1]), (a[0], b[2]))
            ):
                h[dest, j] = np.einsum("i,bij,j->b", u, values, v)
    return h, {"input_values": reads}


def run(X, mu, sigma, cfg, geometry, method):
    profiles = make_profiles(mu, sigma, cfg)
    return (
        direct(X, profiles, cfg)
        if method == "direct"
        else aggregate(X, profiles, cfg, geometry, method)
    )
