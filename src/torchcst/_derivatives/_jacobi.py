"""Fixed-sweep symmetric Jacobi eigensolver, with no data-dependent host reads."""

from functools import cache

import torch

from torchcst._runtime.graphs import CapturedCall


def pair_schedule(size):
    if size < 2 or size % 2:
        raise ValueError("Jacobi size must be positive and even (at least two)")
    ring = list(range(size))
    rows = []
    for _ in range(size - 1):
        partner = [0] * size
        for p, q in zip(ring[: size // 2], reversed(ring[size // 2 :])):
            partner[p], partner[q] = q, p
        rows.append(partner)
        ring = [ring[0], ring[-1], *ring[1:-1]]
    return rows


def jacobi_round(matrix, vectors, partner):
    n = matrix.shape[0]
    index = torch.arange(n, device=matrix.device)
    first = index < partner
    diagonal = matrix.diagonal()
    delta = torch.where(
        first, diagonal[partner] - diagonal, diagonal - diagonal[partner]
    )
    off = matrix[torch.minimum(index, partner), torch.maximum(index, partner)]
    # Stable tangent formula, also for equal diagonal entries and zero off-diagonal.
    norm = torch.sqrt(delta.square() + 4 * off.square())
    denominator = delta + torch.copysign(norm, delta)
    t = torch.where(
        off != 0, 2 * off / torch.where(denominator != 0, denominator, 1), 0
    )
    c = torch.rsqrt(1 + t.square())
    s = torch.where(first, -t * c, t * c)
    columns = matrix * c[None, :] + matrix[:, partner] * s[None, :]
    result = columns * c[:, None] + columns[partner, :] * s[:, None]
    result = torch.where(index[None, :] == partner[:, None], 0, result)
    basis = vectors * c[None, :] + vectors[:, partner] * s[None, :]
    return result, basis


@cache
def _round_compiled():
    return torch.compile(jacobi_round, fullgraph=True, dynamic=False)


def jacobi_eigh(matrix, schedule, sweeps=12, *, compiled=False):
    size = matrix.shape[0]
    scale = matrix.abs().amax().clamp_min(torch.finfo(matrix.dtype).tiny)
    current = 0.5 * (matrix + matrix.T) / scale
    vectors = torch.eye(size, device=matrix.device, dtype=matrix.dtype)
    step = _round_compiled() if compiled else jacobi_round
    for _ in range(sweeps):
        for row in range(size - 1):
            current, vectors = step(current, vectors, schedule[row])
    values = current.diagonal() * scale
    indices = torch.argsort(values)
    return values[indices], vectors[:, indices]


@cache
def _runner(sweeps):
    return CapturedCall(
        lambda a, schedule: jacobi_eigh(a, schedule, sweeps, compiled=True)
    )


@cache
def _schedule(size, device):
    return torch.tensor(pair_schedule(size), device=device)


def device_eigh(matrix, *, sweeps=12):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    size = matrix.shape[0]
    # Pad odd sizes with a zero eigenvalue; callers can use the padded basis.
    if size % 2 or size < 2:
        raise ValueError("device Jacobi currently requires an even size >= 2")
    schedule = _schedule(size, matrix.device)
    if matrix.device.type == "cuda":
        return _runner(sweeps)(matrix, schedule)
    return jacobi_eigh(matrix, schedule, sweeps)


def device_pinv_solve(matrix, rhs, *, rtol=None, sweeps=12):
    original_size = matrix.shape[0]
    if original_size % 2 or original_size < 2:
        matrix = torch.nn.functional.pad(matrix, (0, 1, 0, 1))
        rhs = torch.nn.functional.pad(rhs, (0, 1))
    values, vectors = device_eigh(matrix, sweeps=sweeps)
    cutoff = original_size * torch.finfo(matrix.dtype).eps if rtol is None else rtol
    keep = values.abs() > cutoff * values.abs().amax()
    inverse = torch.where(keep, 1 / torch.where(keep, values, 1), 0)
    solution = vectors @ ((vectors.T @ rhs) * inverse)
    # A posteriori diagonalization check stays on device; no silent host fallback.
    residual = (matrix @ vectors - vectors * values).norm()
    tolerance = 64 * original_size * torch.finfo(matrix.dtype).eps
    valid = torch.isfinite(solution).all() & (
        residual <= tolerance * matrix.norm().clamp_min(1e-30)
    )
    return solution[:original_size], valid
