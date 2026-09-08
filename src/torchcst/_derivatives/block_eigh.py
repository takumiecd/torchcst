"""Small per-atom eigensystems; no full Gram or CUDA host info reads."""

from functools import cache

import torch

from torchcst._runtime.graphs import CapturedCall

from ._jacobi import _schedule


def rotate(matrix, vectors, partner):
    n = matrix.shape[-1]
    index = torch.arange(n, device=matrix.device)
    first = index < partner
    diagonal = matrix.diagonal(dim1=-2, dim2=-1)
    delta = diagonal.index_select(-1, partner) - diagonal
    delta = torch.where(first, delta, -delta)
    entry = torch.minimum(index, partner) * n + torch.maximum(index, partner)
    off = matrix.flatten(-2).index_select(-1, entry)
    norm = torch.sqrt(delta.square() + 4 * off.square())
    denominator = delta + torch.copysign(norm, delta)
    t = torch.where(
        off != 0, 2 * off / torch.where(denominator != 0, denominator, 1), 0
    )
    c = torch.rsqrt(1 + t.square())
    s = torch.where(first, -t * c, t * c)
    columns = matrix * c[:, None, :] + matrix.index_select(-1, partner) * s[:, None, :]
    current = (
        columns * c[:, :, None] + columns.index_select(-2, partner) * s[:, :, None]
    )
    current = torch.where(index[None, :] == partner[:, None], 0, current)
    basis = vectors * c[:, None, :] + vectors.index_select(-1, partner) * s[:, None, :]
    return current, basis


@cache
def _rotation():
    return torch.compile(rotate, fullgraph=True, dynamic=True)


def eigensystems(blocks, schedule):
    n = blocks.shape[-1]
    scale = blocks.abs().amax(dim=(-2, -1)).clamp_min(torch.finfo(blocks.dtype).tiny)
    current = 0.5 * (blocks + blocks.transpose(-1, -2)) / scale[:, None, None]
    vectors = (
        torch.eye(n, dtype=blocks.dtype, device=blocks.device).expand_as(blocks).clone()
    )
    for _ in range(12):
        for row in range(n - 1):
            current, vectors = _rotation()(current, vectors, schedule[row])
    # Ordering is irrelevant to the global secular equation. Avoid advanced
    # per-batch sorting/indexing and its host checks during graph capture.
    return current.diagonal(dim1=-2, dim2=-1) * scale[:, None], vectors


@cache
def _runner():
    return CapturedCall(eigensystems)


def block_eigh(blocks, *, capture=True):
    if not blocks.is_cuda:
        return torch.linalg.eigh(blocks)
    q = blocks.shape[-1]
    if q % 2 or q < 2:
        blocks = torch.nn.functional.pad(blocks, (0, 1, 0, 1))
    schedule = _schedule(blocks.shape[-1], blocks.device)
    # Callers capturing a larger solve must not depend on another graph's event.
    if not capture:
        return eigensystems(blocks, schedule)
    return _runner()(blocks, schedule)
