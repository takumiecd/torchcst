"""Per-forward input-event membership; no parameters or optimizer rows move."""

from dataclasses import dataclass

import torch


def layout(atoms, sites, minimum=2):
    if type(atoms) is not int or not 0 <= atoms < 2**30:
        raise ValueError("atoms must be a nonnegative int below 2**30")
    if type(sites) is not int or sites < 8 or sites & (sites - 1):
        raise ValueError("sites must be a power of two >=8")
    if type(minimum) is not int or not 1 <= minimum < sites.bit_length():
        raise ValueError("invalid minimum block log")
    points = (sites.bit_length() - minimum) * sites + 2 * (sites >> minimum) - 1
    bits = (points - 1).bit_length()
    if bits > 16:
        raise ValueError("four exact event keys require at most 16 bits each")
    capacity = 1 << max(8, (max(2 * atoms, 1) - 1).bit_length())
    return capacity, bits


@dataclass(frozen=True)
class InputGroups:
    counts: torch.Tensor
    ends: torch.Tensor
    ids: torch.Tensor
    chunk_ends: torch.Tensor
    mapping: torch.Tensor
    chunk_atoms: int


def prepare(pos, count, atoms, sites, chunk_atoms=4, minimum=2):
    import triton as tr

    from . import input_group_kernels as k

    capacity, bits = layout(atoms, sites, minimum)
    if type(chunk_atoms) is not int or chunk_atoms not in (2, 4, 8, 16):
        raise ValueError("chunk_atoms must be 2, 4, 8 or 16")
    if pos.shape != (2, 6, atoms) or pos.dtype != torch.int32 or not pos.is_cuda:
        raise ValueError("pos must be contiguous CUDA int32 [2,6,A]")
    if not pos.is_contiguous() or count.device != pos.device or count.numel() != 1:
        raise ValueError("invalid position or count metadata")
    if count.dtype != torch.int32:
        raise ValueError("count must be int32")

    def empty(n, dtype=torch.int32):
        return pos.new_empty(n, dtype=dtype)

    keys = empty(capacity, torch.int64).fill_(-1)
    hist, cursor = empty(capacity).zero_(), empty(capacity).zero_()
    ends, chunks, chunk_ends = empty(capacity), empty(capacity), empty(capacity)
    bucket, ids, mapping = empty(atoms), empty(atoms), empty(atoms)
    if atoms:
        k.members[(tr.cdiv(atoms, 128),)](
            pos, count, keys, hist, bucket, atoms, capacity, bits
        )
    torch.cumsum(hist, 0, dtype=torch.int32, out=ends)
    if atoms:
        k.place[(tr.cdiv(atoms, 128),)](bucket, hist, ends, cursor, ids, count, atoms)
    k.chunk_counts[(tr.cdiv(capacity, 256),)](hist, chunks, capacity, chunk_atoms)
    torch.cumsum(chunks, 0, dtype=torch.int32, out=chunk_ends)
    if atoms:
        k.chunk_map[(tr.cdiv(atoms, 128),)](
            chunk_ends, mapping, atoms, capacity, capacity.bit_length()
        )
    return InputGroups(hist, ends, ids, chunk_ends, mapping, chunk_atoms)
