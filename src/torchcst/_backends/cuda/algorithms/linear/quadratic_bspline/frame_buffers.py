"""Aligned views with the same immutable forward/backward lifetime."""

import math

import torch


def make_frame(source, atoms, *, shards, endpoints, active):
    entries = [
        ("meta", (atoms, 12), torch.float64, 8),
        ("order", (atoms,), torch.int64, 8),
        ("pos", (2, 6, atoms), torch.int32, 4),
        ("co", (2, 3, 6, atoms), torch.float32, 4),
        ("half", (2, atoms), torch.float32, 4),
        ("bp", (2, 2, atoms) if shards else (0,), torch.int32, 4),
        ("injection", (2, atoms) if endpoints else (0,), torch.int16, 2),
        ("active", (2,) if active else (0,), torch.int32, 4),
    ]
    layout = []
    offset = 0
    for name, shape, dtype, itemsize in entries:
        offset = (offset + 255) // 256 * 256
        size = math.prod(shape) * itemsize
        layout.append((name, shape, dtype, offset, size))
        offset += size
    storage = source.new_empty((offset,), dtype=torch.uint8)
    return {
        name: storage.narrow(0, begin, size).view(dtype).view(shape)
        for name, shape, dtype, begin, size in layout
    }
