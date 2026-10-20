"""Aligned views with the same immutable forward/backward lifetime."""

import math

import torch


def make_frame(
    source,
    atoms,
    *,
    shards,
    endpoints,
    active,
    numeric_elements=0,
    compact_positions=False,
    coefficient_float64=False,
    compact_c2=False,
):
    position_dtype = torch.int16 if compact_positions else torch.int32
    position_bytes = 2 if compact_positions else 4
    coefficient_dtype = torch.float64 if coefficient_float64 else torch.float32
    coefficient_bytes = 8 if coefficient_float64 else 4
    if compact_c2 and not coefficient_float64:
        raise ValueError("shared quadratic coefficients require FP64 storage")
    entries = [
        ("meta", (atoms, 12), torch.float64, 8),
        ("order", (atoms,), torch.int64, 8),
        ("pos", (2, 6, atoms), position_dtype, position_bytes),
        (
            "co",
            (2, 13, atoms) if compact_c2 else (2, 3, 6, atoms),
            coefficient_dtype,
            coefficient_bytes,
        ),
        ("half", (2, atoms), coefficient_dtype, coefficient_bytes),
        ("bp", (2, 2, atoms) if shards else (0,), position_dtype, position_bytes),
        ("injection", (2, atoms) if endpoints else (0,), torch.int16, 2),
        ("active", (2,) if active else (0,), torch.int32, 4),
    ]
    if numeric_elements:
        entries.append(("numeric", (numeric_elements,), torch.float32, 4))
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


def numeric_layout(*, input_points, output_points, blocks, batch, shards):
    shapes = {
        "input": (3, input_points, batch),
        "adjoint": (3, output_points, batch),
        "delta": (3, input_points, batch),
        "border": (3, blocks, shards, batch),
    }
    offsets = {}
    offset = 0
    for name, shape in shapes.items():
        offset = (offset + 63) // 64 * 64
        size = math.prod(shape)
        offsets[name] = offset, size
        offset += size
    return shapes, offsets, offset


def numeric_workspace(
    source, *, input_points, output_points, blocks, batch, shards, storage=None
):
    """Keep input values immutable; reuse forward delta as the adjoint bank."""
    shapes, offsets, offset = numeric_layout(
        input_points=input_points,
        output_points=output_points,
        blocks=blocks,
        batch=batch,
        shards=shards,
    )
    if storage is None:
        storage = source.new_empty((offset,), dtype=torch.float32)
    elif storage.dtype != torch.float32 or storage.numel() != offset:
        raise ValueError("numeric workspace layout mismatch")
    views = {
        name: storage.narrow(0, offsets[name][0], offsets[name][1]).view(shape)
        for name, shape in shapes.items()
    }
    return storage, views
