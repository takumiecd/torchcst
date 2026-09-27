"""Prepared-only group culling experiment for the 5% mapped GEMM.

The original trainable atom table is untouched. This module permutes an already
prepared forward table within each I/B bucket; it does not provide backward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

from prototypes.local_atom_kernels import _bucket
from torchcst.nn._backends._triton_kernels import _profile


@dataclass(frozen=True)
class GroupedPrepared:
    packed: torch.Tensor
    circle: torch.Tensor
    section: torch.Tensor
    group_offsets: torch.Tensor
    group_starts: torch.Tensor
    group_counts: torch.Tensor
    group_bounds: torch.Tensor
    site_bounds: torch.Tensor
    groups: int
    build_seconds: float


def _outward(values: np.ndarray, direction: float, steps: int = 8) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    for _ in range(steps):
        result = np.nextafter(result, np.float32(direction))
    return result


def _nearby_order(bucket: np.ndarray, width: int) -> list[np.ndarray]:
    centers = bucket[:, 2:6].astype(np.float64)
    precision = bucket[:, 1].astype(np.float64)
    if not np.isfinite(centers).all() or not np.isfinite(precision).all():
        return [
            np.arange(start, min(start + width, len(bucket)))
            for start in range(0, len(bucket), width)
        ]
    if np.any(precision <= 0):
        return [
            np.arange(start, min(start + width, len(bucket)))
            for start in range(0, len(bucket), width)
        ]
    radius = 1.0 / np.sqrt(precision)
    available = np.ones(len(bucket), dtype=np.bool_)
    groups = []
    while available.any():
        seed = int(np.flatnonzero(available)[0])
        candidates = np.flatnonzero(available)
        distance = np.sum((centers[candidates] - centers[seed]) ** 2, axis=1)
        distance += (radius[candidates] - radius[seed]) ** 2
        selected = candidates[np.argsort(distance, kind="stable")[:width]]
        available[selected] = False
        groups.append(selected)
    return groups


def build_grouped(
    prepared, *, station_rows: int = 64, columns: int = 64, group_width: int = 8
):
    """Offline ideal-neighbor grouping, intentionally outside timed forward.

    Group and site boxes are expanded outward before use. Nonfinite or invalid
    group precision gets an infinite radius so it is never culled.
    """
    import time

    if (station_rows, columns, group_width) != (64, 64, 8):
        raise ValueError(
            "the experimental kernel requires 64x64 stations and groups of 8"
        )
    started = time.perf_counter()
    packed, circle, section, offsets = prepared
    if packed.ndim != 2 or packed.shape[1] != 6:
        raise ValueError("packed atoms must have six FP32 columns")
    device = packed.device
    source = packed.detach().cpu().numpy()
    old_offsets = offsets.detach().cpu().numpy().astype(np.int64)
    if old_offsets[0] != 0 or old_offsets[-1] != len(source):
        raise ValueError("invalid atom bucket offsets")
    output = np.empty_like(source)
    group_offsets = np.zeros(len(old_offsets), dtype=np.int32)
    starts: list[int] = []
    counts: list[int] = []
    bounds: list[np.ndarray] = []
    for bucket in range(len(old_offsets) - 1):
        begin, end = int(old_offsets[bucket]), int(old_offsets[bucket + 1])
        members = source[begin:end]
        write = begin
        for indices in _nearby_order(members, group_width):
            chunk = members[indices]
            count = len(chunk)
            output[write : write + count] = chunk
            centers = chunk[:, 2:6]
            lower = _outward(centers.min(axis=0), -math.inf)
            upper = _outward(centers.max(axis=0), math.inf)
            precision = chunk[:, 1].astype(np.float64)
            if (
                np.isfinite(centers).all()
                and np.isfinite(precision).all()
                and np.all(precision > 0)
            ):
                radius = np.max(1.0 / np.sqrt(precision))
                radius = float(_outward(np.asarray(radius), math.inf))
                radius2 = np.float32((radius + 1e-4) ** 2)
            else:
                radius2 = np.float32(math.inf)
            starts.append(write)
            counts.append(count)
            bounds.append(
                np.concatenate((lower, upper, np.asarray([radius2], dtype=np.float32)))
            )
            write += count
        if write != end:
            raise AssertionError("bucket grouping did not preserve all atoms")
        group_offsets[bucket + 1] = len(starts)

    circle_np = circle.detach().cpu().numpy()
    section_np = section.detach().cpu().numpy()
    stations = (len(old_offsets) - 2) // 2
    if circle_np.shape != (stations * station_rows, 2) or section_np.shape != (
        columns,
        3,
    ):
        raise ValueError("unexpected geometry dimensions")
    site_bounds = np.empty((stations * 8, 8), dtype=np.float32)
    for station in range(stations):
        for row_tile in range(4):
            direction = circle_np[
                station * station_rows + row_tile * 16 : station * station_rows
                + (row_tile + 1) * 16
            ]
            for col_tile in range(2):
                sec = section_np[col_tile * 32 : (col_tile + 1) * 32]
                xy = direction[:, None, :] * sec[None, :, :1]
                xy = xy.reshape(-1, 2)
                sites = np.concatenate(
                    (xy, np.repeat(sec[None, :, 1:], 16, axis=0).reshape(-1, 2)), axis=1
                )
                low = _outward(sites.min(axis=0), -math.inf)
                high = _outward(sites.max(axis=0), math.inf)
                site_bounds[station * 8 + row_tile * 2 + col_tile] = np.concatenate(
                    (low, high)
                )

    def on_device(array, dtype=None):
        return torch.as_tensor(array, device=device, dtype=dtype).contiguous()

    return GroupedPrepared(
        packed=on_device(output),
        circle=circle,
        section=section,
        group_offsets=on_device(group_offsets),
        group_starts=on_device(np.asarray(starts, dtype=np.int32)),
        group_counts=on_device(np.asarray(counts, dtype=np.int32)),
        group_bounds=on_device(np.asarray(bounds, dtype=np.float32).reshape(-1, 9)),
        site_bounds=on_device(site_bounds),
        groups=len(starts),
        build_seconds=time.perf_counter() - started,
    )


@tr.jit
def _weight_grouped(
    P,
    Circle,
    Section,
    GroupOffsets,
    GroupStarts,
    GroupCounts,
    GroupBounds,
    SiteBounds,
    station,
    row_start,
    col_start,
    G: tl.constexpr,
    PROFILE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = row_start + tl.arange(0, BN)
    cols = col_start + tl.arange(0, BK)
    cosine = tl.load(Circle + rows * 2)
    sine = tl.load(Circle + rows * 2 + 1)
    rho = tl.load(Section + cols * 3)
    z = tl.load(Section + cols * 3 + 1)
    w = tl.load(Section + cols * 3 + 2)
    sx = cosine[:, None] * rho[None, :]
    sy = sine[:, None] * rho[None, :]
    tile = station * 8 + ((row_start - station * 64) // 16) * 2 + col_start // 32
    lx = tl.load(SiteBounds + tile * 8)
    ly = tl.load(SiteBounds + tile * 8 + 1)
    lz = tl.load(SiteBounds + tile * 8 + 2)
    lw = tl.load(SiteBounds + tile * 8 + 3)
    hx = tl.load(SiteBounds + tile * 8 + 4)
    hy = tl.load(SiteBounds + tile * 8 + 5)
    hz = tl.load(SiteBounds + tile * 8 + 6)
    hw = tl.load(SiteBounds + tile * 8 + 7)
    partial = tl.full((BN, BK), 0.0, tl.float32)
    for slot in tl.static_range(1 if G == 1 else 3):
        bucket = _bucket(station, slot, G)
        first = tl.load(GroupOffsets + bucket)
        last = tl.load(GroupOffsets + bucket + 1)
        for group in range(first, last):
            blx = tl.load(GroupBounds + group * 9)
            bly = tl.load(GroupBounds + group * 9 + 1)
            blz = tl.load(GroupBounds + group * 9 + 2)
            blw = tl.load(GroupBounds + group * 9 + 3)
            bhx = tl.load(GroupBounds + group * 9 + 4)
            bhy = tl.load(GroupBounds + group * 9 + 5)
            bhz = tl.load(GroupBounds + group * 9 + 6)
            bhw = tl.load(GroupBounds + group * 9 + 7)
            radius2 = tl.load(GroupBounds + group * 9 + 8)
            gx = tl.maximum(tl.maximum(lx - bhx, blx - hx), 0.0)
            gy = tl.maximum(tl.maximum(ly - bhy, bly - hy), 0.0)
            gz = tl.maximum(tl.maximum(lz - bhz, blz - hz), 0.0)
            gw = tl.maximum(tl.maximum(lw - bhw, blw - hw), 0.0)
            lower = (gx * gx + gy * gy) + (gz * gz + gw * gw)
            if (lower <= radius2) | (radius2 == float("inf")) | libdevice.isnan(lower):
                first_atom = tl.load(GroupStarts + group)
                count = tl.load(GroupCounts + group)
                for atom in range(first_atom, first_atom + count):
                    cx = tl.load(P + atom * 6 + 2)
                    cy = tl.load(P + atom * 6 + 3)
                    cz = tl.load(P + atom * 6 + 4)
                    cw = tl.load(P + atom * 6 + 5)
                    dx = sx - cx
                    dy = sy - cy
                    dz = z - cz
                    dw = w - cw
                    squared = (dx * dx + dy * dy) + (dz * dz + dw * dw)[None, :]
                    precision = tl.load(P + atom * 6 + 1)
                    value, _ = _profile(squared, precision, PROFILE)
                    amplitude = tl.load(P + atom * 6)
                    partial += value * amplitude
    return partial


@tr.jit
def block_grouped_fused(
    X,
    P,
    Circle,
    Section,
    GroupOffsets,
    GroupStarts,
    GroupCounts,
    GroupBounds,
    SiteBounds,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    T: tl.constexpr,
    CG: tl.constexpr,
    G: tl.constexpr,
    PROFILE: tl.constexpr,
    BM: tl.constexpr = 128,
    BN: tl.constexpr = 16,
    BK: tl.constexpr = 32,
    SPLIT_K: tl.constexpr = 8,
):
    tl.static_assert(S == 64)
    tl.static_assert(T == 64)
    tl.static_assert(BN == 16)
    tl.static_assert(BK == 32)
    tile = tl.program_id(1)
    row_group = tile // 4
    local = (tile % 4) * 16
    n = row_group * S + local + tl.arange(0, BN)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    split = tl.program_id(2)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for c in range(split, CG, SPLIT_K):
        station = row_group * CG + c
        for col_start in range(0, T, BK):
            k = col_start + tl.arange(0, BK)
            x = tl.load(
                X + m[:, None] * K + (c * T + k)[None, :],
                (m[:, None] < M) & (c * T + k[None, :] < K),
                0.0,
            )
            weight = _weight_grouped(
                P,
                Circle,
                Section,
                GroupOffsets,
                GroupStarts,
                GroupCounts,
                GroupBounds,
                SiteBounds,
                station,
                station * S + local,
                col_start,
                G,
                PROFILE,
                BN,
                BK,
            )
            acc = tl.dot(x, tl.trans(weight), acc, input_precision="ieee")
    tl.store(
        Y + split * M * N + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )
