"""Conservative fixed-site boxes before exact neighbor support classification."""

import torch
import triton as tr
import triton.language as tl


@torch.no_grad()
def station_site_boxes(circle, section, station_rows):
    """Outward-padded AABBs for all sampled 4D sites in each station."""
    assert circle.shape[0] % station_rows == 0 and section.shape[1] == 3
    directions = circle.reshape(-1, station_rows, 2)
    cmin, cmax = directions.amin(dim=1), directions.amax(dim=1)
    rmin, rmax = section[:, 0].amin(), section[:, 0].amax()
    products = torch.stack((cmin * rmin, cmin * rmax, cmax * rmin, cmax * rmax))
    xymin, xymax = products.amin(dim=0), products.amax(dim=0)
    zwmin = section[:, 1:].amin(dim=0).expand(xymin.shape[0], -1)
    zwmax = section[:, 1:].amax(dim=0).expand(xymax.shape[0], -1)
    lower, upper = torch.cat((xymin, zwmin), 1), torch.cat((xymax, zwmax), 1)
    magnitude = torch.maximum(lower.abs(), upper.abs())
    margin = magnitude * (32 * torch.finfo(circle.dtype).eps) + 1e-4
    return torch.cat((lower - margin, upper + margin), 1).contiguous()


@tr.jit
def support_buckets_batched_box(
    Centers,
    Precision,
    Owners,
    Circle,
    Section,
    Boxes,
    Keys,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    STATION_ROWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    A: tl.constexpr,
    BA: tl.constexpr,
):
    tl.static_assert(D == 4)
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    atom_valid = a < A
    owner = tl.load(Owners + a, atom_valid, 0)
    cx = tl.load(Centers + a * CS0, atom_valid, 0.0)
    cy = tl.load(Centers + a * CS0 + CS1, atom_valid, 0.0)
    cz = tl.load(Centers + a * CS0 + 2 * CS1, atom_valid, 0.0)
    cw = tl.load(Centers + a * CS0 + 3 * CS1, atom_valid, 0.0)
    r = tl.sqrt(cx * cx + cy * cy)
    ux, uy = cx / r, cy / r
    precision = tl.load(Precision + a, atom_valid, 0.0)
    radius2 = tl.where(precision > 0.0, 1.0 / precision, float("inf"))
    count = tl.full((BA,), 0, tl.int32)
    first = tl.full((BA,), 0, tl.int32)
    second = tl.full((BA,), 0, tl.int32)
    for shift in tl.static_range(3 if G >= 3 else G):  # noqa: FURB136
        station = (owner + G - 1 + shift) % G
        lx = tl.load(Boxes + station * 8)
        ly = tl.load(Boxes + station * 8 + 1)
        lz = tl.load(Boxes + station * 8 + 2)
        lw = tl.load(Boxes + station * 8 + 3)
        hx = tl.load(Boxes + station * 8 + 4)
        hy = tl.load(Boxes + station * 8 + 5)
        hz = tl.load(Boxes + station * 8 + 6)
        hw = tl.load(Boxes + station * 8 + 7)
        gx = tl.maximum(tl.maximum(lx - cx, cx - hx), 0.0)
        gy = tl.maximum(tl.maximum(ly - cy, cy - hy), 0.0)
        gz = tl.maximum(tl.maximum(lz - cz, cz - hz), 0.0)
        gw = tl.maximum(tl.maximum(lw - cw, cw - hw), 0.0)
        lower = (gx * gx + gy * gy) + (gz * gz + gw * gw)
        possible = atom_valid & (
            (lower * precision <= 1.0001) | (radius2 == float("inf"))
        )
        hit = tl.full((BA,), False, tl.int1)
        if tl.sum(possible.to(tl.int32), 0) > 0:
            row = station[:, None] * STATION_ROWS + tl.arange(0, ROWS)[None, :]
            valid = (
                (row < N)
                & (tl.arange(0, ROWS)[None, :] < STATION_ROWS)
                & possible[:, None]
            )
            cosine = tl.load(Circle + row * 2, valid, 0.0)
            sine = tl.load(Circle + row * 2 + 1, valid, 0.0)
            distance = (cosine - ux[:, None]) * (cosine - ux[:, None]) + (
                sine - uy[:, None]
            ) * (sine - uy[:, None])
            nearest = tl.argmin(tl.where(valid, distance, float("inf")), axis=1)
            chosen = station * STATION_ROWS + nearest
            cosine = tl.load(Circle + chosen * 2, possible, 0.0)
            sine = tl.load(Circle + chosen * 2 + 1, possible, 0.0)
            for start in range(tr.cdiv(K, COLS)):
                k = start * COLS + tl.arange(0, COLS)
                rho = tl.load(Section + k * 3, k < K, 0.0)
                sx = cosine[:, None] * rho[None, :]
                sy = sine[:, None] * rho[None, :]
                dx = sx - cx[:, None]
                dy = sy - cy[:, None]
                z = tl.load(Section + k * 3 + 1, k < K, 0.0)
                w = tl.load(Section + k * 3 + 2, k < K, 0.0)
                dz = z[None, :] - cz[:, None]
                dw = w[None, :] - cw[:, None]
                squared = (dx * dx + dy * dy) + (dz * dz + dw * dw)
                hit |= (
                    tl.sum(
                        ((k[None, :] < K) & (squared * precision[:, None] < 1.0)).to(
                            tl.int32
                        ),
                        axis=1,
                    )
                    > 0
                ) & possible
        second = tl.where(hit & (count == 1), station, second)
        first = tl.where(hit & (count == 0), station, first)
        count += hit.to(tl.int32)
    if G == 2:
        boundary = 0
    else:
        boundary = tl.where((first + 1) % G == second, first, second)
    key = tl.where(count == 0, 2 * G, tl.where(count == 1, 2 * first, 2 * boundary + 1))
    tl.store(Keys + a, key, atom_valid)
