"""Conservative fixed-site boxes before exact neighbor support classification."""

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _decode_intrinsic_torus(
    Center,
    Major,
    Minor,
    Decoded,
    A: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    B: tl.constexpr,
):
    atom = tl.program_id(0) * B + tl.arange(0, B)
    valid = atom < A
    arc = tl.load(Center + atom * CS0, valid, 0.0)
    u = tl.load(Center + atom * CS0 + CS1, valid, 0.0)
    v = tl.load(Center + atom * CS0 + 2 * CS1, valid, 0.0)
    major = tl.load(Major)
    minor = tl.load(Minor)
    angle = tl.div_rn(tl.sqrt(u * u + v * v), minor)
    sinc = tl.where(angle == 0.0, 1.0, libdevice.sin(angle) / angle)
    radial = major + minor * libdevice.cos(angle)
    theta = tl.div_rn(arc, major)
    tl.store(Decoded + atom * 4, radial * libdevice.cos(theta), valid)
    tl.store(Decoded + atom * 4 + 1, radial * libdevice.sin(theta), valid)
    tl.store(Decoded + atom * 4 + 2, sinc * u, valid)
    tl.store(Decoded + atom * 4 + 3, sinc * v, valid)


@torch.no_grad()
def decode_intrinsic_torus_compact(geometry, center):
    """Forward-only fused center decode without intermediate A-sized tensors."""
    if (
        geometry.representation != "intrinsic"
        or center.shape[1] != 3
        or center.device.type != "cuda"
        or center.dtype != torch.float32
    ):
        raise ValueError("compact decoder requires intrinsic 4D CUDA FP32 centers")
    decoded = torch.empty(
        (center.shape[0], 4), device=center.device, dtype=center.dtype
    )
    if center.shape[0]:
        _decode_intrinsic_torus[(tr.cdiv(center.shape[0], 256),)](
            center,
            geometry.major_radius,
            geometry.minor_radius,
            decoded,
            center.shape[0],
            *center.stride(),
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return decoded


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


@torch.no_grad()
def balanced_home_columns(site):
    """Initial balanced-site column as a reusable positive support hint."""
    if site.atom_init != "balanced":
        raise ValueError("home columns require balanced atom initialization")
    count = site.atoms.p.shape[0]
    positions = (
        torch.linspace(
            0,
            site.chart.features - 1,
            count,
            device=site.atoms.p.device,
            dtype=torch.float64,
        )
        .round()
        .to(torch.int32)
    )
    return (positions % site.chart.shape[1]).contiguous()


@torch.no_grad()
def boxed_prepare(
    site,
    p,
    *,
    boxes=None,
    witness_cols=None,
    fast_witness=False,
    fast_decode=False,
):
    """Prepare current atom values with exact fallback for box-overlapping stations."""
    from torchcst.nn._backends._preparation import execution_plan
    from torchcst.nn._backends._triton_preparation import (
        Pack,
        route_and_layout,
        tile_parameters,
    )

    plan = execution_plan(site)
    station_rows = site.chart.tile_shape[0]
    if boxes is None:
        boxes = station_site_boxes(plan.circle, plan.section, station_rows)
    center, amplitude, precision = tile_parameters(site.kernel, p)
    decoded = (
        decode_intrinsic_torus_compact(site.chart.geometry, center)
        if fast_decode
        else site.chart.geometry.decode_centers(center)
    )
    _, order, offsets = route_and_layout(
        plan.routing,
        decoded,
        support=(plan.circle, plan.section, precision, station_rows),
        retain_owners=False,
        support_boxes=boxes,
        support_witness_cols=witness_cols,
        support_fast_witness=fast_witness,
    )
    packed = Pack.apply(amplitude, precision, decoded, order)
    return packed, plan.circle, plan.section, offsets


@tr.jit
def support_buckets_batched_box(
    Centers,
    Precision,
    Owners,
    Circle,
    Section,
    Boxes,
    OwnerRows,
    WitnessCols,
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
    USE_WITNESS: tl.constexpr = False,
    FAST_WITNESS: tl.constexpr = False,
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
        if (G == 1) or (shift == 1):
            possible = atom_valid
        else:
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
        if FAST_WITNESS and ((G == 1) or (shift == 1)):
            fast_row = tl.load(OwnerRows + a, atom_valid, 0)
            fast_hint = tl.load(WitnessCols + a, atom_valid, 0)
            fast_valid = atom_valid & (fast_hint >= 0) & (fast_hint < K)
            fast_cosine = tl.load(Circle + fast_row * 2, fast_valid, 0.0)
            fast_sine = tl.load(Circle + fast_row * 2 + 1, fast_valid, 0.0)
            fast_rho = tl.load(Section + fast_hint * 3, fast_valid, 0.0)
            fast_z = tl.load(Section + fast_hint * 3 + 1, fast_valid, 0.0)
            fast_w = tl.load(Section + fast_hint * 3 + 2, fast_valid, 0.0)
            fast_dx = fast_cosine * fast_rho - cx
            fast_dy = fast_sine * fast_rho - cy
            fast_dz = fast_z - cz
            fast_dw = fast_w - cw
            fast_distance = (fast_dx * fast_dx + fast_dy * fast_dy) + (
                fast_dz * fast_dz + fast_dw * fast_dw
            )
            fast_hit = fast_valid & (fast_distance * precision < 0.9999)
            hit = fast_hit
            possible = possible & ~fast_hit
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
            if (USE_WITNESS & (not FAST_WITNESS)) & ((G == 1) | (shift == 1)):
                hint = tl.load(WitnessCols + a, atom_valid, 0)
                hint_valid = (hint >= 0) & (hint < K) & possible
                hint_rho = tl.load(Section + hint * 3, hint_valid, 0.0)
                hint_z = tl.load(Section + hint * 3 + 1, hint_valid, 0.0)
                hint_w = tl.load(Section + hint * 3 + 2, hint_valid, 0.0)
                hint_dx = cosine * hint_rho - cx
                hint_dy = sine * hint_rho - cy
                hint_dz = hint_z - cz
                hint_dw = hint_w - cw
                witness_distance = (hint_dx * hint_dx + hint_dy * hint_dy) + (
                    hint_dz * hint_dz + hint_dw * hint_dw
                )
                witness = hint_valid & (witness_distance * precision < 0.9999)
                hit = witness
                need_exact = possible & ~witness
            else:
                need_exact = possible
            if tl.sum(need_exact.to(tl.int32), 0) > 0:
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
                            (
                                (k[None, :] < K) & (squared * precision[:, None] < 1.0)
                            ).to(tl.int32),
                            axis=1,
                        )
                        > 0
                    ) & need_exact
        second = tl.where(hit & (count == 1), station, second)
        first = tl.where(hit & (count == 0), station, first)
        count += hit.to(tl.int32)
    if G == 2:
        boundary = 0
    else:
        boundary = tl.where((first + 1) % G == second, first, second)
    key = tl.where(count == 0, 2 * G, tl.where(count == 1, 2 * first, 2 * boundary + 1))
    tl.store(Keys + a, key, atom_valid)
