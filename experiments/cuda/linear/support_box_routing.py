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


@tr.jit
def _bucket_histogram(Keys, Counts, A: tl.constexpr, B: tl.constexpr):
    atom = tl.program_id(0) * B + tl.arange(0, B)
    valid = atom < A
    key = tl.load(Keys + atom, valid, 0)
    tl.atomic_add(Counts + key, 1, valid, sem="relaxed")


@tr.jit
def _bucket_scatter(Keys, Cursors, SortedKeys, Order, A: tl.constexpr, B: tl.constexpr):
    atom = tl.program_id(0) * B + tl.arange(0, B)
    valid = atom < A
    key = tl.load(Keys + atom, valid, 0)
    position = tl.atomic_add(Cursors + key, 1, valid, sem="relaxed")
    tl.store(SortedKeys + position, key, valid)
    tl.store(Order + position, atom, valid)


@torch.no_grad()
def atomic_bucket_sort(keys, buckets):
    """Forward-only low-scratch bucket permutation; order within a bucket varies."""
    if keys.dtype != torch.int32 or keys.device.type != "cuda":
        raise ValueError("atomic bucket sort requires CUDA int32 keys")
    counts = torch.zeros(buckets, dtype=torch.int32, device=keys.device)
    if keys.numel():
        _bucket_histogram[(tr.cdiv(keys.numel(), 128),)](
            keys, counts, keys.numel(), 128, num_warps=4
        )
    cursors = torch.cumsum(counts, dim=0, dtype=torch.int32) - counts
    sorted_keys = torch.empty_like(keys)
    order = torch.empty(keys.numel(), dtype=torch.int32, device=keys.device)
    if keys.numel():
        _bucket_scatter[(tr.cdiv(keys.numel(), 128),)](
            keys, cursors, sorted_keys, order, keys.numel(), 128, num_warps=4
        )
    return sorted_keys, order


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
    from torchcst._backends.cuda.algorithms.strip_torus.fused.preparation import (
        Pack,
        route_and_layout,
        tile_parameters,
    )
    from torchcst._backends.torch.operators.strip_torus.preparation import (
        execution_plan,
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
