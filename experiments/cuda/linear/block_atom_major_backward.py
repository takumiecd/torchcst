"""Station-local atom gradient reduction from a bounded logical dW window."""

import triton as tr
import triton.language as tl

from torchcst._backends.cuda.algorithms.strip_torus.fused.kernels import _profile


@tr.jit
def mapped_backward_atoms_station(
    DW,
    P,
    Circle,
    Section,
    Offsets,
    DP,
    K: tl.constexpr,
    S: tl.constexpr,
    T: tl.constexpr,
    CG: tl.constexpr,
    G: tl.constexpr,
    PROFILE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BA: tl.constexpr,
    LANES: tl.constexpr,
    STATION_START,
    ROW_START,
):
    """Write one five-component partial gradient per atom and station."""
    tl.static_assert(S % BN == 0 and T % BK == 0)
    station = STATION_START + tl.program_id(0)
    lane = tl.program_id(1)
    row_group = station // CG
    col_group = station % CG
    for neighbor in tl.static_range(1 if G == 1 else 3):
        if G == 1:
            bucket = 0
        else:
            bucket = tl.where(
                neighbor == 0,
                2 * ((station + G - 1) % G) + 1,
                2 * station + neighbor - 1,
            )
        begin, end = tl.load(Offsets + bucket), tl.load(Offsets + bucket + 1)
        for atom_start in range(begin + lane * BA, end, LANES * BA):
            atoms = atom_start + tl.arange(0, BA)
            active = atoms < end
            amplitude = tl.load(P + atoms * 6, active, 0.0)
            precision = tl.load(P + atoms * 6 + 1, active, 0.0)
            cx = tl.load(P + atoms * 6 + 2, active, 0.0)
            cy = tl.load(P + atoms * 6 + 3, active, 0.0)
            cz = tl.load(P + atoms * 6 + 4, active, 0.0)
            cw = tl.load(P + atoms * 6 + 5, active, 0.0)
            da = tl.full((BA,), 0.0, tl.float32)
            dcx = tl.full((BA,), 0.0, tl.float32)
            dcy = tl.full((BA,), 0.0, tl.float32)
            dcz = tl.full((BA,), 0.0, tl.float32)
            dcw = tl.full((BA,), 0.0, tl.float32)
            for rb in tl.static_range(S // BN):
                virtual_rows = station * S + rb * BN + tl.arange(0, BN)
                logical_rows = row_group * S + rb * BN + tl.arange(0, BN)
                cosine = tl.load(Circle + virtual_rows * 2)
                sine = tl.load(Circle + virtual_rows * 2 + 1)
                for cb in tl.static_range(T // BK):
                    local_cols = cb * BK + tl.arange(0, BK)
                    logical_cols = col_group * T + local_cols
                    rho = tl.load(Section + local_cols * 3)
                    z = tl.load(Section + local_cols * 3 + 1)
                    w = tl.load(Section + local_cols * 3 + 2)
                    sx = tl.reshape(cosine[:, None] * rho[None, :], (BN * BK,))
                    sy = tl.reshape(sine[:, None] * rho[None, :], (BN * BK,))
                    sz = tl.reshape(tl.broadcast_to(z[None, :], (BN, BK)), (BN * BK,))
                    sw = tl.reshape(tl.broadcast_to(w[None, :], (BN, BK)), (BN * BK,))
                    grad = tl.load(
                        DW
                        + (logical_rows[:, None] - ROW_START) * K
                        + logical_cols[None, :]
                    )
                    grad = tl.reshape(grad, (BN * BK,))
                    dx = sx[:, None] - cx[None, :]
                    dy = sy[:, None] - cy[None, :]
                    dz = sz[:, None] - cz[None, :]
                    dw = sw[:, None] - cw[None, :]
                    squared = (dx * dx + dy * dy) + (dz * dz + dw * dw)
                    value, slope = _profile(squared, precision[None, :], PROFILE)
                    da += tl.sum(grad[:, None] * value, axis=0)
                    scale = -2.0 * grad[:, None] * slope * amplitude[None, :]
                    dcx += tl.sum(scale * dx, axis=0)
                    dcy += tl.sum(scale * dy, axis=0)
                    dcz += tl.sum(scale * dz, axis=0)
                    dcw += tl.sum(scale * dw, axis=0)
            tl.atomic_add(DP + atoms * 6, da, active, sem="relaxed")
            tl.atomic_add(DP + atoms * 6 + 2, dcx, active, sem="relaxed")
            tl.atomic_add(DP + atoms * 6 + 3, dcy, active, sem="relaxed")
            tl.atomic_add(DP + atoms * 6 + 4, dcz, active, sem="relaxed")
            tl.atomic_add(DP + atoms * 6 + 5, dcw, active, sem="relaxed")
