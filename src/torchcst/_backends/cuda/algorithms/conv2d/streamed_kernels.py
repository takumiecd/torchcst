"""Keep the X-to-Atom tile within the same CTA as output accumulation."""

import triton as tr
import triton.language as tl


@tr.jit
def _project_tiles(
    X,
    Left,
    Right,
    Partial,
    Rows,
    I: tl.constexpr,
    O: tl.constexpr,
    A: tl.constexpr,
    LI: tl.constexpr,
    LA: tl.constexpr,
    RO: tl.constexpr,
    RA: tl.constexpr,
    BI: tl.constexpr,
    BO: tl.constexpr,
    BM: tl.constexpr,
    BA: tl.constexpr,
    TILES_PER_SPLIT: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    split = tl.program_id(1)
    ci = tl.arange(0, BI)
    co = tl.arange(0, BO)
    aa = tl.arange(0, BA)
    x = tl.load(
        X + row[:, None] * I + ci[None, :], (row[:, None] < Rows) & (ci[None, :] < I), 0
    )
    y = tl.full((BM, BO), 0, tl.float32)
    for tile in range(TILES_PER_SPLIT):
        atom = (split * TILES_PER_SPLIT + tile) * BA + aa
        left = tl.load(
            Left + ci[:, None] * LI + atom[None, :] * LA,
            (ci[:, None] < I) & (atom[None, :] < A),
            0,
        )
        right = tl.load(
            Right + atom[:, None] * RA + co[None, :] * RO,
            (atom[:, None] < A) & (co[None, :] < O),
            0,
        )
        response = tl.dot(x, left, input_precision="tf32x3")
        y = tl.dot(response, right, y, input_precision="tf32x3")
    tl.store(
        Partial + (split * Rows + row[:, None]) * O + co[None, :],
        y,
        (row[:, None] < Rows) & (co[None, :] < O),
    )


def project(rows, left, right):
    row_count, inputs = rows.shape
    outputs, atoms = right.shape
    if not row_count:
        return rows.new_empty((0, outputs))
    rows_per_block = 32
    atoms_per_tile = 32
    atom_tiles = tr.cdiv(atoms, atoms_per_tile)
    # The 4096-row shortcut favors four splits over seven. Cap the split
    # count so the global partial outputs do not grow with the atom count.
    tiles_per_split = max(14, tr.cdiv(atom_tiles, 4))
    splits = tr.cdiv(atom_tiles, tiles_per_split)
    partial = rows.new_empty((splits, row_count, outputs))
    _project_tiles[(tr.cdiv(row_count, rows_per_block), splits)](
        rows,
        left,
        right,
        partial,
        row_count,
        inputs,
        outputs,
        atoms,
        *left.stride(),
        *right.stride(),
        max(16, tr.next_power_of_2(inputs)),
        max(16, tr.next_power_of_2(outputs)),
        rows_per_block,
        atoms_per_tile,
        tiles_per_split,
        num_warps=4,
        num_stages=1,
    )
    return partial[0] if splits == 1 else partial.sum(dim=0)
