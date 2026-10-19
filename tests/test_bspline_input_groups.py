"""Index membership checks, independent of a future grouped H/G executor."""

import pytest
import torch

from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.input_groups import (
    layout,
)

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


def test_input_group_layout_bounds():
    assert layout(52428, 1024) == (131072, 14)
    assert layout(0, 8) == (256, 5)
    assert layout(129, 4096) == (512, 16)
    for args in ((True, 64), (-1, 64), (4, 7), (4, 8192), (4, 64, 0)):
        with pytest.raises(ValueError):
            layout(*args)


def positions(a, n, seed=42, same=False):
    gen = torch.Generator().manual_seed(seed)
    points = (n.bit_length() - 2) * n + 2 * (n >> 2) - 1
    rows = torch.randint(
        points, (max(1, (a + 4) // 5), 4), generator=gen, dtype=torch.int32
    )
    membership = torch.arange(a) // 5
    if same:
        membership.zero_()
    pos = torch.full((2, 6, a), -1, dtype=torch.int32)
    pos[1, :4] = rows[membership].T
    return pos.cuda()


def check_groups(groups, pos, count):
    a = pos.shape[-1]
    hist, ends, ids, chunk_ends, mapping = (
        v.cpu()
        for v in (
            groups.counts,
            groups.ends,
            groups.ids,
            groups.chunk_ends,
            groups.mapping,
        )
    )
    assert torch.equal(ends, hist.cumsum(0).int())
    assert int(hist.sum()) == a - count
    assert torch.equal(
        ids[: a - count].sort()[0], torch.arange(count, a, dtype=torch.int32)
    )
    rows = pos[1].T.cpu()
    expected = {}
    for row in rows[count:].tolist():
        key = tuple(row)
        expected[key] = expected.get(key, 0) + 1
    actual = {}
    for bucket in hist.nonzero().flatten().tolist():
        members = ids[int(ends[bucket] - hist[bucket]) : int(ends[bucket])].long()
        selected = rows[members]
        assert torch.equal(selected, selected[:1].expand_as(selected))
        key = tuple(selected[0].tolist())
        assert key not in actual
        actual[key] = len(members)
    assert actual == expected
    chunks = (hist + groups.chunk_atoms - 1) // groups.chunk_atoms
    assert torch.equal(chunk_ends, chunks.cumsum(0).int())
    total = int(chunk_ends[-1])
    expected_map = torch.searchsorted(chunk_ends, torch.arange(total), right=True)
    assert torch.equal(mapping[:total].long(), expected_map)


@GPU
@pytest.mark.parametrize(
    "a,n,count",
    [
        (0, 8, 0),
        (1, 8, 0),
        (1, 8, 1),
        (137, 64, 0),
        (137, 64, 3),
        (137, 64, 137),
        (513, 4096, 0),
        (1025, 1024, 511),
    ],
)
@pytest.mark.parametrize("chunk_atoms", [2, 4, 8])
def test_input_groups_membership(a, n, count, chunk_atoms):
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.input_groups import (
        prepare,
    )

    pos = positions(a, n)
    groups = prepare(pos, pos.new_full((), count), a, n, chunk_atoms)
    check_groups(groups, pos, count)


@GPU
def test_input_groups_one_key_and_graph_snapshots():
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.input_groups import (
        prepare,
    )

    a, n = 1025, 1024
    pos = positions(a, n, same=True)
    count = pos.new_full((), 0)
    retained = prepare(pos, count, a, n)
    snapshots = [
        v.clone()
        for v in (
            retained.counts,
            retained.ends,
            retained.ids,
            retained.chunk_ends,
            retained.mapping,
        )
    ]
    check_groups(retained, pos, 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        groups = prepare(pos, count, a, n)
    for step, native in enumerate((3, a, 0, a - 1)):
        pos.copy_(positions(a, n, seed=43 + step))
        count.fill_(native)
        graph.replay()
        check_groups(groups, pos, native)
    for actual, expected in zip(
        (
            retained.counts,
            retained.ends,
            retained.ids,
            retained.chunk_ends,
            retained.mapping,
        ),
        snapshots,
    ):
        # Unwritten ID/mapping suffixes remain private to the retained frame too.
        assert torch.equal(actual, expected)


@GPU
def test_input_groups_full_primary_geometry():
    from types import SimpleNamespace as NS

    from benchmarks.cuda.linear import bspline as bs
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.input_groups import (
        prepare,
    )

    a, n = 52428, 1024
    case = NS(
        size=n,
        atoms=a,
        seed=41,
        widths=NS(
            minimum=0.1,
            birth=0.1,
            maximum=12.8,
            rho=(0.1, 0.4, 1.6, 6.4, 12.0),
            fractions=(0.2,) * 5,
        ),
    )
    p = bs.initialize(case)
    decoded = bs.decode(bs.fixture_state(case), p.double())
    radius = (6.0**0.5) * decoded[:, 1].rsqrt()
    spacing = float(torch.tensor(0.1))
    size = (2 ** torch.ceil(torch.log2(radius / spacing))).clamp(4, n).long()
    shift = size.double().log2().round().long()
    tiles = n // size
    cuts = (
        torch.ceil(
            (
                p[:, 3, None].double()
                + radius[:, None]
                * torch.tensor([-1, -1 / 3, 1 / 3, 1], dtype=torch.float64)
            )
            / spacing
        )
        .clamp(0, n)
        .long()
    )
    offset = (shift - 2) * n + 2 * (n >> 2) - 2 * tiles
    tile = torch.minimum(((cuts - 1) // size[:, None]).clamp_min(0), tiles[:, None] - 1)
    point = offset[:, None] + tile * (size[:, None] + 1) + cuts - tile * size[:, None]
    pos = torch.full((2, 6, a), -1, dtype=torch.int32)
    pos[1, :4] = point.T.int()
    first = tile[:, 0]
    for event in (4, 5):
        boundary = (first + event - 3) * size
        before = torch.minimum(first + event - 4, tiles - 1)
        active = (boundary < n) & (boundary < cuts[:, 3])
        point = offset + before * (size + 1) + size
        pos[1, event] = torch.where(active, point, -1).int()
    pos = pos.cuda()
    groups = prepare(pos, pos.new_full((), 0), a, n)
    # Check every member and all six positions, beyond group-size equality.
    check_groups(groups, pos, 0)
