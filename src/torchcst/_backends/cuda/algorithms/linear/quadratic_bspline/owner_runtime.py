"""Per-forward immutable event tables; scratch is never shared across frames."""

import torch
import triton as tr

from . import local_owner_kernels as ok
from .executor import _report, _stamp
from .local_executor import _points


def prepare(x, pos, injection, count, a, sizes, recipe):
    _stamp("owner_prepare", 0)
    tables = []
    for dim, n in enumerate(sizes):
        points = _points(n, recipe)

        def empty(size):
            return x.new_empty(size, dtype=torch.int32)

        hist, cursor = empty(points).zero_(), empty(points).zero_()
        ends, chunks, chunk_ends = empty(points), empty(points), empty(points)
        ids = empty(6 * a)
        bound = points + tr.cdiv(6 * a, recipe.owner_events)
        mapping = empty(bound)
        compiled = ok.histogram[(tr.cdiv(6 * a, 128),)](
            pos, injection, count, hist, a, dim
        )
        _report(f"owner_histogram_{dim}", compiled)
        torch.cumsum(hist, 0, dtype=torch.int32, out=ends)
        compiled = ok.place[(tr.cdiv(6 * a, 128),)](
            pos, injection, count, hist, ends, cursor, ids, a, dim
        )
        _report(f"owner_place_{dim}", compiled)
        compiled = ok.chunk_counts[(tr.cdiv(points, 256),)](
            hist, chunks, points, recipe.owner_events
        )
        _report(f"owner_chunk_counts_{dim}", compiled)
        torch.cumsum(chunks, 0, dtype=torch.int32, out=chunk_ends)
        compiled = ok.chunk_map[(tr.cdiv(bound, 128),)](
            chunk_ends, mapping, points, bound, points.bit_length() + 1
        )
        _report(f"owner_map_{dim}", compiled)
        tables.append((hist, ends, ids, chunk_ends, mapping))
    _stamp("owner_prepare", 1)
    return tables


def inject(weight, co, table, delta, actual, a, dim, recipe, label):
    hist, ends, ids, chunk_ends, mapping = table
    _stamp(label, 0)
    compiled = ok.reduce[(len(mapping),)](
        weight,
        co,
        hist,
        ends,
        ids,
        chunk_ends,
        mapping,
        delta,
        a,
        actual,
        weight.shape[1],
        len(hist),
        dim,
        recipe.owner_events,
        tr.next_power_of_2(actual),
        num_warps=4,
        enable_fp_fusion=False,
    )
    _report(label, compiled)
    _stamp(label, 1)
