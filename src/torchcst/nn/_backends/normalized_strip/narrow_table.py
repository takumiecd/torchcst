"""Conservative candidates for narrower widths; no kernel approximation."""

import torch

from .atlas_table import build_atlas


def build_narrow_tables():
    tables, counts = [], []
    for radius, capacity in [(0.30, 2), (0.65, 8), (1.10, 32)]:
        code, count = build_atlas(radius=radius, pad=0.004)
        if count.max() > capacity:
            raise RuntimeError("conservative narrow table capacity exceeded")
        tables.append(code[:, :capacity].reshape(-1))
        counts.append(count)
    return torch.cat(tables), torch.cat(counts)
