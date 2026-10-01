"""Conservative phase-box routing; site values use continuous centers."""

import torch


def build_atlas(radius=3.0502, pad=0.004):
    offsets = torch.cartesian_prod(
        torch.arange(-4, 5), torch.arange(-8, 9), torch.arange(-8, 9)
    )
    spacing = torch.tensor([1.0, 0.5, 0.5], dtype=torch.float64)
    sites = offsets * spacing
    phases = torch.cartesian_prod(*[torch.arange(16)] * 3).to(torch.float64)
    lo = (phases / 16 - 0.5) * spacing - pad
    hi = (phases / 16 - 0.5 + 1 / 16) * spacing + pad
    encoded = (
        (offsets[:, 0] + 4) | ((offsets[:, 1] + 8) << 4) | ((offsets[:, 2] + 8) << 9)
    ).to(torch.int32)
    table = torch.zeros((4096, 512), dtype=torch.int32)
    counts = torch.zeros(4096, dtype=torch.int32)
    for begin in range(0, 4096, 128):
        delta = torch.maximum(
            torch.maximum(
                lo[begin : begin + 128, None] - sites,
                sites - hi[begin : begin + 128, None],
            ),
            torch.tensor(0.0, dtype=torch.float64),
        )
        mask = delta.square().sum(-1) <= radius * radius
        for row, keep in enumerate(mask, begin):
            codes = encoded[keep]
            counts[row] = len(codes)
            if len(codes) > 512:
                raise RuntimeError("conservative atlas capacity exceeded")
            table[row, : len(codes)] = codes
    # Match the validated 16-bit table layout without requiring NumPy.
    return table.to(torch.uint16), counts
