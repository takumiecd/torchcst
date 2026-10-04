"""Small regular domains; full-domain norms survive local slicing."""

import math
from dataclasses import dataclass

import torch

from torchcst._backends.torch.charts import construction


@dataclass(frozen=True)
class Domain:
    input_size: int
    output_size: int
    spacing: float = 1.0
    input_origin: float = 0.0
    output_origin: float = 0.0
    input_start: int = 0
    output_start: int = 0
    input_count: int | None = None
    output_count: int | None = None

    def __post_init__(self):
        for side in ("input", "output"):
            size = getattr(self, side + "_size")
            start = getattr(self, side + "_start")
            count = getattr(self, side + "_count")
            count = size - start if count is None else count
            if not (
                2 <= size <= 128
                and 0 <= start < size
                and 1 <= count <= 64
                and start + count <= size
            ):
                raise ValueError("small domain or slice out of bounds")
            object.__setattr__(self, side + "_count", count)
        if not math.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("spacing must be positive")
        if not all(math.isfinite(o) for o in (self.input_origin, self.output_origin)):
            raise ValueError("origins must be finite")

    def charts(self, *, device="cpu", dtype=torch.float32):
        return tuple(
            construction.linspace(
                getattr(self, side + "_size"),
                low=getattr(self, side + "_origin"),
                high=getattr(self, side + "_origin")
                + (getattr(self, side + "_size") - 1) * self.spacing,
            ).to(device=device, dtype=dtype)
            for side in ("input", "output")
        )
