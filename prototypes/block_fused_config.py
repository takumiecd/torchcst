"""Execution-only tuning for the mapped forward prototype; chart tiles stay fixed."""

import math
from dataclasses import dataclass

from torchcst.nn._backends._preparation import PROFILE_KINDS


@dataclass(frozen=True)
class FusedConfig:
    batch_rows: int = 64
    output_rows: int = 16
    columns: int = 16
    atoms: int = 8
    warps: int = 4
    late_reduce: bool = False

    def __post_init__(self):
        for name, allowed in (
            ("batch_rows", (16, 32, 64, 128)),
            ("output_rows", (16, 32)),
            ("columns", (16, 32, 64)),
            ("atoms", (4, 8, 16, 32, 64)),
            ("warps", (4, 8)),
        ):
            if (
                type(getattr(self, name)) is not int
                or getattr(self, name) not in allowed
            ):
                raise ValueError(f"unsupported fused {name}")
        if type(self.late_reduce) is not bool:
            raise ValueError("late_reduce must be bool")


def launch_fused(layer, x, prepared, y, config):
    from prototypes.block_strip_kernels import block_fused

    p, circle, section, offsets = prepared
    s, t = layer.tile_shape
    return block_fused[
        (
            math.ceil(x.shape[0] / config.batch_rows),
            layer.row_groups * math.ceil(s / config.output_rows),
        )
    ](
        x,
        p,
        circle,
        section,
        offsets,
        y,
        M=x.shape[0],
        N=layer.shape[0],
        K=layer.shape[1],
        S=s,
        T=t,
        CG=layer.column_groups,
        G=layer.strip.chart.tile_count,
        D=p.shape[1] - 2,
        PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
        BM=config.batch_rows,
        BN=config.output_rows,
        BK=config.columns,
        BA=config.atoms,
        LATE_REDUCE=config.late_reduce,
        num_warps=config.warps,
        enable_fp_fusion=False,
    )
