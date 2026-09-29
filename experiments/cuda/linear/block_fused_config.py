"""Execution-only tuning for the mapped forward prototype; chart tiles stay fixed."""

import math
from dataclasses import dataclass

import torch

from torchcst.nn._backends._preparation import PROFILE_KINDS


@dataclass(frozen=True)
class FusedConfig:
    batch_rows: int = 64
    output_rows: int = 16
    columns: int = 16
    atoms: int = 8
    warps: int = 4
    late_reduce: bool = False
    fp_fusion: bool = False
    split_k: int = 1
    hoist_section: bool = False
    column_factored: bool = False

    def __post_init__(self):
        for name, allowed in (
            ("batch_rows", (16, 32, 64, 128)),
            ("output_rows", (16, 32)),
            ("columns", (16, 32, 64)),
            ("atoms", (1, 2, 4, 8, 16, 32, 64)),
            ("warps", (4, 8)),
        ):
            if (
                type(getattr(self, name)) is not int
                or getattr(self, name) not in allowed
            ):
                raise ValueError(f"unsupported fused {name}")
        if type(self.late_reduce) is not bool:
            raise ValueError("late_reduce must be bool")
        if type(self.fp_fusion) is not bool:
            raise ValueError("fp_fusion must be bool")
        if type(self.split_k) is not int or self.split_k not in (1, 2, 4, 8, 16):
            raise ValueError("unsupported fused split_k")
        if type(self.hoist_section) is not bool:
            raise ValueError("hoist_section must be bool")
        if type(self.column_factored) is not bool:
            raise ValueError("column_factored must be bool")
        if self.column_factored and (self.atoms != 1 or not self.late_reduce):
            raise ValueError("column_factored requires atoms=1 and late_reduce")


def default_fused_config(
    tile_shape,
    input_rows,
    requested_batch_rows=None,
    *,
    atom_density=0.0,
    gpu_name="",
    logical_shape=None,
):
    """Use the measured A100 schedule for 64x64 charts with enough input rows.

    FP fusion stays off unless that schedule applies on an A100 and the atom
    count is at least 5% of the dense matrix.
    """
    if requested_batch_rows is not None:
        return FusedConfig(batch_rows=requested_batch_rows)
    if tuple(tile_shape) == (64, 64) and input_rows >= 128:
        threshold = 0.05
        if logical_shape is not None:
            elements = math.prod(logical_shape)
            threshold = round(elements * 0.05) / elements
        fp_fusion = "A100" in gpu_name and atom_density >= threshold
        if (
            fp_fusion
            and input_rows == 128
            and logical_shape in ((4096, 4096), (8192, 8192))
            and atom_density < 0.051
        ):
            return FusedConfig(128, 32, 32, 1, 4, True, True, 8, True, True)
        return FusedConfig(batch_rows=128, late_reduce=True, fp_fusion=fp_fusion)
    return FusedConfig()


def launch_fused(layer, x, prepared, y, config):
    from experiments.cuda.linear.block_strip_kernels import (
        block_fused,
        block_split_reduce,
    )

    p, circle, section, offsets = prepared
    if config.column_factored and p.shape[1] - 2 != 4:
        raise ValueError("column_factored requires four-dimensional atom coordinates")
    s, t = layer.tile_shape
    partial = (
        y
        if config.split_k == 1
        else torch.empty((config.split_k, *y.shape), device=y.device, dtype=y.dtype)
    )
    grid = (
        math.ceil(x.shape[0] / config.batch_rows),
        layer.row_groups * math.ceil(s / config.output_rows),
    )
    if config.split_k > 1:
        grid += (config.split_k,)
    compiled = block_fused[grid](
        x,
        p,
        circle,
        section,
        offsets,
        partial,
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
        SPLIT_K=config.split_k,
        HOIST_SECTION=config.hoist_section,
        COLUMN_FACTORED=config.column_factored,
        num_warps=config.warps,
        enable_fp_fusion=config.fp_fusion,
    )
    if config.split_k > 1:
        block_split_reduce[(math.ceil(y.numel() / 256),)](
            partial, y, x.shape[0], layer.shape[0], config.split_k
        )
    return compiled
