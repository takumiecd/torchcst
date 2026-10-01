"""Validated launch settings; mathematical meaning stays in OperatorSpec."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FullRecipe:
    id: str = "normalized_full.default.v1"
    atom_num_warps: int = 1
    sorted_forward: bool = True
    sorted_backward: bool = True
    support: str = "ball"
    enable_fp_fusion: bool = True
    saved_support_flags: bool = True
    tuple_grads: bool = False
