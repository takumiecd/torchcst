"""Execution choices independent of the production mathematical kernel."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Recipe:
    rho_upper: tuple[float, ...] = (1.0, 2.0, 4.0, 16.0)
    atom_block: int = 16
    batch_block: int = 16
    pack: bool = True
    support_limit: int = 8

    @property
    def output_block(self):
        return 16

    @property
    def recompute_h(self):
        return False

    @property
    def recompute_param_h(self):
        return self.recompute_h

    @property
    def unroll_support(self):
        return False

    @property
    def vector_support(self):
        return False

    @property
    def contraction_warps(self):
        return 0  # Preserve the size-dependent launch when unspecified.

    @property
    def parameter_warps(self):
        return 0

    @property
    def parameter_atom_block(self):
        return self.atom_block

    @property
    def support_prepare(self):
        return False

    @property
    def save_g(self):
        return False

    @property
    def band_dispatch(self):
        return False

    @property
    def owner_index(self):
        return False

    @property
    def index_bits(self):
        return 32

    @property
    def release_forward_index(self):
        return False

    @property
    def fuse_owner_index(self):
        return False

    @property
    def release_forward_h(self):
        return False

    @property
    def ordered_layout(self):
        return False

    @property
    def order_by_position(self):
        return False

    @property
    def compact_order_key(self):
        return False

    @property
    def parallel_owner_ranges(self):
        return False

    @property
    def parallel_order_copy(self):
        return False

    @property
    def vector_owner_ranges(self):
        return False

    @property
    def cached_order(self):
        return False

    @property
    def cache_repair_rounds(self):
        return 0

    @property
    def gather_validation_key(self):
        return False

    @property
    def validate_cached_order(self):
        return False

    @property
    def compact_cached_order(self):
        return False

    @property
    def preparation_warps(self):
        return 4

    @property
    def owner_splits(self):
        return 1

    @property
    def parameter_splits(self):
        return 1

    def __post_init__(self):
        if not self.rho_upper or any(
            not math.isfinite(x) or x <= 0 for x in self.rho_upper
        ):
            raise ValueError("invalid sigma/spacing boundaries")
        if any(a >= b for a, b in zip(self.rho_upper, self.rho_upper[1:])):
            raise ValueError("boundaries must increase")
        if type(self.support_limit) is not int or self.support_limit not in (
            1,
            2,
            4,
            8,
        ):
            raise ValueError("support limit must be 1/2/4/8")
        if self.atom_block not in (16, 32) or self.batch_block != 16:
            raise ValueError("initial local blocks: atoms 16/32, batch 16")


DEFAULT_RECIPE = Recipe()
