"""Small profile product execution choices, separate from KernelSpec."""

from dataclasses import dataclass

from ..local_product.recipe import Recipe


@dataclass(frozen=True)
class ProductRecipe(Recipe):
    execution_route: str = "local"
    pack: bool = False

    @property
    def parameter_batch_block(self):
        return 0 if self.reuse_layout else 16

    @property
    def parameter_warps(self):
        return 4

    @property
    def ordered_layout(self):
        return self.execution_route in ("ordered", "reuse")

    @property
    def reuse_layout(self):
        return self.execution_route == "reuse"

    @property
    def order_by_position(self):
        return self.reuse_layout

    @property
    def compact_order_key(self):
        return self.reuse_layout

    @property
    def parallel_owner_ranges(self):
        return self.reuse_layout

    @property
    def histogram_owner_ranges(self):
        return self.reuse_layout

    @property
    def fused_histogram_owner_ranges(self):
        return self.reuse_layout

    @property
    def tight_histogram_owner_ranges(self):
        return self.reuse_layout

    @property
    def preparation_warps(self):
        return 8 if self.reuse_layout else 4

    @property
    def parameter_atom_block(self):
        return 16 if self.reuse_layout else self.atom_block

    @property
    def recompute_h(self):
        return self.ordered_layout

    @property
    def band_dispatch(self):
        return self.ordered_layout

    @property
    def vector_support(self):
        return self.ordered_layout and not self.reuse_layout

    @property
    def contraction_warps(self):
        return 4 if self.ordered_layout else 0

    @property
    def owner_splits(self):
        return 4 if self.ordered_layout else 1

    @property
    def parameter_splits(self):
        return 2 if self.ordered_layout else 1

    def __post_init__(self):
        if not isinstance(self.rho_upper, (tuple, list)):
            raise TypeError("requires explicit rho boundaries")
        object.__setattr__(self, "rho_upper", tuple(self.rho_upper))
        super().__post_init__()
        if (
            self.execution_route not in ("torch", "local", "saved", "ordered", "reuse")
            or type(self.pack) is not bool
            or self.pack
            or self.rho_upper != (1.0, 2.0, 4.0, 16.0)
            or type(self.atom_block) is not int
            or type(self.batch_block) is not int
            or self.support_limit != 8
        ):
            raise ValueError(
                "requires declared profile product route with canonical small blocks"
            )
