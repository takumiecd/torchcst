"""Small profile product execution choices, separate from KernelSpec."""

from dataclasses import dataclass

from ..local_product.recipe import Recipe


@dataclass(frozen=True)
class ProductRecipe(Recipe):
    execution_route: str = "local"
    pack: bool = False

    @property
    def parameter_batch_block(self):
        return 16

    @property
    def parameter_warps(self):
        return 4

    @property
    def ordered_layout(self):
        return self.execution_route == "ordered"

    @property
    def recompute_h(self):
        return self.ordered_layout

    @property
    def band_dispatch(self):
        return self.ordered_layout

    @property
    def vector_support(self):
        return self.ordered_layout

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
            self.execution_route not in ("torch", "local", "saved", "ordered")
            or type(self.pack) is not bool
            or self.pack
            or self.rho_upper != (1.0, 2.0, 4.0, 16.0)
            or type(self.atom_block) is not int
            or type(self.batch_block) is not int
            or self.support_limit != 8
        ):
            raise ValueError("requires torch/local/saved/ordered with canonical small blocks")
