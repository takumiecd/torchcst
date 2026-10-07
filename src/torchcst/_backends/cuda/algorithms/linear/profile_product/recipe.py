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

    def __post_init__(self):
        if not isinstance(self.rho_upper, (tuple, list)):
            raise TypeError("requires explicit rho boundaries")
        object.__setattr__(self, "rho_upper", tuple(self.rho_upper))
        super().__post_init__()
        if (
            self.execution_route not in ("torch", "local", "saved")
            or type(self.pack) is not bool
            or self.pack
            or self.rho_upper != (1.0, 2.0, 4.0, 16.0)
            or type(self.atom_block) is not int
            or type(self.batch_block) is not int
            or self.support_limit != 8
        ):
            raise ValueError("requires torch/local/saved with canonical small blocks")
