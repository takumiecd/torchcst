"""Explicit preparation granularity; earlier recipe schemas remain immutable."""

from dataclasses import asdict, dataclass

from .recipe_v2 import GroupedProductRecipe


@dataclass(frozen=True)
class PreparationProductRecipe(GroupedProductRecipe):
    prep_group: int = 16
    prep_sites: int = 16

    def __post_init__(self):
        if type(self.prep_group) is not int or self.prep_group not in (1, 8, 16):
            raise ValueError("requires preparation groups1,8 or16")
        if type(self.prep_sites) is not int or self.prep_sites not in (16, 32):
            raise ValueError("requires preparation site blocks16 or32")
        # Reuse unchanged field validation without extending the v2 schema.
        fields = asdict(self)
        del fields["prep_sites"]
        if self.prep_group == 16:
            fields["prep_group"] = 8
        GroupedProductRecipe(**fields)


@dataclass(frozen=True)
class PreparationStripRecipe(PreparationProductRecipe):
    """Preparation choices on the same live physical Strip coordinates."""
