"""Fixed shared preparation and grouped support contraction settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodicRecipe:
    gemm: str = "torch"
    prep_group: int = 16
    prep_sites: int = 16
    atom_group: int = 8
    patch_sites: int = 8

    def __post_init__(self):
        if type(self.gemm) is not str or self.gemm not in ("torch", "triton"):
            raise ValueError("gemm must be torch or triton")
        for name, choices in (
            ("prep_group", (1, 4, 8, 16)),
            ("prep_sites", (8, 16, 32)),
            ("atom_group", (1, 4, 8)),
            ("patch_sites", (8, 16, 32)),
        ):
            value = getattr(self, name)
            if type(value) is not int or value not in choices:
                raise ValueError(f"unsupported {name}")
