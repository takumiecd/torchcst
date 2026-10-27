"""Explicit initial recipes, separate from mathematical tile boundaries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HierarchicalCellMassRecipe:
    preset: str = "conservative"

    def arguments(self):
        if self.preset == "conservative":
            return {}
        common = {
            "c_splits": 2,
            "fuse_g_grad": True,
            "layout_method": "counting_compact",
        }
        if self.preset == "validated_128":
            return common
        if self.preset == "validated_1024":
            return dict(
                common,
                c_splits=128,
                z_chunk_sites=32,
                atom_threads=64,
                window_moments=True,
                reverse_layout=True,
                atoms_per_split=64,
                compensated_values=True,
                compensated_grad=True,
                reuse_shape_coefficients=True,
                specialize_g_grad=True,
                fuse_h_c=True,
                ragged_c_splits=True,
            )
        raise ValueError("unknown hierarchical cell-mass preset")
