"""Explicit initial recipes, separate from mathematical tile boundaries."""

import re
from dataclasses import dataclass

_SITE_SCALED_TUNING = re.compile(
    r"site_scaled_hc(4|8)_g(64|128|256|512|1024)_cin(256|512|1024)(?:_slab(8|16))?(?:_(reuse_sigma|known_bank|exact_reuse))?(_active_z|_node_z)?(?:_cached(32|64))?(_expand)?"
)


@dataclass(frozen=True)
class HierarchicalCellMassRecipe:
    preset: str = "conservative"

    @property
    def is_site_scaled(self):
        return self.preset in ("site_scaled", "site_scaled_release") or (
            isinstance(self.preset, str)
            and _SITE_SCALED_TUNING.fullmatch(self.preset) is not None
        )

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
        if self.is_site_scaled:
            options = dict(
                common,
                c_splits=128,
                layout_method="sort",
                reverse_layout=True,
                atoms_per_split=256,
                z_chunk_sites=32,
                atom_threads=64,
                window_moments=True,
                compensated_values=True,
                compensated_grad=True,
                fuse_h_c=True,
                ragged_c_splits=True,
            )
            if self.preset != "site_scaled":
                options["release_workspaces"] = True
            tuning = _SITE_SCALED_TUNING.fullmatch(self.preset)
            if tuning:
                batch_tile, forward_grain, reverse_grain = map(int, tuning.groups()[:3])
                options.update(
                    hc_batch_tile=batch_tile,
                    hc_atoms_per_split=forward_grain,
                    atoms_per_split=reverse_grain,
                )
                if tuning.group(4):
                    options["cin_batch_slab"] = int(tuning.group(4))
                if tuning.group(5):
                    options["hc_reuse_sigma_decode"] = tuning.group(5) in (
                        "reuse_sigma",
                        "exact_reuse",
                    )
                    options["hc_known_input_bank"] = tuning.group(5) in (
                        "known_bank",
                        "exact_reuse",
                    )
                if tuning.group(6):
                    options["active_bank_z"] = True
                    if tuning.group(6) == "_node_z":
                        options["z_chunk_sites"] = 0
                if tuning.group(7):
                    options["cache_polar_coefficients"] = True
                    options["atom_threads"] = int(tuning.group(7))
                if tuning.group(8):
                    options["active_bank_expand"] = True
            return options
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
