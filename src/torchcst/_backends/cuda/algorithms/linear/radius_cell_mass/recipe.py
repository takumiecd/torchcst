"""Explicit execution choices for radius-native cell masses."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RadiusCellMassRecipe:
    # Merge reuses DC and divisible frequencies; unmatched modes are direct.
    moment_method: str = "merge"
    save_response: bool = True
    # Atoms below these levels read X/dY directly or expand directly to sites.
    # The two axes are independent; zero preserves the original reference.
    input_direct_levels: int = 0
    output_direct_levels: int = 0
    # global_batched_cached retains global hierarchical Z/H/G while keeping
    # expansion coefficients in one bank/tile-local bounded shared cache.
    schedule: str = "sequential"
    output_tile_sites: int = 32
    # Opt-in C++ host/autograd graph; all CUDA arithmetic stages are identical.
    native_autograd: bool = False
    # Validate tensor/configuration/stream flags in the native execution entry.
    native_entry: bool = False


from torchcst._builtin_callbacks import seal_functions as _seal_functions

_seal_functions(globals())
